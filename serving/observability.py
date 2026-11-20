"""Structured logging, request IDs and in-memory metrics.

Every request is logged as ONE JSON line (request id, method, path, status,
latency, cache result) so logs can be searched and aggregated instead of read
by eye. Every response carries an X-Request-ID header, so a user's report
("this answer looked wrong") can be matched to the exact log line.

Metrics are kept in memory: enough for a single free-tier instance and for
/metrics to answer "how slow, how often failing, how often degraded?".
A multi-instance deployment would export them to Prometheus instead.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import threading
import time
import uuid
from collections import defaultdict, deque

import numpy as np

REQUEST_ID_HEADER = "X-Request-ID"
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
LATENCY_WINDOW = 1000  # most recent requests per route used for percentiles


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + "Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("oil_api")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


def log_event(logger: logging.Logger, msg: str, level: int = logging.INFO, **fields) -> None:
    logger.log(level, msg, extra={"fields": fields})


def request_id_from(header_value: str | None) -> str:
    """Reuse a caller's request id if it is safe to log; otherwise mint one."""
    if header_value and _SAFE_ID.match(header_value):
        return header_value
    return uuid.uuid4().hex[:16]


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.started = time.time()
        self.requests: dict[tuple[str, int], int] = defaultdict(int)
        self.latency: dict[str, deque] = defaultdict(lambda: deque(maxlen=LATENCY_WINDOW))
        self.llm_fallbacks: dict[str, int] = defaultdict(int)

    def observe(self, route: str, status: int, latency_ms: float) -> None:
        with self._lock:
            self.requests[(route, status)] += 1
            self.latency[route].append(latency_ms)

    def record_fallbacks(self, kinds: list[str]) -> None:
        with self._lock:
            for k in kinds:
                self.llm_fallbacks[k] += 1

    def snapshot(self) -> dict:
        with self._lock:
            routes = {}
            for route in sorted({r for r, _ in self.requests}):
                by_status = {str(s): n for (r, s), n in sorted(self.requests.items()) if r == route}
                lat = np.asarray(self.latency[route], dtype=float)
                routes[route] = {
                    "requests": sum(by_status.values()),
                    "by_status": by_status,
                    "errors_5xx": sum(n for s, n in by_status.items() if s.startswith("5")),
                    "latency_ms": {
                        "p50": round(float(np.percentile(lat, 50)), 1) if lat.size else None,
                        "p95": round(float(np.percentile(lat, 95)), 1) if lat.size else None,
                        "window": int(lat.size),
                    },
                }
            return {
                "uptime_seconds": round(time.time() - self.started, 1),
                "routes": routes,
                "llm_fallbacks": dict(self.llm_fallbacks),
            }
