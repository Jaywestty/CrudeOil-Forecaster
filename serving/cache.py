"""A small thread-safe LRU cache with a time-to-live, for simulation responses.

A simulation costs several LLM calls and one SARIMAX run per scenario, and the
UI's example buttons send identical questions. Caching identical requests
saves that work and makes repeated questions return the same answer.

The key includes everything that can change an answer: the normalised
question, the horizon, the data version and the LLM model. New data or a
model swap therefore never serves an old answer.
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
from collections import OrderedDict
from typing import Any, Callable


def normalise_query(query: str) -> str:
    """Case- and whitespace-insensitive form, so trivial edits share a cache entry."""
    return re.sub(r"\s+", " ", query.strip().lower())


def make_key(query: str, weeks: int, data_version: str, model: str) -> str:
    raw = "\x1f".join([normalise_query(query), str(weeks), data_version, model])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class TTLCache:
    def __init__(self, maxsize: int = 256, ttl_seconds: float = 6 * 3600,
                 clock: Callable[[], float] = time.monotonic):
        if maxsize < 1 or ttl_seconds <= 0:
            raise ValueError("maxsize must be >= 1 and ttl_seconds > 0")
        self.maxsize, self.ttl, self._clock = maxsize, ttl_seconds, clock
        self._items: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = self.misses = self.evictions = 0

    def get(self, key: str) -> Any | None:
        with self._lock:
            item = self._items.get(key)
            if item is None:
                self.misses += 1
                return None
            expires_at, value = item
            if self._clock() >= expires_at:
                del self._items[key]
                self.misses += 1
                return None
            self._items.move_to_end(key)  # most recently used
            self.hits += 1
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._items[key] = (self._clock() + self.ttl, value)
            self._items.move_to_end(key)
            while len(self._items) > self.maxsize:
                self._items.popitem(last=False)  # least recently used
                self.evictions += 1

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            return {
                "size": len(self._items), "maxsize": self.maxsize, "ttl_seconds": self.ttl,
                "hits": self.hits, "misses": self.misses, "evictions": self.evictions,
                "hit_rate": round(self.hits / total, 3) if total else None,
            }
