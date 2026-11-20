# API image: FastAPI + SARIMAX serving model.
#
#   docker build -t oil-api .
#   docker run -p 8000:8000 -e GROQ_API_KEY=... oil-api
#
# Secrets are passed at run time, never baked into the image.

FROM python:3.11.8-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, so code changes reuse this cached layer. The lock file
# pins every transitive package to an exact version and hash.
COPY requirements.lock .
RUN pip install --require-hashes -r requirements.lock

# Only what the API needs at run time.
COPY app.py scenario_engine.py llm_explainer.py scenario_library.py modeling.py utils.py train.py ./
COPY serving/ serving/
COPY data/oil_macro_weekly.csv data/oil_macro_transformed.csv data/scenario_events.csv data/

# Fit the model at build time and keep only the small serving copy (~3 MB),
# so containers start in seconds instead of refitting on every start.
RUN python train.py

RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:' + __import__('os').environ.get('PORT', '8000') + '/health', timeout=4).status == 200 else 1)"

# PORT lets platforms like Render choose the port.
CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
