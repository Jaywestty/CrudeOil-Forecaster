# Streamlit UI image. Build from the repository root:
#
#   docker build -f streamlit_ui/Dockerfile -t oil-ui .
#   docker run -p 8501:8501 -e API_URL=http://host.docker.internal:8000 oil-ui

FROM python:3.11.8-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

COPY streamlit_ui/requirements.lock .
RUN pip install --require-hashes -r requirements.lock

COPY streamlit_ui/app_ui.py streamlit_ui/app_ui.py
COPY .streamlit/ .streamlit/

RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).status == 200 else 1)"

CMD ["streamlit", "run", "streamlit_ui/app_ui.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
