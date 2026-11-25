FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# run as an unprivileged user
RUN useradd --system --uid 10001 --no-create-home app

COPY --chown=app:app prices_service ./prices_service
# synthetic prices used when PRICE_SOURCE=demo (read by prices_service/sources.py at /app/data)
COPY --chown=app:app data ./data
USER app

EXPOSE 8001
HEALTHCHECK --interval=10s --timeout=4s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=3)"

# One process: the cache lives in memory and is shared by every request.
CMD ["python", "-m", "uvicorn", "prices_service.app:app", "--host", "0.0.0.0", "--port", "8001"]
