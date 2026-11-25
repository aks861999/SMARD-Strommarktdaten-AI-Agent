FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# run as an unprivileged user
RUN useradd --system --uid 10001 --no-create-home app

COPY --chown=app:app *.py catalog.json ./
COPY --chown=app:app energy_agent ./energy_agent
USER app

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=4s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"

# One process on purpose: running jobs and their replayable event logs live in memory.
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
