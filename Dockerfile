# FinSight AI — FastAPI Backend (Cloud Run ready)
FROM python:3.11-slim

WORKDIR /app

# System deps occasionally needed by yfinance/pandas wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ api/
COPY agents/ agents/
COPY graph/ graph/

# Cloud Run provides the PORT env var (default 8080). The app must bind to it.
ENV PORT=8080
EXPOSE 8080

# Use shell form so $PORT is expanded at runtime.
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT}
