# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/app/data \
    KNOWN_FACES_DIR=/app/known_faces

WORKDIR /app

# Install native runtime and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    binutils \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install Python requirements and InsightFace with stripped shared objects to keep image slim
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-deps "insightface>=0.7.3" && \
    find /usr/local/lib/python3.11/site-packages -name "*.so*" -exec strip --strip-unneeded {} + 2>/dev/null || true && \
    apt-get purge -y binutils && \
    apt-get autoremove -y && \
    find /usr/local -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    rm -rf /root/.cache /tmp/*

# Copy application source code
COPY app /app/app

# Ensure runtime directories exist
RUN mkdir -p /app/known_faces /app/data /app/data/history

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
