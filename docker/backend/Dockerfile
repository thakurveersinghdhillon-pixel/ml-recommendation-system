# Production Dockerfile for FastAPI Recommendation Backend
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system utilities (curl for container health check)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY ml/requirements.txt /app/ml/requirements.txt
RUN pip install --no-cache-dir -r ml/requirements.txt

# Copy Alembic database configuration and migration scripts
COPY alembic.ini /app/alembic.ini
COPY alembic /app/alembic

# Copy ML application codebase
COPY ml /app/ml

# Copy and setup entrypoint script
COPY docker/backend/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Create non-root application user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
