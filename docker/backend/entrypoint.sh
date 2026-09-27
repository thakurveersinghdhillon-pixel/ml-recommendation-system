#!/bin/sh
set -e

echo "Running Alembic database migrations..."
alembic upgrade head

PORT="${PORT:-8000}"
echo "Starting Uvicorn server for FastAPI backend on port ${PORT}..."
exec uvicorn ml.api.main:app --host 0.0.0.0 --port "${PORT}"
