"""Production Request Logging Middleware for FastAPI Backend."""

import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, Response

logger = logging.getLogger("ml.api.request")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests with status codes and execution time without leaking sensitive payload data."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.perf_counter()
        
        try:
            response = await call_next(request)
            process_time_ms = (time.perf_counter() - start_time) * 1000
            
            # Log request path, method, status code, and execution time
            logger.info(
                "HTTP %s %s -> Status %d (%.2f ms)",
                request.method,
                request.url.path,
                response.status_code,
                process_time_ms,
            )
            response.headers["X-Process-Time"] = f"{process_time_ms:.2f}ms"
            return response

        except Exception as exc:
            process_time_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "HTTP %s %s -> Exception: %s (%.2f ms)",
                request.method,
                request.url.path,
                str(exc),
                process_time_ms,
                exc_info=True,
            )
            raise exc
