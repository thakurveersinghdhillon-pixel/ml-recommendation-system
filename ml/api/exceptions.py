"""Custom exceptions and exception handlers for FastAPI Recommendation API."""

import logging
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import IntegrityError
from ml.api.schemas.common import ErrorResponse, ErrorDetail

logger = logging.getLogger(__name__)


class APIException(Exception):
    """Base API exception with HTTP status code and error details."""
    def __init__(self, status_code: int, code: str, message: str):
        self.status_code = status_code
        self.code = code
        self.message = message
        super().__init__(message)


class MovieNotFoundException(APIException):
    def __init__(self, movie_id_or_title: str | int):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            code="MOVIE_NOT_FOUND",
            message=f"The requested movie '{movie_id_or_title}' does not exist in the dataset catalog."
        )


class UserNotFoundException(APIException):
    def __init__(self, user_id: int):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            code="USER_NOT_FOUND",
            message=f"User ID {user_id} was not found."
        )


class DuplicateUserException(APIException):
    def __init__(self, detail: str = "A user with this username or email already exists."):
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            code="DUPLICATE_USER",
            message=detail
        )


class RatingNotFoundException(APIException):
    def __init__(self, user_id: int, movie_id: int):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            code="RATING_NOT_FOUND",
            message=f"Rating for user ID {user_id} and movie ID {movie_id} was not found."
        )


class DatabaseConnectionException(APIException):
    def __init__(self, detail: str = "Database connection failure or service unavailable."):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            code="DATABASE_UNAVAILABLE",
            message=detail
        )


class ModelNotReadyException(APIException):
    def __init__(self, detail: str = "Recommendation models are not initialized or ready."):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            code="MODEL_NOT_READY",
            message=detail
        )


class InvalidRecommendationRequestException(APIException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="INVALID_RECOMMENDATION_REQUEST",
            message=detail
        )


def register_exception_handlers(app):
    """Register custom exception handlers with FastAPI application."""

    @app.exception_handler(APIException)
    async def api_exception_handler(request: Request, exc: APIException):
        logger.warning("API Exception [%s]: %s (path: %s)", exc.code, exc.message, request.url.path)
        payload = ErrorResponse(error=ErrorDetail(code=exc.code, message=exc.message))
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump())

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        # Format Pydantic validation errors cleanly
        error_msgs = []
        for err in exc.errors():
            loc = " -> ".join([str(x) for x in err.get("loc", []) if x != "body"])
            msg = err.get("msg", "Invalid value")
            error_msgs.append(f"{loc}: {msg}" if loc else msg)
        
        detail_msg = "; ".join(error_msgs) if error_msgs else "Invalid request body or parameters."
        logger.info("Validation Error: %s (path: %s)", detail_msg, request.url.path)
        payload = ErrorResponse(error=ErrorDetail(code="VALIDATION_ERROR", message=detail_msg))
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, content=payload.model_dump())


    @app.exception_handler(IntegrityError)
    async def integrity_error_handler(request: Request, exc: IntegrityError):
        logger.warning("Database Integrity Error: %s (path: %s)", str(exc), request.url.path)
        payload = ErrorResponse(error=ErrorDetail(code="INTEGRITY_ERROR", message="Database integrity constraint violation."))
        return JSONResponse(status_code=status.HTTP_409_CONFLICT, content=payload.model_dump())

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        logger.info("Value Error: %s (path: %s)", str(exc), request.url.path)
        payload = ErrorResponse(error=ErrorDetail(code="INVALID_REQUEST", message=str(exc)))
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=payload.model_dump())

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled Server Error: %s (path: %s)", str(exc), request.url.path, exc_info=True)
        payload = ErrorResponse(error=ErrorDetail(code="INTERNAL_SERVER_ERROR", message="An unexpected error occurred."))
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=payload.model_dump())
