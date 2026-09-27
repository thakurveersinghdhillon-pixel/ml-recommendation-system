"""Common API Pydantic schemas for health and error responses."""

from typing import Dict
from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Detailed error object."""
    code: str = Field(..., description="Machine-readable error code string.", examples=["MOVIE_NOT_FOUND"])
    message: str = Field(..., description="Human-readable error explanation message.", examples=["Movie ID 99999 not found."])


class ErrorResponse(BaseModel):
    """Standardized error response payload."""
    error: ErrorDetail


class ModelHealthStatus(BaseModel):
    """Health status for individual recommendation models."""
    content_based: str = Field(..., examples=["ready"])
    collaborative: str = Field(..., examples=["ready"])
    hybrid: str = Field(..., examples=["ready"])


class HealthResponse(BaseModel):
    """Overall API health check response payload."""
    status: str = Field(..., examples=["healthy"])
    service: str = Field(..., examples=["movie-recommendation-api"])
    database: str = Field("healthy", examples=["healthy"])
    models: ModelHealthStatus
