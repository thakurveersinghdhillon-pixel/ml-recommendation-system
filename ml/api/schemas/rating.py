"""Rating Pydantic schemas for request validation and response serialization."""

import math
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator


class RatingCreate(BaseModel):
    """Schema for creating or updating a rating."""
    user_id: int = Field(..., gt=0, description="Target User ID (positive integer).", examples=[1])
    movie_id: int = Field(..., gt=0, description="MovieLens Movie ID (positive integer).", examples=[1])
    rating: float = Field(..., ge=0.5, le=5.0, description="Rating value from 0.5 to 5.0 in 0.5 increments.", examples=[4.5])

    @field_validator("rating")
    @classmethod
    def validate_half_step_increment(cls, v: float) -> float:
        if math.isnan(v) or math.isinf(v):
            raise ValueError("Rating value must be a finite number.")
        # Check if 2 * v is an integer (i.e. v is a multiple of 0.5)
        if not math.isclose(round(v * 2), v * 2, abs_tol=1e-5):
            raise ValueError("Rating must be a multiple of 0.5 (e.g. 0.5, 1.0, 1.5, ..., 5.0).")
        return round(float(v), 1)


class RatingUpdate(BaseModel):
    """Schema for updating an existing rating value."""
    rating: float = Field(..., ge=0.5, le=5.0, description="New rating value from 0.5 to 5.0 in 0.5 increments.", examples=[4.0])

    @field_validator("rating")
    @classmethod
    def validate_half_step_increment(cls, v: float) -> float:
        if math.isnan(v) or math.isinf(v):
            raise ValueError("Rating value must be a finite number.")
        if not math.isclose(round(v * 2), v * 2, abs_tol=1e-5):
            raise ValueError("Rating must be a multiple of 0.5 (e.g. 0.5, 1.0, 1.5, ..., 5.0).")
        return round(float(v), 1)


class RatingResponse(BaseModel):
    """Schema for Rating entity response payload."""
    model_config = ConfigDict(from_attributes=True)

    id: int = Field(..., description="Rating record ID integer.", examples=[1])
    user_id: int = Field(..., description="Target User ID integer.", examples=[1])
    movie_id: int = Field(..., description="Movie ID integer.", examples=[1])
    rating: float = Field(..., description="Rating value float.", examples=[4.5])
    created_at: datetime = Field(..., description="Creation timestamp.")
    updated_at: datetime = Field(..., description="Last update timestamp.")


class RatingListResponse(BaseModel):
    """Schema for paginated rating list response."""
    items: List[RatingResponse] = Field(..., description="List of user rating items.")
    total: int = Field(..., description="Total user ratings count.", examples=[10])
    limit: int = Field(..., description="Pagination limit.", examples=[20])
    offset: int = Field(..., description="Pagination offset.", examples=[0])
