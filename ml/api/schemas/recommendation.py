"""Recommendation domain Pydantic schemas."""

from datetime import datetime
from typing import List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator


class ContentRecommendationRequest(BaseModel):
    """Content-based recommendation request payload."""
    movie_id: int = Field(..., gt=0, description="Seed movie ID (must be a positive integer).", examples=[1])
    top_k: int = Field(10, ge=1, le=50, description="Number of recommendations to return (1 to 50).", examples=[10])


class CollaborativeRecommendationRequest(BaseModel):
    """Collaborative filtering recommendation request payload."""
    user_id: int = Field(..., gt=0, description="Target user ID (must be a positive integer).", examples=[1])
    top_k: int = Field(10, ge=1, le=50, description="Number of recommendations to return (1 to 50).", examples=[10])


class HybridRecommendationRequest(BaseModel):
    """Hybrid recommendation request payload."""
    user_id: Optional[int] = Field(None, gt=0, description="Target user ID (positive integer) or None for unauthenticated user.", examples=[1])
    movie_id: Optional[int] = Field(None, gt=0, description="Optional seed movie ID for context / cold-start fallback.", examples=[1])
    top_k: int = Field(10, ge=1, le=50, description="Number of recommendations to return (1 to 50).", examples=[10])
    content_weight: float = Field(0.5, ge=0.0, le=1.0, description="Weight for content signal (must be >= 0.0).", examples=[0.5])
    collaborative_weight: float = Field(0.5, ge=0.0, le=1.0, description="Weight for collaborative signal (must be >= 0.0).", examples=[0.5])

    @field_validator("collaborative_weight")
    @classmethod
    def validate_weights_sum(cls, v: float, info) -> float:
        content_w = info.data.get("content_weight", 0.5)
        weight_sum = content_w + v
        if abs(weight_sum - 1.0) > 1e-5:
            raise ValueError(f"content_weight ({content_w}) and collaborative_weight ({v}) must sum to 1.0.")
        return v


class RecommendationItem(BaseModel):
    """Individual movie recommendation item schema."""
    movie_id: int = Field(..., description="Recommended Movie ID.", examples=[3114])
    title: str = Field(..., description="Recommended movie title.", examples=["Toy Story 2 (1999)"])
    genres: List[str] = Field(..., description="List of genre labels.", examples=[["Adventure", "Animation", "Children", "Comedy", "Fantasy"]])
    score: float = Field(..., description="Model recommendation or similarity score.", examples=[0.7337])
    content_score: Optional[float] = Field(None, description="Raw content similarity score if available.", examples=[0.7337])
    collaborative_score: Optional[float] = Field(None, description="Raw collaborative score if available.", examples=[0.6336])
    normalized_content_score: Optional[float] = Field(None, description="Normalized content score [0, 1].", examples=[1.0])
    normalized_collaborative_score: Optional[float] = Field(None, description="Normalized collaborative score [0, 1].", examples=[0.8])
    explanation: Optional[str] = Field(None, description="Model-generated explanation string if supported.")


class RecommendationResponse(BaseModel):
    """Recommendation response payload."""
    model: str = Field(..., description="Name of recommendation model used.", examples=["content_based"])
    user_id: Optional[int] = Field(None, description="Target user ID if applicable.", examples=[1])
    seed_movie_id: Optional[int] = Field(None, description="Seed movie ID if applicable.", examples=[1])
    seed_movie_title: Optional[str] = Field(None, description="Seed movie title if applicable.", examples=["Toy Story (1995)"])
    content_weight: Optional[float] = Field(None, description="Content weight used if hybrid.", examples=[0.5])
    collaborative_weight: Optional[float] = Field(None, description="Collaborative weight used if hybrid.", examples=[0.5])
    recommendations: List[RecommendationItem] = Field(..., description="List of top K recommendation items.")


class RecommendationHistoryResponse(BaseModel):
    """Schema for RecommendationHistory record response."""
    model_config = ConfigDict(from_attributes=True)

    id: int = Field(..., description="History record ID integer.", examples=[1])
    user_id: Optional[int] = Field(None, description="User ID if associated.", examples=[1])
    model: str = Field(..., description="Name of recommendation model used.", examples=["hybrid"])
    seed_movie_id: Optional[int] = Field(None, description="Seed movie ID if provided.", examples=[1])
    top_k: int = Field(..., description="Top K recommendations count requested.", examples=[10])
    content_weight: Optional[float] = Field(None, description="Content weight used.", examples=[0.5])
    collaborative_weight: Optional[float] = Field(None, description="Collaborative weight used.", examples=[0.5])
    created_at: datetime = Field(..., description="Request creation timestamp.")
    recommendations: Any = Field(..., description="Stored list of recommendation dictionaries.")


class RecommendationHistoryListResponse(BaseModel):
    """Schema for paginated recommendation history list response."""
    items: List[RecommendationHistoryResponse] = Field(..., description="List of recommendation history items.")
    total: int = Field(..., description="Total recommendation history count.", examples=[5])
    limit: int = Field(..., description="Pagination limit.", examples=[20])
    offset: int = Field(..., description="Pagination offset.", examples=[0])
