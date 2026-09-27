"""Recommendations Router for Content, Collaborative, and Hybrid models."""

from fastapi import APIRouter, Depends, status
from ml.api.dependencies import get_recommendation_service
from ml.api.services.recommendation_service import RecommendationService
from ml.api.schemas.recommendation import (
    ContentRecommendationRequest,
    CollaborativeRecommendationRequest,
    HybridRecommendationRequest,
    RecommendationResponse
)
from ml.api.schemas.common import ErrorResponse

router = APIRouter(prefix="/recommendations", tags=["Recommendations"])


@router.post(
    "/content",
    response_model=RecommendationResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"model": RecommendationResponse, "description": "Successful content-based recommendations."},
        400: {"model": ErrorResponse, "description": "Invalid recommendation parameters."},
        404: {"model": ErrorResponse, "description": "Seed movie not found in dataset catalog."},
        422: {"model": ErrorResponse, "description": "Validation error in request payload."}
    }
)
async def get_content_recommendations(
    request: ContentRecommendationRequest,
    service: RecommendationService = Depends(get_recommendation_service)
):
    """Generate Content-Based Movie Recommendations using TF-IDF and Cosine Similarity."""
    return service.get_content_recommendations(movie_id=request.movie_id, top_k=request.top_k)


@router.post(
    "/collaborative",
    response_model=RecommendationResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"model": RecommendationResponse, "description": "Successful collaborative filtering recommendations."},
        400: {"model": ErrorResponse, "description": "Invalid recommendation parameters."},
        404: {"model": ErrorResponse, "description": "User ID not found in training dataset."},
        422: {"model": ErrorResponse, "description": "Validation error in request payload."}
    }
)
async def get_collaborative_recommendations(
    request: CollaborativeRecommendationRequest,
    service: RecommendationService = Depends(get_recommendation_service)
):
    """Generate Item-Based Collaborative Filtering Recommendations for a User."""
    return service.get_collaborative_recommendations(user_id=request.user_id, top_k=request.top_k)


@router.post(
    "/hybrid",
    response_model=RecommendationResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"model": RecommendationResponse, "description": "Successful hybrid recommendations."},
        400: {"model": ErrorResponse, "description": "Invalid weights or request parameters."},
        404: {"model": ErrorResponse, "description": "User ID or seed movie not found."},
        422: {"model": ErrorResponse, "description": "Validation error in request payload."}
    }
)
async def get_hybrid_recommendations(
    request: HybridRecommendationRequest,
    service: RecommendationService = Depends(get_recommendation_service)
):
    """Generate Hybrid Recommendations combining Content-Based and Collaborative Filtering signals."""
    return service.get_hybrid_recommendations(
        user_id=request.user_id,
        movie_id=request.movie_id,
        top_k=request.top_k,
        content_weight=request.content_weight,
        collaborative_weight=request.collaborative_weight
    )
