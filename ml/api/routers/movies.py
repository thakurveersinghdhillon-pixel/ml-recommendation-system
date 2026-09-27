"""Movies Router for Search and Details Lookup."""

from fastapi import APIRouter, Depends, Query, status
from ml.api.dependencies import get_recommendation_service
from ml.api.services.recommendation_service import RecommendationService
from ml.api.schemas.movie import MovieResponse, MovieSearchResponse
from ml.api.schemas.common import ErrorResponse

router = APIRouter(prefix="/movies", tags=["Movies"])


@router.get(
    "/search",
    response_model=MovieSearchResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"model": MovieSearchResponse, "description": "Successful movie title search results."},
        400: {"model": ErrorResponse, "description": "Invalid query parameters."},
        422: {"model": ErrorResponse, "description": "Validation error."}
    }
)
async def search_movies(
    q: str = Query(..., description="Movie title or partial title query string.", examples=["toy"]),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of search results to return.", examples=[10]),
    service: RecommendationService = Depends(get_recommendation_service)
):
    """Search for movies by case-insensitive partial title query."""
    return service.search_movies(query=q, limit=limit)


@router.get(
    "/{movie_id}",
    response_model=MovieResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"model": MovieResponse, "description": "Successful movie details retrieval."},
        404: {"model": ErrorResponse, "description": "Movie ID not found in dataset catalog."},
        422: {"model": ErrorResponse, "description": "Invalid movie_id parameter."}
    }
)
async def get_movie_details(
    movie_id: int,
    service: RecommendationService = Depends(get_recommendation_service)
):
    """Get metadata details for a specific movie by ID."""
    return service.get_movie_details(movie_id=movie_id)
