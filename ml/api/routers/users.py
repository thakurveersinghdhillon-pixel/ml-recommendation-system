"""Users Router for CRUD operations, User Ratings, and Recommendation History."""

from fastapi import APIRouter, Depends, Query, Response, status
from sqlalchemy.orm import Session

from ml.api.database.session import get_db
from ml.api.services.user_service import UserService
from ml.api.services.rating_service import RatingService
from ml.api.database.repositories.recommendation_repository import RecommendationHistoryRepository
from ml.api.schemas.user import UserCreate, UserUpdate, UserResponse
from ml.api.schemas.rating import RatingResponse, RatingListResponse
from ml.api.schemas.recommendation import (
    RecommendationHistoryResponse,
    RecommendationHistoryListResponse
)
from ml.api.schemas.common import ErrorResponse

router = APIRouter(prefix="/users", tags=["Users"])


@router.post(
    "",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {"model": UserResponse, "description": "User created successfully."},
        400: {"model": ErrorResponse, "description": "Invalid input payload."},
        409: {"model": ErrorResponse, "description": "User with username or email already exists."},
        422: {"model": ErrorResponse, "description": "Validation error in request body."}
    }
)
async def create_user(payload: UserCreate, db: Session = Depends(get_db)):
    """Create a new user entity in database."""
    service = UserService(db)
    return service.create_user(payload)


@router.get(
    "/{user_id}",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"model": UserResponse, "description": "User details returned successfully."},
        404: {"model": ErrorResponse, "description": "User not found."}
    }
)
async def get_user(user_id: int, db: Session = Depends(get_db)):
    """Fetch user by primary key ID."""
    service = UserService(db)
    return service.get_user_by_id(user_id)


@router.patch(
    "/{user_id}",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"model": UserResponse, "description": "User updated successfully."},
        404: {"model": ErrorResponse, "description": "User not found."},
        409: {"model": ErrorResponse, "description": "Username or email already in use."}
    }
)
async def update_user(user_id: int, payload: UserUpdate, db: Session = Depends(get_db)):
    """Update user attributes (username/email)."""
    service = UserService(db)
    return service.update_user(user_id, payload)


@router.delete(
    "/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        204: {"description": "User deleted successfully."},
        404: {"model": ErrorResponse, "description": "User not found."}
    }
)
async def delete_user(user_id: int, db: Session = Depends(get_db)):
    """Delete user by ID and cascade ratings/history."""
    service = UserService(db)
    service.delete_user(user_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/{user_id}/ratings",
    response_model=RatingListResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"model": RatingListResponse, "description": "List of user ratings."},
        404: {"model": ErrorResponse, "description": "User not found."}
    }
)
async def list_user_ratings(
    user_id: int,
    limit: int = Query(20, ge=1, le=100, description="Pagination limit (1 to 100)."),
    offset: int = Query(0, ge=0, description="Pagination offset."),
    db: Session = Depends(get_db)
):
    """List paginated movie ratings for a target user."""
    service = RatingService(db)
    return service.list_user_ratings(user_id, limit=limit, offset=offset)


@router.get(
    "/{user_id}/ratings/{movie_id}",
    response_model=RatingResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"model": RatingResponse, "description": "User movie rating details."},
        404: {"model": ErrorResponse, "description": "User or rating not found."}
    }
)
async def get_user_rating(user_id: int, movie_id: int, db: Session = Depends(get_db)):
    """Fetch rating for a specific movie by a specific user."""
    service = RatingService(db)
    return service.get_user_rating(user_id, movie_id)


@router.delete(
    "/{user_id}/ratings/{movie_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        204: {"description": "User rating deleted successfully."},
        404: {"model": ErrorResponse, "description": "User or rating not found."}
    }
)
async def delete_user_rating(user_id: int, movie_id: int, db: Session = Depends(get_db)):
    """Delete specific rating for a user and movie."""
    service = RatingService(db)
    service.delete_user_rating(user_id, movie_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/{user_id}/recommendation-history",
    response_model=RecommendationHistoryListResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"model": RecommendationHistoryListResponse, "description": "Recommendation history list for user."},
        404: {"model": ErrorResponse, "description": "User not found."}
    }
)
async def get_user_recommendation_history(
    user_id: int,
    limit: int = Query(20, ge=1, le=100, description="Pagination limit (1 to 100)."),
    offset: int = Query(0, ge=0, description="Pagination offset."),
    db: Session = Depends(get_db)
):
    """List recommendation request history logged for a user."""
    user_service = UserService(db)
    user_service.get_user_by_id(user_id)  # Ensure user exists or raise 404

    history_repo = RecommendationHistoryRepository(db)
    items = history_repo.list_by_user(user_id, limit=limit, offset=offset)
    total = history_repo.count_by_user(user_id)

    history_responses = [RecommendationHistoryResponse.model_validate(h) for h in items]

    return RecommendationHistoryListResponse(
        items=history_responses,
        total=total,
        limit=limit,
        offset=offset
    )
