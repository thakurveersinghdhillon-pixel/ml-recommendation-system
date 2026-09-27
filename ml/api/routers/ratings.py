"""Ratings Router for creating and updating user movie ratings."""

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.orm import Session

from ml.api.database.session import get_db
from ml.api.services.rating_service import RatingService
from ml.api.schemas.rating import RatingCreate, RatingResponse
from ml.api.schemas.common import ErrorResponse

router = APIRouter(prefix="/ratings", tags=["Ratings"])


@router.put(
    "",
    response_model=RatingResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"model": RatingResponse, "description": "Rating updated successfully."},
        201: {"model": RatingResponse, "description": "Rating created successfully."},
        404: {"model": ErrorResponse, "description": "Target User not found."},
        422: {"model": ErrorResponse, "description": "Validation error in rating payload."}
    }
)
async def upsert_rating(
    payload: RatingCreate,
    response: Response,
    db: Session = Depends(get_db)
):
    """Create or update a user rating for a movie (upsert)."""
    service = RatingService(db)
    rating_resp, is_created = service.upsert_rating(payload)
    if is_created:
        response.status_code = status.HTTP_201_CREATED
    else:
        response.status_code = status.HTTP_200_OK
    return rating_resp
