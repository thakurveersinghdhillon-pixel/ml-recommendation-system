"""Rating Service Module implementing business logic and validation for User Ratings."""

import logging
from typing import Tuple
from sqlalchemy.orm import Session

from ml.api.database.repositories.user_repository import UserRepository
from ml.api.database.repositories.rating_repository import RatingRepository
from ml.api.schemas.rating import RatingCreate, RatingResponse, RatingListResponse
from ml.api.exceptions import UserNotFoundException, RatingNotFoundException

logger = logging.getLogger(__name__)


class RatingService:
    """Service layer for Rating entity operations."""

    def __init__(self, db: Session):
        self.db = db
        self.user_repo = UserRepository(db)
        self.rating_repo = RatingRepository(db)

    def upsert_rating(self, payload: RatingCreate) -> Tuple[RatingResponse, bool]:
        """Create or update user rating atomically.
        
        Returns:
            Tuple of (RatingResponse schema, created_boolean).
        """
        user = self.user_repo.get_by_id(payload.user_id)
        if user is None:
            raise UserNotFoundException(payload.user_id)

        rating_obj, is_created = self.rating_repo.upsert(
            user_id=payload.user_id,
            movie_id=payload.movie_id,
            rating_value=payload.rating
        )
        return RatingResponse.model_validate(rating_obj), is_created

    def get_user_rating(self, user_id: int, movie_id: int) -> RatingResponse:
        """Fetch rating for a given user and movie."""
        user = self.user_repo.get_by_id(user_id)
        if user is None:
            raise UserNotFoundException(user_id)

        rating_obj = self.rating_repo.get_by_user_and_movie(user_id, movie_id)
        if rating_obj is None:
            raise RatingNotFoundException(user_id, movie_id)

        return RatingResponse.model_validate(rating_obj)

    def list_user_ratings(self, user_id: int, limit: int = 20, offset: int = 0) -> RatingListResponse:
        """Fetch paginated ratings list for a user."""
        user = self.user_repo.get_by_id(user_id)
        if user is None:
            raise UserNotFoundException(user_id)

        items = self.rating_repo.list_by_user(user_id, limit=limit, offset=offset)
        total = self.rating_repo.count_by_user(user_id)

        return RatingListResponse(
            items=[RatingResponse.model_validate(r) for r in items],
            total=total,
            limit=limit,
            offset=offset
        )

    def delete_user_rating(self, user_id: int, movie_id: int) -> None:
        """Delete user rating for a movie."""
        user = self.user_repo.get_by_id(user_id)
        if user is None:
            raise UserNotFoundException(user_id)

        success = self.rating_repo.delete(user_id, movie_id)
        if not success:
            raise RatingNotFoundException(user_id, movie_id)
