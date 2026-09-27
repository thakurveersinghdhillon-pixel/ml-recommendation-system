"""Rating Repository Module for database operations on Rating table."""

from datetime import datetime, timezone
from typing import Optional, List, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import select, func

from ml.api.database.models import Rating


class RatingRepository:
    """Repository handling CRUD and upsert operations for Rating model."""

    def __init__(self, db: Session):
        self.db = db

    def upsert(self, user_id: int, movie_id: int, rating_value: float) -> Tuple[Rating, bool]:
        """Atomic create or update user movie rating.

        Returns:
            Tuple of (Rating object, created_boolean).
        """
        stmt = select(Rating).where(
            Rating.user_id == user_id,
            Rating.movie_id == movie_id
        ).with_for_update()

        existing = self.db.scalars(stmt).first()

        now = datetime.now(timezone.utc)
        if existing is not None:
            existing.rating = float(rating_value)
            existing.updated_at = now
            self.db.commit()
            self.db.refresh(existing)
            return existing, False
        else:
            rating_obj = Rating(
                user_id=user_id,
                movie_id=movie_id,
                rating=float(rating_value),
                created_at=now,
                updated_at=now
            )
            self.db.add(rating_obj)
            self.db.commit()
            self.db.refresh(rating_obj)
            return rating_obj, True

    def get_by_user_and_movie(self, user_id: int, movie_id: int) -> Optional[Rating]:
        """Get user rating for a specific movie."""
        stmt = select(Rating).where(
            Rating.user_id == user_id,
            Rating.movie_id == movie_id
        )
        return self.db.scalars(stmt).first()

    def list_by_user(self, user_id: int, limit: int = 20, offset: int = 0) -> List[Rating]:
        """Get paginated ratings for a user ordered deterministically by created_at DESC, id DESC."""
        stmt = (
            select(Rating)
            .where(Rating.user_id == user_id)
            .order_by(Rating.created_at.desc(), Rating.id.desc())
            .offset(offset)
            .limit(limit)
        )
        return list(self.db.scalars(stmt).all())

    def count_by_user(self, user_id: int) -> int:
        """Count total ratings submitted by user."""
        stmt = select(func.count(Rating.id)).where(Rating.user_id == user_id)
        return int(self.db.scalar(stmt) or 0)

    def delete(self, user_id: int, movie_id: int) -> bool:
        """Delete specific rating for a user and movie."""
        rating_obj = self.get_by_user_and_movie(user_id, movie_id)
        if rating_obj is None:
            return False

        self.db.delete(rating_obj)
        self.db.commit()
        return True
