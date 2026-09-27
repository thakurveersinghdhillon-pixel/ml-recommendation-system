"""Recommendation History Repository Module."""

from datetime import datetime, timezone
from typing import Optional, List, Any
from sqlalchemy.orm import Session
from sqlalchemy import select, func

from ml.api.database.models import RecommendationHistory


class RecommendationHistoryRepository:
    """Repository handling persistence and queries for recommendation history."""

    def __init__(self, db: Session):
        self.db = db

    def create_history(
        self,
        user_id: Optional[int],
        model: str,
        seed_movie_id: Optional[int],
        top_k: int,
        content_weight: Optional[float],
        collaborative_weight: Optional[float],
        recommendations: Any
    ) -> RecommendationHistory:
        """Persist successful recommendation request output."""
        history = RecommendationHistory(
            user_id=user_id,
            model=model,
            seed_movie_id=seed_movie_id,
            top_k=top_k,
            content_weight=content_weight,
            collaborative_weight=collaborative_weight,
            created_at=datetime.now(timezone.utc),
            recommendations=recommendations
        )
        self.db.add(history)
        self.db.commit()
        self.db.refresh(history)
        return history

    def list_by_user(self, user_id: int, limit: int = 20, offset: int = 0) -> List[RecommendationHistory]:
        """Get paginated recommendation history for a user ordered by created_at DESC, id DESC."""
        stmt = (
            select(RecommendationHistory)
            .where(RecommendationHistory.user_id == user_id)
            .order_by(RecommendationHistory.created_at.desc(), RecommendationHistory.id.desc())
            .offset(offset)
            .limit(limit)
        )
        return list(self.db.scalars(stmt).all())

    def count_by_user(self, user_id: int) -> int:
        """Count recommendation history records for user."""
        stmt = select(func.count(RecommendationHistory.id)).where(RecommendationHistory.user_id == user_id)
        return int(self.db.scalar(stmt) or 0)
