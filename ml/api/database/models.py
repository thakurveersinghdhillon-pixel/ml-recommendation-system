"""SQLAlchemy 2.x ORM Models for Users, Ratings, and Recommendation History."""

from datetime import datetime, timezone
from typing import List, Optional, Any
from sqlalchemy import (
    Integer, String, Float, DateTime, ForeignKey,
    UniqueConstraint, CheckConstraint, JSON
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from ml.api.database.base import Base


def utc_now() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)


class User(Base):
    """User ORM Model representing application users."""
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    # Relationships
    ratings: Mapped[List["Rating"]] = relationship("Rating", back_populates="user", cascade="all, delete-orphan")
    recommendation_history: Mapped[List["RecommendationHistory"]] = relationship("RecommendationHistory", back_populates="user")


class Rating(Base):
    """Rating ORM Model representing user movie ratings."""
    __tablename__ = "ratings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    movie_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    rating: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    # Table constraints
    __table_args__ = (
        UniqueConstraint("user_id", "movie_id", name="uq_user_movie_rating"),
        CheckConstraint("rating >= 0.5 AND rating <= 5.0", name="ck_rating_range"),
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="ratings")


class RecommendationHistory(Base):
    """RecommendationHistory ORM Model representing logged recommendation requests."""
    __tablename__ = "recommendation_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    model: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    seed_movie_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    top_k: Mapped[int] = mapped_column(Integer, nullable=False)
    content_weight: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    collaborative_weight: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False, index=True)
    recommendations: Mapped[Any] = mapped_column(JSON, nullable=False)

    # Relationships
    user: Mapped[Optional["User"]] = relationship("User", back_populates="recommendation_history")
