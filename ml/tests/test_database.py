"""Unit tests for SQLAlchemy Database ORM Models, Session, and Health Check."""

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError

from ml.api.database.base import Base
from ml.api.database.models import User, Rating, RecommendationHistory
from ml.api.database.session import check_db_health


@pytest.fixture
def db_session():
    """Create in-memory SQLite engine and session for unit testing ORM models."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    Base.metadata.drop_all(bind=engine)


def test_db_health_check(db_session):
    """Verify check_db_health returns True for an active database session."""
    assert check_db_health(db_session) is True


def test_user_orm_creation(db_session):
    """Test creating and querying User ORM record."""
    user = User(username="testuser", email="test@example.com")
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    assert user.id is not None
    assert user.username == "testuser"
    assert user.email == "test@example.com"
    assert user.created_at is not None
    assert user.updated_at is not None


def test_user_unique_constraints(db_session):
    """Test unique username constraint on User ORM model."""
    u1 = User(username="unique_name", email="u1@example.com")
    db_session.add(u1)
    db_session.commit()

    u2 = User(username="unique_name", email="u2@example.com")
    db_session.add(u2)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_rating_orm_creation_and_cascade(db_session):
    """Test creating Rating ORM model linked to User and delete cascade behavior."""
    user = User(username="rater", email="rater@example.com")
    db_session.add(user)
    db_session.commit()

    rating = Rating(user_id=user.id, movie_id=101, rating=4.5)
    db_session.add(rating)
    db_session.commit()

    assert rating.id is not None
    assert rating.user_id == user.id
    assert len(user.ratings) == 1

    # Test Cascade Delete on User removal
    db_session.delete(user)
    db_session.commit()

    ratings_in_db = db_session.scalars(select(Rating).where(Rating.user_id == user.id)).all()
    assert len(ratings_in_db) == 0


def test_recommendation_history_orm_creation(db_session):
    """Test creating RecommendationHistory ORM model record."""
    user = User(username="history_user", email="history@example.com")
    db_session.add(user)
    db_session.commit()

    history = RecommendationHistory(
        user_id=user.id,
        model="hybrid",
        seed_movie_id=1,
        top_k=5,
        content_weight=0.5,
        collaborative_weight=0.5,
        recommendations=[{"movie_id": 10, "score": 0.95}]
    )
    db_session.add(history)
    db_session.commit()

    assert history.id is not None
    assert history.user_id == user.id
    assert history.model == "hybrid"
    assert history.recommendations == [{"movie_id": 10, "score": 0.95}]
