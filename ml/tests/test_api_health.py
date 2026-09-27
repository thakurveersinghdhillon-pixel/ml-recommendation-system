"""Unit tests for FastAPI health check router."""

import pytest
from fastapi.testclient import TestClient
import pandas as pd

from ml.api.main import app
from ml.src.content_based import ContentBasedRecommender
from ml.src.collaborative_filtering import ItemBasedCollaborativeRecommender
from ml.src.hybrid_recommender import HybridRecommender


@pytest.fixture
def mock_app_state():
    """Fixture initializing fitted synthetic models on app.state for fast API testing."""
    movies_df = pd.DataFrame({
        "movieId": [1, 2, 3, 4, 5],
        "title": ["Toy Story (1995)", "Jumanji (1995)", "Grumpier Old Men (1995)", "Waiting to Exhale (1995)", "Father of the Bride Part II (1995)"],
        "genres": ["Adventure|Animation", "Adventure|Children", "Comedy|Romance", "Comedy|Drama", "Comedy"]
    })
    ratings_df = pd.DataFrame({
        "userId": [1, 1, 1, 2, 2, 3, 3],
        "movieId": [1, 2, 3, 1, 4, 2, 4],
        "rating": [5.0, 4.0, 2.0, 4.5, 5.0, 4.0, 1.0],
        "timestamp": [1, 2, 3, 4, 5, 6, 7]
    })

    c_rec = ContentBasedRecommender().fit(movies_df)
    cf_rec = ItemBasedCollaborativeRecommender().fit(ratings_df, movies_df=movies_df)
    h_rec = HybridRecommender(c_rec, cf_rec, 0.5, 0.5)

    app.state.movies_df = movies_df
    app.state.content_recommender = c_rec
    app.state.collaborative_recommender = cf_rec
    app.state.hybrid_recommender = h_rec

    yield app

    app.state.movies_df = None
    app.state.content_recommender = None
    app.state.collaborative_recommender = None
    app.state.hybrid_recommender = None


def test_health_check_healthy(mock_app_state):
    """Test GET /api/v1/health when all models are ready."""
    client = TestClient(mock_app_state)
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "movie-recommendation-api"
    assert data["models"]["content_based"] == "ready"
    assert data["models"]["collaborative"] == "ready"
    assert data["models"]["hybrid"] == "ready"


def test_health_check_unhealthy(mock_app_state):
    """Test GET /api/v1/health when models are uninitialized."""
    mock_app_state.state.content_recommender = None
    client = TestClient(mock_app_state)

    response = client.get("/api/v1/health")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "unhealthy"
    assert data["models"]["content_based"] == "not_ready"
