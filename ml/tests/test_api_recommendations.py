"""Unit tests for FastAPI recommendations router."""

import pytest
from fastapi.testclient import TestClient
import pandas as pd

from ml.api.main import app
from ml.src.content_based import ContentBasedRecommender
from ml.src.collaborative_filtering import ItemBasedCollaborativeRecommender
from ml.src.hybrid_recommender import HybridRecommender


@pytest.fixture
def mock_app_state():
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


def test_post_content_recommendations_success(mock_app_state):
    """Test POST /api/v1/recommendations/content with valid movie_id."""
    client = TestClient(mock_app_state)
    payload = {"movie_id": 1, "top_k": 3}
    response = client.post("/api/v1/recommendations/content", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "content_based"
    assert data["seed_movie_id"] == 1
    assert data["seed_movie_title"] == "Toy Story (1995)"
    assert len(data["recommendations"]) <= 3


def test_post_content_recommendations_not_found(mock_app_state):
    """Test POST /api/v1/recommendations/content with unknown movie_id returns 404."""
    client = TestClient(mock_app_state)
    payload = {"movie_id": 99999, "top_k": 5}
    response = client.post("/api/v1/recommendations/content", json=payload)

    assert response.status_code == 404
    data = response.json()
    assert data["error"]["code"] == "MOVIE_NOT_FOUND"


def test_post_collaborative_recommendations_success(mock_app_state):
    """Test POST /api/v1/recommendations/collaborative with valid user_id."""
    client = TestClient(mock_app_state)
    payload = {"user_id": 1, "top_k": 2}
    response = client.post("/api/v1/recommendations/collaborative", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "collaborative_filtering"
    assert data["user_id"] == 1
    assert len(data["recommendations"]) <= 2


def test_post_collaborative_recommendations_user_not_found(mock_app_state):
    """Test POST /api/v1/recommendations/collaborative with unknown user_id returns 404."""
    client = TestClient(mock_app_state)
    payload = {"user_id": 99999, "top_k": 5}
    response = client.post("/api/v1/recommendations/collaborative", json=payload)

    assert response.status_code == 404
    data = response.json()
    assert data["error"]["code"] == "USER_NOT_FOUND"


def test_post_hybrid_recommendations_success(mock_app_state):
    """Test POST /api/v1/recommendations/hybrid with default 50/50 weights."""
    client = TestClient(mock_app_state)
    payload = {
        "user_id": 1,
        "movie_id": 1,
        "top_k": 3,
        "content_weight": 0.5,
        "collaborative_weight": 0.5
    }
    response = client.post("/api/v1/recommendations/hybrid", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "hybrid"
    assert data["user_id"] == 1
    assert data["content_weight"] == 0.5
    assert data["collaborative_weight"] == 0.5
    assert len(data["recommendations"]) <= 3


def test_post_hybrid_recommendations_invalid_weights(mock_app_state):
    """Test POST /api/v1/recommendations/hybrid with weights not summing to 1.0 returns 422 validation error."""
    client = TestClient(mock_app_state)
    payload = {
        "user_id": 1,
        "top_k": 3,
        "content_weight": 0.8,
        "collaborative_weight": 0.5
    }
    response = client.post("/api/v1/recommendations/hybrid", json=payload)

    assert response.status_code == 422
    data = response.json()
    assert data["error"]["code"] == "VALIDATION_ERROR"
