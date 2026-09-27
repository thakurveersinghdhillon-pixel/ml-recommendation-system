"""Unit tests for FastAPI Pydantic request validation."""

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
        "movieId": [1, 2, 3],
        "title": ["Toy Story (1995)", "Jumanji (1995)", "Grumpier Old Men (1995)"],
        "genres": ["Adventure|Animation", "Adventure|Children", "Comedy|Romance"]
    })
    ratings_df = pd.DataFrame({
        "userId": [1, 1, 2],
        "movieId": [1, 2, 1],
        "rating": [5.0, 4.0, 4.5],
        "timestamp": [1, 2, 3]
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


def test_validation_negative_movie_id(mock_app_state):
    """Test negative movie_id in POST /recommendations/content returns 422."""
    client = TestClient(mock_app_state)
    response = client.post("/api/v1/recommendations/content", json={"movie_id": -5, "top_k": 10})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_validation_out_of_bounds_top_k(mock_app_state):
    """Test top_k > 50 in POST /recommendations/content returns 422."""
    client = TestClient(mock_app_state)
    response = client.post("/api/v1/recommendations/content", json={"movie_id": 1, "top_k": 100})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_validation_zero_top_k(mock_app_state):
    """Test top_k = 0 in POST /recommendations/collaborative returns 422."""
    client = TestClient(mock_app_state)
    response = client.post("/api/v1/recommendations/collaborative", json={"user_id": 1, "top_k": 0})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_validation_invalid_search_limit(mock_app_state):
    """Test limit > 50 in GET /movies/search returns 422."""
    client = TestClient(mock_app_state)
    response = client.get("/api/v1/movies/search?q=toy&limit=100")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_validation_negative_weights(mock_app_state):
    """Test negative content_weight in POST /recommendations/hybrid returns 422."""
    client = TestClient(mock_app_state)
    payload = {"user_id": 1, "top_k": 5, "content_weight": -0.5, "collaborative_weight": 1.5}
    response = client.post("/api/v1/recommendations/hybrid", json=payload)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
