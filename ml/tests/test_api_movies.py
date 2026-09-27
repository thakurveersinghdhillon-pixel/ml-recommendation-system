"""Unit tests for FastAPI movies router."""

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


def test_get_movie_details_success(mock_app_state):
    """Test GET /api/v1/movies/{movie_id} with valid movie ID."""
    client = TestClient(mock_app_state)
    response = client.get("/api/v1/movies/1")

    assert response.status_code == 200
    data = response.json()
    assert data["movie_id"] == 1
    assert data["title"] == "Toy Story (1995)"
    assert "Animation" in data["genres"]


def test_get_movie_details_not_found(mock_app_state):
    """Test GET /api/v1/movies/{movie_id} with unknown movie ID returns 404."""
    client = TestClient(mock_app_state)
    response = client.get("/api/v1/movies/99999")

    assert response.status_code == 404
    data = response.json()
    assert data["error"]["code"] == "MOVIE_NOT_FOUND"


def test_search_movies_success(mock_app_state):
    """Test GET /api/v1/movies/search with matching query."""
    client = TestClient(mock_app_state)
    response = client.get("/api/v1/movies/search?q=toy&limit=5")

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "toy"
    assert data["total_matches"] == 1
    assert data["movies"][0]["movie_id"] == 1


def test_search_movies_empty_results(mock_app_state):
    """Test GET /api/v1/movies/search with non-matching query returns empty list."""
    client = TestClient(mock_app_state)
    response = client.get("/api/v1/movies/search?q=nonexistentmoviequery")

    assert response.status_code == 200
    data = response.json()
    assert data["total_matches"] == 0
    assert data["movies"] == []
