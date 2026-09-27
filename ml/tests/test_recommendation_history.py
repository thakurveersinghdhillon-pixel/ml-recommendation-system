"""Unit tests for Recommendation History logging and query endpoints."""

import pytest


@pytest.fixture
def test_user(client):
    response = client.post("/api/v1/users", json={"username": "history_tester", "email": "hist@example.com"})
    return response.json()


def test_get_recommendation_history_empty(client, test_user):
    """Test GET /api/v1/users/{id}/recommendation-history returns empty list when no history exists."""
    user_id = test_user["id"]
    response = client.get(f"/api/v1/users/{user_id}/recommendation-history")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0
    assert data["items"] == []


def test_recommendation_request_logs_history(client, test_user):
    """Test POST /api/v1/recommendations/hybrid logs a history record when user_id is provided."""
    user_id = test_user["id"]

    rec_resp = client.post(
        "/api/v1/recommendations/hybrid",
        json={
            "user_id": 1,  # ML dataset user 1
            "movie_id": 1,
            "top_k": 5,
            "content_weight": 0.5,
            "collaborative_weight": 0.5
        }
    )
    assert rec_resp.status_code == 200

    # Also log history associated directly with test_user
    rec_resp2 = client.post(
        "/api/v1/recommendations/hybrid",
        json={
            "user_id": user_id,
            "movie_id": 1,
            "top_k": 5,
            "content_weight": 0.5,
            "collaborative_weight": 0.5
        }
    )
    assert rec_resp2.status_code == 200

    hist_resp = client.get(f"/api/v1/users/{user_id}/recommendation-history")
    assert hist_resp.status_code == 200
    hist_data = hist_resp.json()
    assert hist_data["total"] >= 1
    item = hist_data["items"][0]
    assert item["model"] == "hybrid"
    assert item["user_id"] == user_id
    assert item["top_k"] == 5


def test_get_recommendation_history_user_not_found(client):
    """Test GET /api/v1/users/{id}/recommendation-history returns 404 for non-existent user."""
    response = client.get("/api/v1/users/99999/recommendation-history")
    assert response.status_code == 404
