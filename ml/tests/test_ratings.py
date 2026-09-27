"""Unit tests for Ratings REST API endpoints."""

import pytest


@pytest.fixture
def test_user(client):
    response = client.post("/api/v1/users", json={"username": "rater_user", "email": "rater@example.com"})
    return response.json()


def test_upsert_rating_create_and_update(client, test_user):
    """Test PUT /api/v1/ratings creates new rating (201) and updates existing rating (200)."""
    user_id = test_user["id"]

    # 1. Create rating -> 201 Created
    put_resp1 = client.put("/api/v1/ratings", json={"user_id": user_id, "movie_id": 1, "rating": 4.5})
    assert put_resp1.status_code == 201
    data1 = put_resp1.json()
    assert data1["user_id"] == user_id
    assert data1["movie_id"] == 1
    assert data1["rating"] == 4.5

    # 2. Update same rating -> 200 OK
    put_resp2 = client.put("/api/v1/ratings", json={"user_id": user_id, "movie_id": 1, "rating": 5.0})
    assert put_resp2.status_code == 200
    data2 = put_resp2.json()
    assert data2["rating"] == 5.0


def test_upsert_rating_invalid_user_not_found(client):
    """Test PUT /api/v1/ratings returns 404 when user_id does not exist."""
    response = client.put("/api/v1/ratings", json={"user_id": 99999, "movie_id": 1, "rating": 4.0})
    assert response.status_code == 404
    data = response.json()
    assert data["error"]["code"] == "USER_NOT_FOUND"


def test_upsert_rating_out_of_bounds_and_step_validation(client, test_user):
    """Test PUT /api/v1/ratings validates rating range (0.5 to 5.0) and 0.5 step increments."""
    user_id = test_user["id"]

    # Below minimum (0.0) -> 422
    resp1 = client.put("/api/v1/ratings", json={"user_id": user_id, "movie_id": 1, "rating": 0.0})
    assert resp1.status_code == 422

    # Above maximum (5.5) -> 422
    resp2 = client.put("/api/v1/ratings", json={"user_id": user_id, "movie_id": 1, "rating": 5.5})
    assert resp2.status_code == 422

    # Non 0.5 increment (4.2) -> 422
    resp3 = client.put("/api/v1/ratings", json={"user_id": user_id, "movie_id": 1, "rating": 4.2})
    assert resp3.status_code == 422


def test_get_user_rating(client, test_user):
    """Test GET /api/v1/users/{id}/ratings/{movie_id}."""
    user_id = test_user["id"]
    client.put("/api/v1/ratings", json={"user_id": user_id, "movie_id": 10, "rating": 3.5})

    response = client.get(f"/api/v1/users/{user_id}/ratings/10")
    assert response.status_code == 200
    data = response.json()
    assert data["rating"] == 3.5


def test_get_user_rating_not_found(client, test_user):
    """Test GET /api/v1/users/{id}/ratings/{movie_id} returns 404 when rating not found."""
    user_id = test_user["id"]
    response = client.get(f"/api/v1/users/{user_id}/ratings/999")
    assert response.status_code == 404
    data = response.json()
    assert data["error"]["code"] == "RATING_NOT_FOUND"


def test_list_user_ratings_paginated(client, test_user):
    """Test GET /api/v1/users/{id}/ratings returns paginated ratings list."""
    user_id = test_user["id"]
    client.put("/api/v1/ratings", json={"user_id": user_id, "movie_id": 1, "rating": 4.0})
    client.put("/api/v1/ratings", json={"user_id": user_id, "movie_id": 2, "rating": 4.5})

    response = client.get(f"/api/v1/users/{user_id}/ratings?limit=10&offset=0")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["items"]) == 2


def test_delete_user_rating(client, test_user):
    """Test DELETE /api/v1/users/{id}/ratings/{movie_id}."""
    user_id = test_user["id"]
    client.put("/api/v1/ratings", json={"user_id": user_id, "movie_id": 5, "rating": 2.5})

    del_resp = client.delete(f"/api/v1/users/{user_id}/ratings/5")
    assert del_resp.status_code == 204

    get_resp = client.get(f"/api/v1/users/{user_id}/ratings/5")
    assert get_resp.status_code == 404
