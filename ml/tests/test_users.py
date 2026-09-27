"""Unit tests for Users REST API endpoints."""

import pytest


def test_create_user_success(client):
    """Test POST /api/v1/users creates new user successfully."""
    response = client.post("/api/v1/users", json={"username": "alice", "email": "alice@example.com"})
    assert response.status_code == 201
    data = response.json()
    assert data["id"] is not None
    assert data["username"] == "alice"
    assert data["email"] == "alice@example.com"
    assert "created_at" in data
    assert "updated_at" in data


def test_create_user_duplicate_username_conflict(client):
    """Test POST /api/v1/users returns 409 Conflict when username exists."""
    client.post("/api/v1/users", json={"username": "bob", "email": "bob1@example.com"})
    response = client.post("/api/v1/users", json={"username": "bob", "email": "bob2@example.com"})
    assert response.status_code == 409
    data = response.json()
    assert data["error"]["code"] == "DUPLICATE_USER"


def test_create_user_duplicate_email_conflict(client):
    """Test POST /api/v1/users returns 409 Conflict when email exists."""
    client.post("/api/v1/users", json={"username": "charlie1", "email": "charlie@example.com"})
    response = client.post("/api/v1/users", json={"username": "charlie2", "email": "charlie@example.com"})
    assert response.status_code == 409
    data = response.json()
    assert data["error"]["code"] == "DUPLICATE_USER"


def test_create_user_validation_error(client):
    """Test POST /api/v1/users returns 422 Unprocessable Content for invalid payload."""
    response = client.post("/api/v1/users", json={"username": "ab", "email": "invalid-email"})
    assert response.status_code == 422


def test_get_user_by_id_success(client):
    """Test GET /api/v1/users/{id} returns user details."""
    created = client.post("/api/v1/users", json={"username": "dave", "email": "dave@example.com"}).json()
    user_id = created["id"]

    response = client.get(f"/api/v1/users/{user_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == user_id
    assert data["username"] == "dave"


def test_get_user_by_id_not_found(client):
    """Test GET /api/v1/users/{id} returns 404 for non-existent user."""
    response = client.get("/api/v1/users/99999")
    assert response.status_code == 404
    data = response.json()
    assert data["error"]["code"] == "USER_NOT_FOUND"


def test_update_user_success(client):
    """Test PATCH /api/v1/users/{id} updates user fields."""
    created = client.post("/api/v1/users", json={"username": "eve", "email": "eve@example.com"}).json()
    user_id = created["id"]

    response = client.patch(f"/api/v1/users/{user_id}", json={"username": "eve_updated"})
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == "eve_updated"
    assert data["email"] == "eve@example.com"


def test_update_user_not_found(client):
    """Test PATCH /api/v1/users/{id} returns 404 for non-existent user."""
    response = client.patch("/api/v1/users/99999", json={"username": "ghost"})
    assert response.status_code == 404


def test_delete_user_success(client):
    """Test DELETE /api/v1/users/{id} removes user."""
    created = client.post("/api/v1/users", json={"username": "frank", "email": "frank@example.com"}).json()
    user_id = created["id"]

    del_resp = client.delete(f"/api/v1/users/{user_id}")
    assert del_resp.status_code == 204

    get_resp = client.get(f"/api/v1/users/{user_id}")
    assert get_resp.status_code == 404


def test_delete_user_not_found(client):
    """Test DELETE /api/v1/users/{id} returns 404 for non-existent user."""
    response = client.delete("/api/v1/users/99999")
    assert response.status_code == 404
