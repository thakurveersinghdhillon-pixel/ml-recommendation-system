"""End-to-end Database Integration Tests for User, Ratings, and Recommendation Flow."""

import pytest


def test_full_database_user_rating_recommendation_lifecycle(client):
    """Verify full workflow:
    1. Create user
    2. Add ratings for user
    3. Retrieve ratings list
    4. Request hybrid recommendation for user
    5. Verify recommendation history persisted
    """
    # 1. Create User
    create_user_resp = client.post(
        "/api/v1/users",
        json={"username": "integration_user", "email": "integ@example.com"}
    )
    assert create_user_resp.status_code == 201
    user_data = create_user_resp.json()
    user_id = user_data["id"]

    # 2. Add Ratings (upsert)
    r1 = client.put("/api/v1/ratings", json={"user_id": user_id, "movie_id": 1, "rating": 5.0})
    assert r1.status_code == 201
    r2 = client.put("/api/v1/ratings", json={"user_id": user_id, "movie_id": 3114, "rating": 4.5})
    assert r2.status_code == 201

    # 3. Retrieve Ratings list
    ratings_resp = client.get(f"/api/v1/users/{user_id}/ratings")
    assert ratings_resp.status_code == 200
    ratings_data = ratings_resp.json()
    assert ratings_data["total"] == 2

    # 4. Request Hybrid Recommendation
    rec_resp = client.post(
        "/api/v1/recommendations/hybrid",
        json={
            "user_id": user_id,
            "movie_id": 1,
            "top_k": 5,
            "content_weight": 0.6,
            "collaborative_weight": 0.4
        }
    )
    assert rec_resp.status_code == 200
    rec_data = rec_resp.json()
    assert len(rec_data["recommendations"]) == 5

    # 5. Verify Recommendation History
    hist_resp = client.get(f"/api/v1/users/{user_id}/recommendation-history")
    assert hist_resp.status_code == 200
    hist_data = hist_resp.json()
    assert hist_data["total"] == 1
    hist_item = hist_data["items"][0]
    assert hist_item["user_id"] == user_id
    assert hist_item["model"] == "hybrid"
    assert hist_item["content_weight"] == 0.6
    assert hist_item["collaborative_weight"] == 0.4
    assert len(hist_item["recommendations"]) == 5
