"""Unit tests for ItemBasedCollaborativeRecommender module."""

import pytest
import numpy as np
import pandas as pd
from scipy.sparse import isspmatrix_csr

from ml.src.collaborative_filtering import ItemBasedCollaborativeRecommender


@pytest.fixture
def sample_ratings_df() -> pd.DataFrame:
    """Create a synthetic ratings dataset for collaborative filtering unit tests."""
    return pd.DataFrame({
        "userId": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
        "movieId": [10, 20, 30, 10, 20, 40, 20, 30, 40, 10, 40],
        "rating": [5.0, 4.0, 1.0, 4.5, 5.0, 2.0, 4.0, 5.0, 1.0, 2.0, 5.0],
        "timestamp": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    })


@pytest.fixture
def sample_movies_df() -> pd.DataFrame:
    """Create a synthetic movies catalog for display metadata tests."""
    return pd.DataFrame({
        "movieId": [10, 20, 30, 40, 50],
        "title": [
            "Movie A (2020)",
            "Movie B (2021)",
            "Movie C (2022)",
            "Movie D (2023)",
            "Unrated Movie E (2024)"
        ],
        "genres": ["Action", "Action|Sci-Fi", "Comedy", "Sci-Fi", "Drama"]
    })


def test_fit_success(sample_ratings_df, sample_movies_df):
    """Test 1: Recommender can fit successfully."""
    recommender = ItemBasedCollaborativeRecommender()
    recommender.fit(sample_ratings_df, movies_df=sample_movies_df)
    assert recommender.is_fitted is True


def test_validate_required_columns():
    """Test 2: Required rating columns are validated."""
    invalid_ratings = pd.DataFrame({"userId": [1], "rating": [5.0]})  # Missing movieId
    recommender = ItemBasedCollaborativeRecommender()
    with pytest.raises(ValueError, match="missing required columns"):
        recommender.fit(invalid_ratings)


def test_user_item_matrix_dimensions(sample_ratings_df):
    """Test 3: User-item matrix has expected dimensions."""
    recommender = ItemBasedCollaborativeRecommender()
    recommender.fit(sample_ratings_df)
    assert recommender.user_item_matrix.shape == (4, 4)  # 4 unique users, 4 unique movies in ratings


def test_user_to_index_mapping(sample_ratings_df):
    """Test 4: User-to-index mapping works."""
    recommender = ItemBasedCollaborativeRecommender()
    recommender.fit(sample_ratings_df)
    assert recommender.get_user_index(1) == 0
    assert recommender.get_user_index(4) == 3


def test_movie_to_index_mapping(sample_ratings_df):
    """Test 5: Movie-to-index mapping works."""
    recommender = ItemBasedCollaborativeRecommender()
    recommender.fit(sample_ratings_df)
    assert recommender.get_movie_index(10) == 0
    assert recommender.get_movie_index(40) == 3


def test_unknown_user_raises_error(sample_ratings_df):
    """Test 6: Unknown user raises meaningful ValueError."""
    recommender = ItemBasedCollaborativeRecommender()
    recommender.fit(sample_ratings_df)
    with pytest.raises(ValueError, match="not found in training dataset"):
        recommender.get_user_index(999)

    with pytest.raises(ValueError, match="not found in training dataset"):
        recommender.recommend_for_user(999, n=5)


def test_unknown_movie_raises_error(sample_ratings_df):
    """Test 7: Unknown movie raises meaningful ValueError."""
    recommender = ItemBasedCollaborativeRecommender()
    recommender.fit(sample_ratings_df)
    with pytest.raises(ValueError, match="not found in dataset catalog"):
        recommender.get_movie_index(999)

    with pytest.raises(ValueError, match="not found in dataset catalog"):
        recommender.get_similarity(10, 999)


def test_movie_similarity_calculation(sample_ratings_df):
    """Test 8: Movie similarity works."""
    recommender = ItemBasedCollaborativeRecommender()
    recommender.fit(sample_ratings_df)
    sim = recommender.get_similarity(10, 20)
    assert isinstance(sim, float)


def test_similarity_values_valid(sample_ratings_df):
    """Test 9: Similarity values are valid in range [0.0, 1.0]."""
    recommender = ItemBasedCollaborativeRecommender()
    recommender.fit(sample_ratings_df)
    sim = recommender.get_similarity(10, 30)
    assert 0.0 <= sim <= 1.0


def test_same_movie_similarity(sample_ratings_df):
    """Test 10: Same movie similarity is approximately 1.0."""
    recommender = ItemBasedCollaborativeRecommender()
    recommender.fit(sample_ratings_df)
    sim = recommender.get_similarity(10, 10)
    assert pytest.approx(sim, 0.001) == 1.0


def test_recommend_similar_movies(sample_ratings_df, sample_movies_df):
    """Test 11: Similar movie recommendations work."""
    recommender = ItemBasedCollaborativeRecommender()
    recommender.fit(sample_ratings_df, movies_df=sample_movies_df)
    sims = recommender.recommend_similar_movies(10, n=2)
    assert len(sims) == 2
    assert "title" in sims[0]


def test_recommendation_count(sample_ratings_df):
    """Test 12: Recommendation count N is fulfilled."""
    recommender = ItemBasedCollaborativeRecommender()
    recommender.fit(sample_ratings_df)
    recs = recommender.recommend_for_user(user_id=1, n=1)
    assert len(recs) == 1


def test_already_rated_movies_excluded(sample_ratings_df):
    """Test 13: Mandatory requirement: Already-rated movies are excluded from user recommendations."""
    recommender = ItemBasedCollaborativeRecommender()
    recommender.fit(sample_ratings_df)

    # User 1 has rated movies 10, 20, 30. Candidate left: 40.
    recs = recommender.recommend_for_user(user_id=1, n=10)
    rec_mids = [r["movieId"] for r in recs]

    assert 10 not in rec_mids
    assert 20 not in rec_mids
    assert 30 not in rec_mids
    assert 40 in rec_mids


def test_scores_ordered_descending(sample_ratings_df):
    """Test 14: Recommendation scores are correctly ordered descending."""
    recommender = ItemBasedCollaborativeRecommender()
    recommender.fit(sample_ratings_df)
    # User 4 has rated movies 10 and 40. Candidates left: 20 and 30.
    recs = recommender.recommend_for_user(user_id=4, n=5)
    scores = [r["score"] for r in recs]
    assert scores == sorted(scores, reverse=True)


def test_recommendation_ordering_deterministic(sample_ratings_df):
    """Test 15: Recommendation ordering is deterministic across repeated calls."""
    recommender = ItemBasedCollaborativeRecommender()
    recommender.fit(sample_ratings_df)
    call_1 = recommender.recommend_for_user(user_id=4, n=2)
    call_2 = recommender.recommend_for_user(user_id=4, n=2)
    assert call_1 == call_2


def test_invalid_n_raises_error(sample_ratings_df):
    """Test 16: n <= 0 raises a meaningful ValueError."""
    recommender = ItemBasedCollaborativeRecommender()
    recommender.fit(sample_ratings_df)
    with pytest.raises(ValueError, match="positive integer"):
        recommender.recommend_for_user(user_id=1, n=0)

    with pytest.raises(ValueError, match="positive integer"):
        recommender.recommend_similar_movies(movie_id=10, n=-1)


def test_large_n_handled_safely(sample_ratings_df):
    """Test 17: Large n is handled safely (caps to available candidate count)."""
    recommender = ItemBasedCollaborativeRecommender()
    recommender.fit(sample_ratings_df)
    recs = recommender.recommend_for_user(user_id=1, n=100)
    assert len(recs) == 1  # Only 1 unrated movie available for User 1 in this dataset


def test_duplicate_user_movie_ratings_aggregated():
    """Test 18: Duplicate (userId, movieId) ratings are aggregated deterministically via mean rating."""
    duplicate_ratings = pd.DataFrame({
        "userId": [1, 1, 2, 2],
        "movieId": [10, 10, 20, 20],
        "rating": [5.0, 3.0, 4.0, 2.0],  # User 1 Movie 10 mean = 4.0; User 2 Movie 20 mean = 3.0
        "timestamp": [1, 2, 3, 4]
    })
    recommender = ItemBasedCollaborativeRecommender()
    recommender.fit(duplicate_ratings)

    assert recommender.user_item_matrix.shape == (2, 2)
    assert recommender.user_means[1] == 4.0
    assert recommender.user_means[2] == 3.0


def test_sparse_matrix_representation(sample_ratings_df):
    """Test 19: Sparse scipy CSR matrix representation is used for user-item matrix."""
    recommender = ItemBasedCollaborativeRecommender()
    recommender.fit(sample_ratings_df)
    assert isspmatrix_csr(recommender.user_item_matrix)


def test_user_recommendation_scoring(sample_ratings_df):
    """Test 20: User recommendation scoring works and returns numeric scores."""
    recommender = ItemBasedCollaborativeRecommender()
    recommender.fit(sample_ratings_df)
    recs = recommender.recommend_for_user(user_id=2, n=1)
    assert len(recs) == 1
    assert isinstance(recs[0]["score"], float)


def test_positive_preference_signal_contribution(sample_movies_df):
    """Test 21: High ratings above user mean contribute positively to recommendation score."""
    custom_ratings = pd.DataFrame({
        "userId": [1, 1, 2, 2, 3, 3],
        "movieId": [10, 20, 10, 30, 20, 40],
        "rating": [5.0, 1.0, 5.0, 5.0, 1.0, 5.0],
        "timestamp": [1, 2, 3, 4, 5, 6]
    })
    recommender = ItemBasedCollaborativeRecommender()
    recommender.fit(custom_ratings, movies_df=sample_movies_df)

    recs = recommender.recommend_for_user(user_id=1, n=2)
    # Movie 30 (similar to high-rated Movie 10) should score highest
    assert recs[0]["movieId"] == 30


def test_low_user_ratings_do_not_overcontribute(sample_movies_df):
    """Test 22: Low ratings below user mean contribute negatively/less positively than high ratings."""
    custom_ratings = pd.DataFrame({
        "userId": [1, 1, 2, 2, 3, 3],
        "movieId": [10, 20, 10, 30, 20, 40],
        "rating": [5.0, 1.0, 5.0, 5.0, 1.0, 5.0],
        "timestamp": [1, 2, 3, 4, 5, 6]
    })
    recommender = ItemBasedCollaborativeRecommender()
    recommender.fit(custom_ratings, movies_df=sample_movies_df)

    recs = recommender.recommend_for_user(user_id=1, n=2)
    scores = {r["movieId"]: r["score"] for r in recs}
    assert scores[30] > scores[40]


def test_explanation_uses_collaborative_signals(sample_ratings_df, sample_movies_df):
    """Test 23: Explanation uses actual collaborative signals (source movie, rating, similarity)."""
    recommender = ItemBasedCollaborativeRecommender()
    recommender.fit(sample_ratings_df, movies_df=sample_movies_df)

    # User 1 candidates: Movie 40
    exp = recommender.explain_user_recommendation(user_id=1, movie_id=40)
    assert "user_id" in exp
    assert exp["user_id"] == 1
    assert "top_source_movie_id" in exp
    assert "item_similarity" in exp
    assert "User 1" in exp["explanation"]


def test_cold_start_unknown_user_behavior(sample_ratings_df):
    """Test 24: Cold-start/unknown user behavior is handled explicitly with ValueError."""
    recommender = ItemBasedCollaborativeRecommender()
    recommender.fit(sample_ratings_df)
    with pytest.raises(ValueError, match="not found in training dataset"):
        recommender.recommend_for_user(user_id=99999, n=5)
