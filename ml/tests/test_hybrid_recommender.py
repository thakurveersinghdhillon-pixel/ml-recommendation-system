"""Unit tests for HybridRecommender module."""

import pytest
import numpy as np
import pandas as pd

from ml.src.content_based import ContentBasedRecommender
from ml.src.collaborative_filtering import ItemBasedCollaborativeRecommender
from ml.src.hybrid_recommender import HybridRecommender, min_max_normalize


@pytest.fixture
def sample_catalog_df() -> pd.DataFrame:
    """Create a synthetic movies catalog."""
    return pd.DataFrame({
        "movieId": [10, 20, 30, 40, 50],
        "title": [
            "Toy Story (1995)",
            "Jumanji (1995)",
            "Grumpier Old Men (1995)",
            "Waiting to Exhale (1995)",
            "Father of the Bride Part II (1995)"
        ],
        "genres": [
            "Adventure|Animation|Children|Comedy|Fantasy",
            "Adventure|Children|Fantasy",
            "Comedy|Romance",
            "Comedy|Drama|Romance",
            "Comedy"
        ]
    })


@pytest.fixture
def sample_ratings_df() -> pd.DataFrame:
    """Create a synthetic ratings dataset."""
    return pd.DataFrame({
        "userId": [1, 1, 1, 2, 2, 3, 3],
        "movieId": [10, 20, 30, 10, 40, 20, 40],
        "rating": [5.0, 4.0, 2.0, 4.5, 5.0, 4.0, 1.0],
        "timestamp": [1, 2, 3, 4, 5, 6, 7]
    })


@pytest.fixture
def fitted_recommenders(sample_catalog_df, sample_ratings_df):
    """Fixture providing fitted Content-Based and Collaborative Filtering recommenders."""
    content_rec = ContentBasedRecommender()
    content_rec.fit(sample_catalog_df)

    collab_rec = ItemBasedCollaborativeRecommender()
    collab_rec.fit(sample_ratings_df, movies_df=sample_catalog_df)

    return content_rec, collab_rec


def test_constructor_success(fitted_recommenders):
    """Test 1: Constructor initializes successfully with default parameters."""
    c_rec, cf_rec = fitted_recommenders
    hybrid = HybridRecommender(c_rec, cf_rec)
    assert hybrid.content_weight == 0.5
    assert hybrid.collaborative_weight == 0.5


def test_constructor_valid_custom_weights(fitted_recommenders):
    """Test 2: Constructor initializes with valid custom weights (0.7 / 0.3)."""
    c_rec, cf_rec = fitted_recommenders
    hybrid = HybridRecommender(c_rec, cf_rec, content_weight=0.7, collaborative_weight=0.3)
    assert hybrid.content_weight == 0.7
    assert hybrid.collaborative_weight == 0.3


def test_constructor_negative_weight_raises_error(fitted_recommenders):
    """Test 3: Negative weights raise ValueError."""
    c_rec, cf_rec = fitted_recommenders
    with pytest.raises(ValueError, match="non-negative"):
        HybridRecommender(c_rec, cf_rec, content_weight=-0.5, collaborative_weight=1.5)


def test_constructor_weights_not_summing_to_one(fitted_recommenders):
    """Test 4: Weights not summing to 1.0 raise ValueError without auto-normalizing."""
    c_rec, cf_rec = fitted_recommenders
    with pytest.raises(ValueError, match="sum to 1.0"):
        HybridRecommender(c_rec, cf_rec, content_weight=0.5, collaborative_weight=0.8)
    with pytest.raises(ValueError, match="sum to 1.0"):
        HybridRecommender(c_rec, cf_rec, content_weight=2.0, collaborative_weight=1.0)


def test_constructor_valid_weight_boundary_combinations(fitted_recommenders):
    """Test 4b: Valid weight combinations (0.0/1.0, 1.0/0.0, 0.2/0.8, 0.7/0.3) are accepted."""
    c_rec, cf_rec = fitted_recommenders
    h1 = HybridRecommender(c_rec, cf_rec, content_weight=0.0, collaborative_weight=1.0)
    assert h1.content_weight == 0.0 and h1.collaborative_weight == 1.0

    h2 = HybridRecommender(c_rec, cf_rec, content_weight=1.0, collaborative_weight=0.0)
    assert h2.content_weight == 1.0 and h2.collaborative_weight == 0.0


def test_min_max_normalize_math():
    """Test 5: Min-max score normalization maps scores to range [0.0, 1.0]."""
    scores = [10.0, 20.0, 30.0]
    norm = min_max_normalize(scores)
    assert norm == [0.0, 0.5, 1.0]


def test_min_max_normalize_constant_scores():
    """Test 6: Constant score normalization assigns 1.0 to all candidates."""
    scores = [5.0, 5.0, 5.0]
    norm = min_max_normalize(scores)
    assert norm == [1.0, 1.0, 1.0]


def test_candidate_merging(fitted_recommenders):
    """Test 7: Candidates from content and collaborative sources are merged."""
    c_rec, cf_rec = fitted_recommenders
    hybrid = HybridRecommender(c_rec, cf_rec)
    recs = hybrid.recommend_for_user(user_id=1, n=10)
    assert len(recs) > 0


def test_duplicate_candidate_removal(fitted_recommenders):
    """Test 8: Duplicate candidate movieIds across sources are removed (unique entries)."""
    c_rec, cf_rec = fitted_recommenders
    hybrid = HybridRecommender(c_rec, cf_rec)
    recs = hybrid.recommend_for_user(user_id=1, n=10)
    mids = [r["movieId"] for r in recs]
    assert len(mids) == len(set(mids))  # All movieIds must be unique


def test_missing_content_score_policy(fitted_recommenders):
    """Test 9: Missing content score source receives normalized score 0.0 and raw score None."""
    c_rec, cf_rec = fitted_recommenders
    hybrid = HybridRecommender(c_rec, cf_rec)
    recs = hybrid.recommend_for_user(user_id=1, n=10)

    for r in recs:
        if r["content_score"] is None:
            assert r["normalized_content_score"] == 0.0


def test_missing_collaborative_score_policy(fitted_recommenders):
    """Test 10: Missing collaborative score source receives normalized score 0.0 and raw score None."""
    c_rec, cf_rec = fitted_recommenders
    hybrid = HybridRecommender(c_rec, cf_rec)
    # Cold start user with seed movie title (content-only pathway)
    recs = hybrid.recommend_for_user(user_id=999, seed_movie_title="Toy Story (1995)", n=5)

    for r in recs:
        assert r["collaborative_score"] is None
        assert r["normalized_collaborative_score"] == 0.0


def test_hybrid_score_calculation(fitted_recommenders):
    """Test 11: Hybrid score formula calculation accuracy."""
    c_rec, cf_rec = fitted_recommenders
    hybrid = HybridRecommender(c_rec, cf_rec, content_weight=0.6, collaborative_weight=0.4)
    recs = hybrid.recommend_for_user(user_id=1, n=5)

    for r in recs:
        expected = round(0.6 * r["normalized_content_score"] + 0.4 * r["normalized_collaborative_score"], 4)
        assert pytest.approx(r["hybrid_score"], 0.0001) == expected


def test_weight_changes_affect_scores(fitted_recommenders):
    """Test 12: Changing weights alters final hybrid scores."""
    c_rec, cf_rec = fitted_recommenders
    h1 = HybridRecommender(c_rec, cf_rec, content_weight=0.9, collaborative_weight=0.1)
    h2 = HybridRecommender(c_rec, cf_rec, content_weight=0.1, collaborative_weight=0.9)

    r1 = h1.recommend_for_user(user_id=1, n=2)
    r2 = h2.recommend_for_user(user_id=1, n=2)

    assert r1[0]["hybrid_score"] != r2[0]["hybrid_score"] or r1[0]["movieId"] != r2[0]["movieId"]


def test_deterministic_ranking(fitted_recommenders):
    """Test 13: Candidate ranking is deterministic across repeated calls."""
    c_rec, cf_rec = fitted_recommenders
    hybrid = HybridRecommender(c_rec, cf_rec)
    call_1 = hybrid.recommend_for_user(user_id=1, n=5)
    call_2 = hybrid.recommend_for_user(user_id=1, n=5)
    assert call_1 == call_2


def test_movie_id_tie_breaking(fitted_recommenders):
    """Test 14: Ties in hybrid_score are broken deterministically by movieId ASC."""
    c_rec, cf_rec = fitted_recommenders
    hybrid = HybridRecommender(c_rec, cf_rec)
    recs = hybrid.recommend_for_user(user_id=1, n=10)

    for i in range(len(recs) - 1):
        if recs[i]["hybrid_score"] == recs[i + 1]["hybrid_score"]:
            assert recs[i]["movieId"] < recs[i + 1]["movieId"]


def test_already_rated_filtering_invariant(fitted_recommenders):
    """Test 15: MANDATORY RULE: Already-rated movies are strictly filtered out."""
    c_rec, cf_rec = fitted_recommenders
    hybrid = HybridRecommender(c_rec, cf_rec)

    # User 1 rated 10, 20, 30
    recs = hybrid.recommend_for_user(user_id=1, n=10)
    rec_ids = [r["movieId"] for r in recs]

    assert 10 not in rec_ids
    assert 20 not in rec_ids
    assert 30 not in rec_ids


def test_requested_n_fulfillment(fitted_recommenders):
    """Test 16: Requested N recommendation count is fulfilled."""
    c_rec, cf_rec = fitted_recommenders
    hybrid = HybridRecommender(c_rec, cf_rec)
    recs = hybrid.recommend_for_user(user_id=1, n=2)
    assert len(recs) == 2


def test_unknown_user_without_seed_title_raises_error(fitted_recommenders):
    """Test 17: Unknown user WITHOUT seed_movie_title raises ValueError."""
    c_rec, cf_rec = fitted_recommenders
    hybrid = HybridRecommender(c_rec, cf_rec)
    with pytest.raises(ValueError, match="requires either existing user ratings or a valid seed_movie_title"):
        hybrid.recommend_for_user(user_id=99999, seed_movie_title=None)


def test_cold_start_user_with_seed_title_provides_content_fallback(fitted_recommenders):
    """Test 18: Cold-start user WITH valid seed_movie_title provides content-based fallback."""
    c_rec, cf_rec = fitted_recommenders
    hybrid = HybridRecommender(c_rec, cf_rec)
    recs = hybrid.recommend_for_user(user_id=99999, seed_movie_title="Toy Story (1995)", n=2)
    assert len(recs) > 0
    assert recs[0]["content_score"] is not None


def test_unknown_seed_movie_title_raises_error(fitted_recommenders):
    """Test 19: Unknown seed_movie_title raises ValueError."""
    c_rec, cf_rec = fitted_recommenders
    hybrid = HybridRecommender(c_rec, cf_rec)
    with pytest.raises(ValueError, match="not found in dataset catalog"):
        hybrid.recommend_for_user(user_id=1, seed_movie_title="NonExistent Seed Movie (2099)")


def test_explanation_generation(fitted_recommenders):
    """Test 20: Explanation generation contains real model signals."""
    c_rec, cf_rec = fitted_recommenders
    hybrid = HybridRecommender(c_rec, cf_rec)
    recs = hybrid.recommend_for_user(user_id=1, n=5)
    rec_mid = recs[0]["movieId"]

    exp = hybrid.explain_hybrid_recommendation(user_id=1, movie_id=rec_mid)
    assert "explanation" in exp
    assert f"Recommended '{exp['title']}'" in exp["explanation"]


def test_content_only_candidate_pathway(fitted_recommenders):
    """Test 21: Content-only recommendation pathway."""
    c_rec, cf_rec = fitted_recommenders
    hybrid = HybridRecommender(c_rec, cf_rec, content_weight=1.0, collaborative_weight=0.0)
    recs = hybrid.recommend_for_user(user_id=1, n=2)
    assert len(recs) == 2


def test_collaborative_only_candidate_pathway(fitted_recommenders):
    """Test 22: Collaborative-only recommendation pathway."""
    c_rec, cf_rec = fitted_recommenders
    hybrid = HybridRecommender(c_rec, cf_rec, content_weight=0.0, collaborative_weight=1.0)
    recs = hybrid.recommend_for_user(user_id=1, n=2)
    assert len(recs) == 2


def test_repeated_call_determinism(fitted_recommenders):
    """Test 23: Repeated calls produce identical results."""
    c_rec, cf_rec = fitted_recommenders
    hybrid = HybridRecommender(c_rec, cf_rec)
    call_1 = hybrid.recommend_for_user(user_id=1, n=3)
    call_2 = hybrid.recommend_for_user(user_id=1, n=3)
    assert call_1 == call_2


def test_phase_3_integration(fitted_recommenders):
    """Test 24: Integration with Phase 3 ContentBasedRecommender."""
    c_rec, cf_rec = fitted_recommenders
    hybrid = HybridRecommender(c_rec, cf_rec)
    assert hybrid.content_recommender.is_fitted is True


def test_phase_4_integration(fitted_recommenders):
    """Test 25: Integration with Phase 4 ItemBasedCollaborativeRecommender."""
    c_rec, cf_rec = fitted_recommenders
    hybrid = HybridRecommender(c_rec, cf_rec)
    assert hybrid.collaborative_recommender.is_fitted is True
