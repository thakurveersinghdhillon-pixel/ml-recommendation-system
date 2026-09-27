"""Unit tests for Phase 6 evaluation module and utilities."""

import pytest
import numpy as np
import pandas as pd

from ml.src.content_based import ContentBasedRecommender
from ml.src.collaborative_filtering import ItemBasedCollaborativeRecommender
from ml.src.hybrid_recommender import HybridRecommender
from ml.src.evaluation_utils import (
    temporal_train_test_split,
    get_user_training_content_seed,
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    hit_rate_at_k,
    catalog_coverage
)
from ml.src.evaluation import ModelEvaluator


@pytest.fixture
def synthetic_ratings_df() -> pd.DataFrame:
    """Create synthetic ratings DataFrame with controlled timestamps for temporal testing."""
    records = []
    # User 1: 10 interactions across timestamps 100..109
    for i in range(10):
        records.append({
            "userId": 1,
            "movieId": 10 + i,
            "rating": 4.5 if i % 2 == 0 else 2.0,
            "timestamp": 100 + i
        })

    # User 2: 6 interactions with duplicate timestamps to test tie-breaking (movieId ASC)
    for i in range(6):
        records.append({
            "userId": 2,
            "movieId": 30 - i,  # Decreasing movieIds: 30, 29, 28...
            "rating": 5.0,
            "timestamp": 200  # Duplicate timestamp for all
        })

    # User 3: 3 interactions (insufficient for min_ratings=5)
    for i in range(3):
        records.append({
            "userId": 3,
            "movieId": 50 + i,
            "rating": 4.0,
            "timestamp": 300 + i
        })

    return pd.DataFrame(records)


@pytest.fixture
def synthetic_movies_df() -> pd.DataFrame:
    """Create synthetic movies catalog DataFrame."""
    mids = list(range(10, 60))
    titles = [f"Movie {mid} (1995)" for mid in mids]
    genres = ["Action|Adventure" if mid % 2 == 0 else "Comedy|Drama" for mid in mids]

    return pd.DataFrame({
        "movieId": mids,
        "title": titles,
        "genres": genres
    })


def test_temporal_train_test_split_chronological(synthetic_ratings_df):
    """Test 1: Temporal split preserves chronological order and isolates earlier/later interactions."""
    train_df, test_df, split_info = temporal_train_test_split(
        synthetic_ratings_df, test_ratio=0.20, min_ratings=5
    )

    assert split_info["eligible_users"] == 2
    assert split_info["excluded_users"] == 1

    # User 1 check
    u1_train = train_df[train_df["userId"] == 1]
    u1_test = test_df[test_df["userId"] == 1]

    assert len(u1_train) == 8
    assert len(u1_test) == 2
    assert u1_train["timestamp"].max() <= u1_test["timestamp"].min()


def test_temporal_split_tie_breaking(synthetic_ratings_df):
    """Test 2: Deterministic secondary tie-breaking by movieId ASC when timestamps are identical."""
    train_df, test_df, _ = temporal_train_test_split(
        synthetic_ratings_df, test_ratio=0.33, min_ratings=5
    )

    u2_train = train_df[train_df["userId"] == 2]
    # For User 2, all timestamps were 200. MovieIds were [30, 29, 28, 27, 26, 25].
    # After sorting by (timestamp ASC, movieId ASC), order becomes [25, 26, 27, 28, 29, 30].
    # First 4 (n_train = 4) should be 25, 26, 27, 28.
    assert sorted(u2_train["movieId"].tolist()) == u2_train["movieId"].tolist()


def test_train_test_disjointness_and_no_leakage(synthetic_ratings_df):
    """Test 3: Train and test interactions for evaluated users are strictly disjoint."""
    train_df, test_df, _ = temporal_train_test_split(
        synthetic_ratings_df, test_ratio=0.20, min_ratings=5
    )

    eval_users = test_df["userId"].unique()
    for uid in eval_users:
        train_mids = set(train_df[train_df["userId"] == uid]["movieId"])
        test_mids = set(test_df[test_df["userId"] == uid]["movieId"])
        assert len(train_mids & test_mids) == 0, f"Data leakage detected for user {uid}!"


def test_precision_at_k_hand_computed():
    """Test 4: Precision@K calculation against hand-computed example."""
    recs = [10, 20, 30, 40, 50]
    relevant = {20, 40, 60}

    # Top 5: 2 hits (20, 40) -> 2/5 = 0.4
    assert precision_at_k(recs, relevant, k=5) == 0.4
    # Top 3: 1 hit (20) -> 1/3
    assert pytest.approx(precision_at_k(recs, relevant, k=3), 0.001) == 0.3333


def test_recall_at_k_hand_computed():
    """Test 5: Recall@K calculation against hand-computed example."""
    recs = [10, 20, 30, 40, 50]
    relevant = {20, 40, 60, 80}  # Total relevant = 4

    # Top 5: 2 hits (20, 40) out of 4 -> 2/4 = 0.5
    assert recall_at_k(recs, relevant, k=5) == 0.5
    # Empty relevant items -> returns None
    assert recall_at_k(recs, set(), k=5) is None


def test_ndcg_at_k_hand_computed():
    """Test 6: NDCG@K calculation against hand-computed example."""
    recs = [10, 20, 30, 40]
    relevant = {10, 30}  # Relevant at rank 1 and rank 3 (1-indexed)

    # DCG@4 = 1/log2(2) + 0 + 1/log2(4) + 0 = 1.0 + 0.5 = 1.5
    # IDCG@4 = 1/log2(2) + 1/log2(3) = 1.0 + 0.6309297 = 1.6309297
    # NDCG@4 = 1.5 / 1.6309297 = 0.9197
    ndcg = ndcg_at_k(recs, relevant, k=4)
    expected_dcg = 1.0 / np.log2(2) + 1.0 / np.log2(4)
    expected_idcg = 1.0 / np.log2(2) + 1.0 / np.log2(3)
    expected_ndcg = expected_dcg / expected_idcg

    assert pytest.approx(ndcg, 0.001) == expected_ndcg
    assert ndcg_at_k(recs, set(), k=4) is None


def test_hit_rate_at_k_hand_computed():
    """Test 7: Hit Rate@K calculation against hand-computed example."""
    recs = [10, 20, 30]
    assert hit_rate_at_k(recs, {20}, k=3) == 1.0
    assert hit_rate_at_k(recs, {99}, k=3) == 0.0
    assert hit_rate_at_k(recs, set(), k=3) is None


def test_catalog_coverage_calculation():
    """Test 8: Catalog coverage calculation."""
    all_recs = [10, 20, 30, 40, 10, 20]
    catalog = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100}  # 10 catalog items

    # 4 unique recommended items out of 10 catalog items -> 0.4
    assert catalog_coverage(all_recs, catalog) == 0.4


def test_get_user_training_content_seed(synthetic_ratings_df, synthetic_movies_df):
    """Test 9: Content seed selected exclusively from user training interactions."""
    train_df, _, _ = temporal_train_test_split(synthetic_ratings_df, test_ratio=0.20, min_ratings=5)

    seed_title = get_user_training_content_seed(1, train_df, synthetic_movies_df)
    assert seed_title is not None

    # Verify seed movie belongs to User 1's training ratings
    u1_train_mids = set(train_df[train_df["userId"] == 1]["movieId"])
    seed_mid = synthetic_movies_df[synthetic_movies_df["title"] == seed_title]["movieId"].iloc[0]
    assert seed_mid in u1_train_mids


def test_end_to_end_model_evaluator(synthetic_ratings_df, synthetic_movies_df, tmp_path):
    """Test 10: End-to-end ModelEvaluator execution on synthetic dataset using temporary directory."""
    eval_dir = tmp_path / "evaluation"
    evaluator = ModelEvaluator(
        test_ratio=0.20,
        min_ratings=5,
        relevance_threshold=4.0,
        k_values=[2, 5],
        seed=42,
        output_dir=eval_dir
    )

    results = evaluator.run_evaluation(synthetic_movies_df, synthetic_ratings_df)

    assert "metadata" in results
    assert "models" in results
    assert len(results["models"]) == 3  # Content, Collaborative, Hybrid

    # Verify output files generated in temporary test output directory
    assert (evaluator.metrics_dir / "model_comparison.csv").exists()
    assert (evaluator.metrics_dir / "model_comparison.json").exists()
    assert (evaluator.metrics_dir / "per_user_metrics.csv").exists()

    assert (evaluator.plots_dir / "precision_at_k.png").exists()
    assert (evaluator.plots_dir / "recall_at_k.png").exists()
    assert (evaluator.plots_dir / "ndcg_at_k.png").exists()
    assert (evaluator.plots_dir / "hit_rate_at_k.png").exists()
    assert (evaluator.plots_dir / "catalog_coverage.png").exists()
