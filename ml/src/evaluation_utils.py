"""Evaluation utilities for temporal train/test splitting and ranking metrics."""

import logging
import math
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def temporal_train_test_split(
    ratings_df: pd.DataFrame,
    test_ratio: float = 0.20,
    min_ratings: int = 5,
    seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Perform a leakage-free per-user temporal train/test split.

    Sorts each user's ratings chronologically by timestamp ASC (tie-breaker: movieId ASC).
    The earlier interactions are placed in train_df, and the later interactions in test_df.
    Users with fewer than min_ratings or who cannot yield at least 1 train and 1 test rating are excluded.

    Args:
        ratings_df: DataFrame containing ['userId', 'movieId', 'rating', 'timestamp'].
        test_ratio: Proportion of recent interactions to allocate to test set (0.0 < test_ratio < 1.0).
        min_ratings: Minimum total interactions required per user for eligibility.
        seed: Random seed for reproducibility (used if any random selection is ever needed).

    Returns:
        Tuple of (train_df, test_df, split_info_dict).

    Raises:
        ValueError: If ratings_df is empty or missing required columns, or test_ratio is invalid.
    """
    if ratings_df is None or ratings_df.empty:
        raise ValueError("Input ratings_df cannot be empty or None.")

    required_cols = {"userId", "movieId", "rating", "timestamp"}
    if not required_cols.issubset(ratings_df.columns):
        missing = required_cols - set(ratings_df.columns)
        raise ValueError(f"ratings_df missing required columns: {missing}")

    if not (0.0 < test_ratio < 1.0):
        raise ValueError(f"test_ratio must be between 0.0 and 1.0, got {test_ratio}.")

    if min_ratings < 2:
        raise ValueError(f"min_ratings must be at least 2 to allow splitting, got {min_ratings}.")

    # Make isolated working copy
    df = ratings_df[["userId", "movieId", "rating", "timestamp"]].copy()

    unique_users = sorted(df["userId"].unique().tolist())
    total_users_count = len(unique_users)

    train_rows = []
    test_rows = []

    eligible_users_count = 0
    excluded_users_count = 0

    for user_id in unique_users:
        user_ratings = df[df["userId"] == user_id].copy()

        # Deterministic chronological sort: timestamp ASC, secondary tie-breaker movieId ASC
        user_ratings.sort_values(by=["timestamp", "movieId"], ascending=[True, True], inplace=True)
        user_ratings.reset_index(drop=True, inplace=True)

        n_total = len(user_ratings)

        if n_total < min_ratings:
            excluded_users_count += 1
            # Include all user ratings in train_df so collaborative filtering retains full context,
            # but user is not included in test set evaluation.
            train_rows.append(user_ratings)
            continue

        # Calculate number of test items (at least 1 test item, at least 1 train item)
        n_test = int(np.ceil(n_total * test_ratio))
        n_train = n_total - n_test

        if n_train < 1 or n_test < 1:
            excluded_users_count += 1
            train_rows.append(user_ratings)
            continue

        eligible_users_count += 1

        user_train = user_ratings.iloc[:n_train]
        user_test = user_ratings.iloc[n_train:]

        train_rows.append(user_train)
        test_rows.append(user_test)

    train_df = pd.concat(train_rows, ignore_index=True) if train_rows else pd.DataFrame(columns=df.columns)
    test_df = pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame(columns=df.columns)

    # Verify Data Leakage Safeguard: train_df and test_df interaction sets must be strictly disjoint for evaluated users
    split_info = {
        "total_users": total_users_count,
        "eligible_users": eligible_users_count,
        "excluded_users": excluded_users_count,
        "train_interactions": len(train_df),
        "test_interactions": len(test_df),
        "test_ratio": test_ratio,
        "min_ratings": min_ratings,
        "seed": seed
    }

    logger.info(
        "Temporal split complete: %d total users (%d eligible, %d excluded). Train: %d, Test: %d.",
        total_users_count,
        eligible_users_count,
        excluded_users_count,
        len(train_df),
        len(test_df)
    )

    return train_df, test_df, split_info


def get_user_training_content_seed(
    user_id: int,
    user_train_df: pd.DataFrame,
    movies_df: pd.DataFrame
) -> str | None:
    """Select the most recent highly-rated movie title from a user's training interactions ONLY.

    Uses deterministic tie-breaking: timestamp DESC, rating DESC, movieId ASC.

    Args:
        user_id: Target user ID integer.
        user_train_df: DataFrame containing user's training ratings ONLY.
        movies_df: DataFrame containing movie catalog metadata.

    Returns:
        Movie title string if available, else None.
    """
    if user_train_df is None or user_train_df.empty:
        return None

    # Filter user's training ratings
    u_train = user_train_df[user_train_df["userId"] == user_id]
    if u_train.empty:
        return None

    # Deterministic sort: timestamp DESC, rating DESC, movieId ASC
    u_train_sorted = u_train.sort_values(
        by=["timestamp", "rating", "movieId"],
        ascending=[False, False, True]
    )

    for _, row in u_train_sorted.iterrows():
        mid = int(row["movieId"])
        match = movies_df[movies_df["movieId"] == mid]
        if not match.empty:
            title = str(match.iloc[0]["title"])
            if title and title.strip():
                return title

    return None


def precision_at_k(recommended_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Calculate Precision@K metric.

    Precision@K = |Top-K Recommendations ∩ Relevant Test Items| / K

    Args:
        recommended_ids: Ordered list of recommended movie IDs.
        relevant_ids: Set of relevant test movie IDs (ratings >= relevance_threshold).
        k: Top K cut-off rank.

    Returns:
        Precision score float in [0.0, 1.0].
    """
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}.")

    top_k = recommended_ids[:k]
    if not top_k:
        return 0.0

    hits = len(set(top_k) & relevant_ids)
    return float(hits / k)


def recall_at_k(recommended_ids: list[int], relevant_ids: set[int], k: int) -> float | None:
    """Calculate Recall@K metric.

    Recall@K = |Top-K Recommendations ∩ Relevant Test Items| / |Relevant Test Items|

    Returns None if relevant_ids is empty (user has no relevant test items).

    Args:
        recommended_ids: Ordered list of recommended movie IDs.
        relevant_ids: Set of relevant test movie IDs.
        k: Top K cut-off rank.

    Returns:
        Recall score float in [0.0, 1.0] or None if relevant_ids is empty.
    """
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}.")

    if not relevant_ids:
        return None

    top_k = recommended_ids[:k]
    hits = len(set(top_k) & relevant_ids)
    return float(hits / len(relevant_ids))


def ndcg_at_k(recommended_ids: list[int], relevant_ids: set[int], k: int) -> float | None:
    """Calculate Normalized Discounted Cumulative Gain (NDCG@K) metric.

    NDCG@K = DCG@K / IDCG@K

    Returns None if relevant_ids is empty (user has no relevant test items).

    Args:
        recommended_ids: Ordered list of recommended movie IDs.
        relevant_ids: Set of relevant test movie IDs.
        k: Top K cut-off rank.

    Returns:
        NDCG score float in [0.0, 1.0] or None if relevant_ids is empty.
    """
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}.")

    if not relevant_ids:
        return None

    top_k = recommended_ids[:k]

    # Calculate DCG@K
    dcg = 0.0
    for idx, item_id in enumerate(top_k):
        if item_id in relevant_ids:
            # Rank is 1-indexed (idx + 1), log2(rank + 1) = log2(idx + 2)
            dcg += 1.0 / np.log2(idx + 2)

    # Calculate Ideal DCG@K
    idcg = 0.0
    n_ideal = min(k, len(relevant_ids))
    for idx in range(n_ideal):
        idcg += 1.0 / np.log2(idx + 2)

    if idcg == 0.0:
        return 0.0

    return float(dcg / idcg)


def hit_rate_at_k(recommended_ids: list[int], relevant_ids: set[int], k: int) -> float | None:
    """Calculate Hit Rate@K metric.

    Hit Rate@K = 1.0 if |Top-K Recommendations ∩ Relevant Test Items| > 0 else 0.0

    Returns None if relevant_ids is empty (user has no relevant test items).

    Args:
        recommended_ids: Ordered list of recommended movie IDs.
        relevant_ids: Set of relevant test movie IDs.
        k: Top K cut-off rank.

    Returns:
        Hit Rate score float (1.0 or 0.0) or None if relevant_ids is empty.
    """
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}.")

    if not relevant_ids:
        return None

    top_k = recommended_ids[:k]
    hits = len(set(top_k) & relevant_ids)
    return 1.0 if hits > 0 else 0.0


def catalog_coverage(all_recommended_ids: set[int] | list[int], catalog_movie_ids: set[int]) -> float:
    """Calculate Catalog Coverage metric.

    Catalog Coverage = |Unique Recommended Movie IDs ∩ Catalog Movie IDs| / |Catalog Movie IDs|

    Args:
        all_recommended_ids: Collection of all unique movie IDs recommended across all users.
        catalog_movie_ids: Set of all eligible catalog movie IDs.

    Returns:
        Coverage proportion float in [0.0, 1.0].
    """
    if not catalog_movie_ids:
        return 0.0

    unique_recs = set(all_recommended_ids) & catalog_movie_ids
    return float(len(unique_recs) / len(catalog_movie_ids))
