"""Hybrid Recommendation Engine combining Content-Based and Collaborative Filtering."""

import logging
import math
import numpy as np
import pandas as pd

from ml.src.content_based import ContentBasedRecommender
from ml.src.collaborative_filtering import ItemBasedCollaborativeRecommender

logger = logging.getLogger(__name__)


def min_max_normalize(scores: list[float] | np.ndarray) -> list[float]:
    """Min-max normalize a non-empty sequence of numerical scores to range [0.0, 1.0].

    If all available scores are identical (max_score == min_score), assigns 1.0
    to all candidates to prevent division by zero.

    Args:
        scores: Sequence of float scores.

    Returns:
        List of normalized float scores in [0.0, 1.0].
    """
    if len(scores) == 0:
        return []

    arr = np.array(scores, dtype=np.float64)
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))

    if math.isclose(max_val, min_val, abs_tol=1e-9):
        return [1.0] * len(scores)

    norm_arr = (arr - min_val) / (max_val - min_val)
    return [float(x) for x in norm_arr]


class HybridRecommender:
    """Hybrid Recommendation Engine fusing Content-Based and Collaborative Filtering."""

    def __init__(
        self,
        content_recommender: ContentBasedRecommender,
        collaborative_recommender: ItemBasedCollaborativeRecommender,
        content_weight: float = 0.5,
        collaborative_weight: float = 0.5,
        tolerance: float = 1e-5
    ):
        """Initialize HybridRecommender with fitted recommender models and weights.

        Args:
            content_recommender: Fitted ContentBasedRecommender instance.
            collaborative_recommender: Fitted ItemBasedCollaborativeRecommender instance.
            content_weight: Weight for content-based score signal (must be >= 0).
            collaborative_weight: Weight for collaborative score signal (must be >= 0).
            tolerance: Floating point tolerance for checking weight sum == 1.0.

        Raises:
            ValueError: If weights are negative or do not sum to 1.0 within tolerance.
        """
        if content_weight < 0 or collaborative_weight < 0:
            raise ValueError(
                f"Weights must be non-negative. Got content_weight={content_weight}, "
                f"collaborative_weight={collaborative_weight}."
            )

        weight_sum = content_weight + collaborative_weight
        if abs(weight_sum - 1.0) > tolerance:
            raise ValueError(
                f"Weights must sum to 1.0. Got content_weight={content_weight}, "
                f"collaborative_weight={collaborative_weight} (sum={weight_sum})."
            )

        self.content_recommender = content_recommender
        self.collaborative_recommender = collaborative_recommender
        self.content_weight = content_weight
        self.collaborative_weight = collaborative_weight

    def recommend_for_user(
        self,
        user_id: int | None = None,
        n: int = 10,
        candidate_pool_size: int = 100,
        seed_movie_title: str | None = None
    ) -> list[dict]:
        """Generate top N hybrid recommendations.

        Cold-Start & User Rules:
        - Known user with ratings: generates hybrid recommendations combining content + collaborative.
        - Unknown/new user WITHOUT seed_movie_title: raises ValueError explaining personalization requires user ratings or seed movie.
        - Unknown/new user WITH seed_movie_title: allows content-based fallback using Phase 3 recommender.
        - Unknown seed movie: raises ValueError.

        Args:
            user_id: Target user ID integer (or None if unauthenticated user with seed_movie_title).
            n: Number of recommendations to return (must be positive integer).
            candidate_pool_size: Number of raw candidate items to pull from each recommender.
            seed_movie_title: Optional seed movie title for content context / cold-start fallback.

        Returns:
            List of dictionary items containing hybrid recommendations and score breakdown.
        """
        if n <= 0 or not isinstance(n, int):
            raise ValueError(f"n must be a positive integer, got {n}.")

        if candidate_pool_size <= 0 or not isinstance(candidate_pool_size, int):
            raise ValueError(f"candidate_pool_size must be a positive integer, got {candidate_pool_size}.")

        # Determine user existence in collaborative model
        is_known_collab_user = False
        user_rated_movie_ids = set()

        if user_id is not None:
            try:
                self.collaborative_recommender.get_user_index(user_id)
                is_known_collab_user = True
                user_rated_movie_ids = self.collaborative_recommender.user_rated_movies.get(int(user_id), set())
            except ValueError:
                is_known_collab_user = False

        # Check cold-start requirements
        if not is_known_collab_user and (seed_movie_title is None or not str(seed_movie_title).strip()):
            raise ValueError(
                f"Personalization for user '{user_id}' requires either existing user ratings "
                "or a valid seed_movie_title context."
            )

        # Validate seed_movie_title if provided
        if seed_movie_title is not None and str(seed_movie_title).strip():
            # Will raise ValueError if seed movie does not exist
            self.content_recommender.get_movie_index(seed_movie_title)

        # Candidate Retrieval
        # 1. Collaborative Candidates
        raw_collab_recs = []
        if is_known_collab_user:
            raw_collab_recs = self.collaborative_recommender.recommend_for_user(
                user_id=int(user_id),
                n=candidate_pool_size
            )

        # 2. Content Candidates
        raw_content_recs = []
        content_seed = seed_movie_title

        # If no explicit seed_movie_title provided, infer seed from known user's highest rated movie
        if (content_seed is None or not str(content_seed).strip()) and is_known_collab_user:
            user_ratings = self.collaborative_recommender.user_ratings_map.get(int(user_id), {})
            if user_ratings:
                # Find movie with highest rating from user's history
                best_rated_mid = max(user_ratings, key=user_ratings.get)
                # Lookup title from metadata
                if self.content_recommender.movies_df is not None:
                    match = self.content_recommender.movies_df[
                        self.content_recommender.movies_df["movieId"] == best_rated_mid
                    ]
                    if not match.empty:
                        content_seed = str(match.iloc[0]["title"])

        if content_seed is not None and str(content_seed).strip():
            try:
                raw_content_recs = self.content_recommender.recommend(
                    movie_title=content_seed,
                    n=candidate_pool_size
                )
            except ValueError:
                raw_content_recs = []

        # Independent Min-Max Normalization of Raw Candidate Scores BEFORE Merging & Missing-Score Policy
        # Content Normalization
        content_norm_map: dict[int, float] = {}
        if raw_content_recs:
            raw_c_scores = [item["similarity_score"] for item in raw_content_recs]
            norm_c_scores = min_max_normalize(raw_c_scores)
            for item, norm_val in zip(raw_content_recs, norm_c_scores):
                content_norm_map[item["movieId"]] = norm_val

        # Collaborative Normalization
        collab_norm_map: dict[int, float] = {}
        if raw_collab_recs:
            raw_cf_scores = [item["score"] for item in raw_collab_recs]
            norm_cf_scores = min_max_normalize(raw_cf_scores)
            for item, norm_val in zip(raw_collab_recs, norm_cf_scores):
                collab_norm_map[item["movieId"]] = norm_val

        # Candidate Merging
        # Merge all unique movieIds present in either content or collaborative candidates
        candidates_map: dict[int, dict] = {}

        for item in raw_content_recs:
            mid = item["movieId"]
            candidates_map[mid] = {
                "movieId": mid,
                "title": item["title"],
                "genres": item["genres"],
                "content_score": item["similarity_score"],
                "collaborative_score": None,
            }

        for item in raw_collab_recs:
            mid = item["movieId"]
            if mid in candidates_map:
                candidates_map[mid]["collaborative_score"] = item["score"]
                # Update title/genres if missing
                if not candidates_map[mid]["title"] and "title" in item:
                    candidates_map[mid]["title"] = item["title"]
                if not candidates_map[mid]["genres"] and "genres" in item:
                    candidates_map[mid]["genres"] = item["genres"]
            else:
                title_val = item.get("title", f"Movie {mid}")
                genres_val = item.get("genres", "(no genres listed)")
                candidates_map[mid] = {
                    "movieId": mid,
                    "title": title_val,
                    "genres": genres_val,
                    "content_score": None,
                    "collaborative_score": item["score"],
                }

        # Apply Missing-Score Policy & Calculate Weighted Hybrid Scores
        scored_candidates = []
        for mid, cand in candidates_map.items():
            # MANDATORY RULE: Already-Rated Movie Filtering
            if mid in user_rated_movie_ids:
                continue

            # Normalized scores (missing source receives 0.0)
            norm_c = content_norm_map.get(mid, 0.0)
            norm_cf = collab_norm_map.get(mid, 0.0)

            hybrid_score = (self.content_weight * norm_c) + (self.collaborative_weight * norm_cf)

            cand_entry = {
                "movieId": mid,
                "title": cand["title"],
                "genres": cand["genres"],
                "content_score": cand["content_score"],
                "collaborative_score": cand["collaborative_score"],
                "normalized_content_score": round(float(norm_c), 4),
                "normalized_collaborative_score": round(float(norm_cf), 4),
                "hybrid_score": round(float(hybrid_score), 4)
            }
            scored_candidates.append(cand_entry)

        # Deterministic Ranking: Primary hybrid_score DESC, Secondary movieId ASC
        scored_candidates.sort(key=lambda x: (-x["hybrid_score"], x["movieId"]))

        # Return top N candidates
        return scored_candidates[:n]

    def explain_hybrid_recommendation(
        self,
        user_id: int | None,
        movie_id: int,
        seed_movie_title: str | None = None
    ) -> dict:
        """Provide a factual explanation of a hybrid recommendation based on real model signals.

        Args:
            user_id: Target user ID integer.
            movie_id: Recommended candidate movie ID integer.
            seed_movie_title: Optional seed movie title context.

        Returns:
            Dictionary containing explanation details and text.
        """
        recs = self.recommend_for_user(
            user_id=user_id,
            n=10000,
            seed_movie_title=seed_movie_title
        )

        match = [r for r in recs if r["movieId"] == movie_id]
        if not match:
            raise ValueError(f"Movie ID {movie_id} is not in current hybrid recommendation candidates.")

        rec = match[0]
        c_score = rec["content_score"]
        cf_score = rec["collaborative_score"]
        norm_c = rec["normalized_content_score"]
        norm_cf = rec["normalized_collaborative_score"]
        h_score = rec["hybrid_score"]

        content_contrib = self.content_weight * norm_c
        collab_contrib = self.collaborative_weight * norm_cf

        signals_desc = []
        if c_score is not None:
            signals_desc.append(
                f"Content signal (raw: {c_score:.4f}, norm: {norm_c:.4f}) contributed {content_contrib:.4f}"
            )
        else:
            signals_desc.append("Content signal was unavailable (norm: 0.0000)")

        if cf_score is not None:
            signals_desc.append(
                f"Collaborative signal (raw: {cf_score:.4f}, norm: {norm_cf:.4f}) contributed {collab_contrib:.4f}"
            )
        else:
            signals_desc.append("Collaborative signal was unavailable (norm: 0.0000)")

        explanation_text = (
            f"Recommended '{rec['title']}' with Hybrid Score {h_score:.4f} "
            f"(Content Weight: {self.content_weight:.2f}, Collaborative Weight: {self.collaborative_weight:.2f}). "
            + "; ".join(signals_desc) + "."
        )

        return {
            "user_id": user_id,
            "movie_id": movie_id,
            "title": rec["title"],
            "hybrid_score": h_score,
            "content_score": c_score,
            "collaborative_score": cf_score,
            "normalized_content_score": norm_c,
            "normalized_collaborative_score": norm_cf,
            "content_contribution": round(float(content_contrib), 4),
            "collaborative_contribution": round(float(collab_contrib), 4),
            "explanation": explanation_text
        }
