"""Item-Based Collaborative Filtering Recommendation System module."""

import logging
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class ItemBasedCollaborativeRecommender:
    """Item-Based Collaborative Filtering Recommendation Engine.

    Calculates movie-to-movie similarity strictly based on user rating patterns.
    Generates personalized user recommendations using mean-centered, normalized scoring.
    """

    def __init__(self, epsilon: float = 1e-9):
        """Initialize ItemBasedCollaborativeRecommender.

        Args:
            epsilon: Small constant to prevent division by zero in score normalization.
        """
        self.epsilon = epsilon

        self.user_to_idx: dict[int, int] = {}
        self.idx_to_user: dict[int, int] = {}
        self.movie_to_idx: dict[int, int] = {}
        self.idx_to_movie: dict[int, int] = {}

        self.user_item_matrix: csr_matrix | None = None
        self.similarity_matrix: np.ndarray | None = None

        self.user_means: dict[int, float] = {}
        self.user_rated_movies: dict[int, set[int]] = {}
        self.user_ratings_map: dict[int, dict[int, float]] = {}

        self.movies_df: pd.DataFrame | None = None
        self.is_fitted: bool = False

    def fit(
        self,
        ratings_df: pd.DataFrame,
        movies_df: pd.DataFrame | None = None
    ) -> "ItemBasedCollaborativeRecommender":
        """Fit the collaborative filtering model using supplied ratings data ONLY.

        Args:
            ratings_df: DataFrame containing ['userId', 'movieId', 'rating'].
            movies_df: Optional DataFrame containing movie metadata for display lookups.

        Returns:
            Fitted instance of ItemBasedCollaborativeRecommender.

        Raises:
            ValueError: If required columns are missing or input is invalid.
        """
        if ratings_df is None or ratings_df.empty:
            raise ValueError("Input ratings_df cannot be empty or None.")

        required_cols = {"userId", "movieId", "rating"}
        if not required_cols.issubset(ratings_df.columns):
            missing = required_cols - set(ratings_df.columns)
            raise ValueError(f"Ratings DataFrame missing required columns: {missing}")

        # Data Leakage Protection: Use ONLY ratings_df supplied to fit()
        df = ratings_df[["userId", "movieId", "rating"]].dropna().copy()

        # Handle duplicate (userId, movieId) records deterministically via mean rating
        if df.duplicated(subset=["userId", "movieId"]).any():
            logger.info("Duplicate (userId, movieId) records detected. Aggregating via mean rating.")
            df = df.groupby(["userId", "movieId"])["rating"].mean().reset_index()

        # Store movie metadata if provided for display lookups
        if movies_df is not None:
            self.movies_df = movies_df.copy()

        # Create unique mappings
        unique_users = sorted(df["userId"].unique().tolist())
        unique_movies = sorted(df["movieId"].unique().tolist())

        self.user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.idx_to_user = {idx: uid for idx, uid in enumerate(unique_users)}
        self.movie_to_idx = {mid: idx for idx, mid in enumerate(unique_movies)}
        self.idx_to_movie = {idx: mid for idx, mid in enumerate(unique_movies)}

        num_users = len(unique_users)
        num_movies = len(unique_movies)

        # Build sparse CSR user-item matrix
        row_indices = [self.user_to_idx[uid] for uid in df["userId"]]
        col_indices = [self.movie_to_idx[mid] for mid in df["movieId"]]
        values = df["rating"].values.astype(np.float32)

        self.user_item_matrix = csr_matrix(
            (values, (row_indices, col_indices)),
            shape=(num_users, num_movies),
            dtype=np.float32
        )

        # Precompute user means and user rated movie maps for fast recommendation scoring
        for uid, group in df.groupby("userId"):
            self.user_means[uid] = float(group["rating"].mean())
            self.user_rated_movies[uid] = set(group["movieId"].astype(int))
            self.user_ratings_map[uid] = dict(zip(group["movieId"].astype(int), group["rating"].astype(float)))

        # Compute Item-Item Collaborative Cosine Similarity Matrix
        # Item vectors are the columns of user_item_matrix (shape: num_users x num_movies)
        item_matrix_transposed = self.user_item_matrix.T  # shape: num_movies x num_users
        raw_sim = cosine_similarity(item_matrix_transposed, item_matrix_transposed).astype(np.float32)

        # Ensure diagonal is exactly 1.0 and clip similarity to [0.0, 1.0]
        np.fill_diagonal(raw_sim, 1.0)
        self.similarity_matrix = np.clip(raw_sim, 0.0, 1.0)

        self.is_fitted = True
        logger.info(
            "ItemBasedCollaborativeRecommender fitted on %d users, %d movies, %d ratings.",
            num_users,
            num_movies,
            len(df)
        )
        return self

    def get_user_index(self, user_id: int) -> int:
        """Get internal row index for user_id."""
        if not self.is_fitted:
            raise ValueError("Recommender is not fitted yet. Call fit() first.")
        try:
            uid_int = int(user_id)
        except (ValueError, TypeError):
            raise ValueError(f"User ID '{user_id}' must be an integer.")

        if uid_int not in self.user_to_idx:
            raise ValueError(f"User ID {user_id} not found in training dataset.")
        return self.user_to_idx[uid_int]

    def get_movie_index(self, movie_id: int) -> int:
        """Get internal column index for movie_id."""
        if not self.is_fitted:
            raise ValueError("Recommender is not fitted yet. Call fit() first.")
        try:
            mid_int = int(movie_id)
        except (ValueError, TypeError):
            raise ValueError(f"Movie ID '{movie_id}' must be an integer.")

        if mid_int not in self.movie_to_idx:
            raise ValueError(f"Movie ID {movie_id} not found in dataset catalog.")
        return self.movie_to_idx[mid_int]

    def get_similarity(self, movie_id_a: int, movie_id_b: int) -> float:
        """Get item-item collaborative rating similarity between two movies."""
        idx_a = self.get_movie_index(movie_id_a)
        idx_b = self.get_movie_index(movie_id_b)

        sim = float(self.similarity_matrix[idx_a, idx_b])
        return float(np.clip(sim, 0.0, 1.0))

    def recommend_similar_movies(self, movie_id: int, n: int = 10) -> list[dict]:
        """Find movies collaboratively similar to a given movie based on user rating patterns.

        Args:
            movie_id: Target movie ID integer.
            n: Number of similar movies to return.

        Returns:
            List of dictionaries containing similar movie info.
        """
        if n <= 0 or not isinstance(n, int):
            raise ValueError(f"n must be a positive integer, got {n}.")

        target_idx = self.get_movie_index(movie_id)
        target_mid = self.idx_to_movie[target_idx]
        sim_scores = self.similarity_matrix[target_idx]

        candidates = []
        for idx, score in enumerate(sim_scores):
            mid = self.idx_to_movie[idx]
            if mid == target_mid:
                continue
            candidates.append((mid, float(score)))

        # Primary sort: similarity DESC, Secondary sort: movieId ASC
        candidates.sort(key=lambda x: (-x[1], x[0]))
        top_candidates = candidates[:n]

        results = []
        for mid, score in top_candidates:
            item = {
                "movieId": mid,
                "similarity_score": round(float(np.clip(score, 0.0, 1.0)), 4)
            }
            if self.movies_df is not None:
                match = self.movies_df[self.movies_df["movieId"] == mid]
                if not match.empty:
                    item["title"] = str(match.iloc[0]["title"])
                    item["genres"] = str(match.iloc[0]["genres"])
            results.append(item)

        return results

    def recommend_for_user(self, user_id: int, n: int = 10) -> list[dict]:
        """Generate top N personalized recommendations for a user.

        Uses normalized mean-centered collaborative scoring:
        Score(u, c) = Σ [ S(i, c) * (r(u, i) - mean_u) ] / [ Σ |S(i, c)| + epsilon ]

        EXCLUDES all movies already rated by user_id in training dataset.

        Args:
            user_id: Target user ID integer.
            n: Number of recommendations.

        Returns:
            List of dictionaries containing recommendation scores and metadata.
        """
        if n <= 0 or not isinstance(n, int):
            raise ValueError(f"n must be a positive integer, got {n}.")

        self.get_user_index(user_id)  # Validates user existence

        user_mean = self.user_means[user_id]
        rated_movies = self.user_rated_movies[user_id]
        user_ratings = self.user_ratings_map[user_id]

        if not rated_movies:
            return []

        # Maps rated movieId -> (col_idx, rating_signal = rating - user_mean)
        user_signals = {
            mid: (self.movie_to_idx[mid], rating - user_mean)
            for mid, rating in user_ratings.items()
            if mid in self.movie_to_idx
        }

        num_all_movies = len(self.idx_to_movie)
        candidate_numerators = np.zeros(num_all_movies, dtype=np.float64)
        candidate_denominators = np.zeros(num_all_movies, dtype=np.float64)

        for mid, (col_idx, rating_signal) in user_signals.items():
            sim_vector = self.similarity_matrix[col_idx]  # array of length num_all_movies
            candidate_numerators += sim_vector * rating_signal
            candidate_denominators += np.abs(sim_vector)

        candidate_scores = candidate_numerators / (candidate_denominators + self.epsilon)

        candidates = []
        for idx, score in enumerate(candidate_scores):
            mid = self.idx_to_movie[idx]
            # MANDATORY RULE: Exclude movies already rated by the user
            if mid in rated_movies:
                continue
            candidates.append((mid, float(score)))

        # Primary sort: score DESC, Secondary sort: movieId ASC
        candidates.sort(key=lambda x: (-x[1], x[0]))
        top_candidates = candidates[:n]

        results = []
        for mid, score in top_candidates:
            item = {
                "movieId": mid,
                "score": round(float(score), 4)
            }
            if self.movies_df is not None:
                match = self.movies_df[self.movies_df["movieId"] == mid]
                if not match.empty:
                    item["title"] = str(match.iloc[0]["title"])
                    item["genres"] = str(match.iloc[0]["genres"])
            results.append(item)

        return results

    def explain_user_recommendation(self, user_id: int, movie_id: int) -> dict:
        """Provide a factual explanation for why a movie was recommended to a user.

        Identifies the source movie rated by the user that contributed most positively
        to the candidate's normalized score.

        Args:
            user_id: User ID integer.
            movie_id: Candidate movie ID integer.

        Returns:
            Dictionary containing explanation details and text.
        """
        self.get_user_index(user_id)
        cand_idx = self.get_movie_index(movie_id)

        user_mean = self.user_means[user_id]
        user_ratings = self.user_ratings_map[user_id]
        rated_movies = self.user_rated_movies[user_id]

        if movie_id in rated_movies:
            return {
                "user_id": user_id,
                "movie_id": movie_id,
                "explanation": f"Movie {movie_id} was already rated by User {user_id}."
            }

        best_source_mid = None
        best_contrib = -1e9
        best_sim = 0.0
        best_user_rating = 0.0

        for mid, rating in user_ratings.items():
            if mid not in self.movie_to_idx:
                continue
            src_idx = self.movie_to_idx[mid]
            sim = float(self.similarity_matrix[src_idx, cand_idx])
            rating_signal = rating - user_mean
            contrib = sim * rating_signal

            if contrib > best_contrib:
                best_contrib = contrib
                best_source_mid = mid
                best_sim = sim
                best_user_rating = rating

        rec_scores = self.recommend_for_user(user_id, n=len(self.idx_to_movie))
        score_match = [r["score"] for r in rec_scores if r["movieId"] == movie_id]
        cand_score = score_match[0] if score_match else 0.0

        source_title = f"Movie {best_source_mid}"
        cand_title = f"Movie {movie_id}"

        if self.movies_df is not None:
            src_match = self.movies_df[self.movies_df["movieId"] == best_source_mid]
            if not src_match.empty:
                source_title = str(src_match.iloc[0]["title"])
            cand_match = self.movies_df[self.movies_df["movieId"] == movie_id]
            if not cand_match.empty:
                cand_title = str(cand_match.iloc[0]["title"])

        sentiment = "above" if best_user_rating >= user_mean else "below"

        explanation_text = (
            f"Recommended '{cand_title}' (score: {cand_score:.4f}) to User {user_id} because User {user_id} "
            f"rated '{source_title}' {best_user_rating:.1f} ({sentiment} user average of {user_mean:.2f}). "
            f"Item-item rating similarity between '{source_title}' and '{cand_title}' is {best_sim:.4f}."
        )

        return {
            "user_id": user_id,
            "candidate_movie_id": movie_id,
            "candidate_title": cand_title,
            "candidate_score": cand_score,
            "top_source_movie_id": best_source_mid,
            "top_source_title": source_title,
            "user_source_rating": best_user_rating,
            "user_mean_rating": user_mean,
            "item_similarity": best_sim,
            "explanation": explanation_text
        }
