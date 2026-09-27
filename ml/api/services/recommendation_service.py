"""Recommendation Service Layer wrapping Phase 3, 4, 5 ML models."""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from ml.src.content_based import ContentBasedRecommender
from ml.src.collaborative_filtering import ItemBasedCollaborativeRecommender
from ml.src.hybrid_recommender import HybridRecommender
from ml.api.exceptions import (
    MovieNotFoundException,
    UserNotFoundException,
    ModelNotReadyException,
    InvalidRecommendationRequestException
)
from ml.api.schemas.movie import MovieResponse, MovieSearchResponse
from ml.api.schemas.recommendation import RecommendationItem, RecommendationResponse

logger = logging.getLogger(__name__)


def parse_genres_list(genres_val: Any) -> List[str]:
    """Parse pipe-separated genres string into a clean list of strings."""
    if not genres_val or pd.isna(genres_val) or str(genres_val) == "(no genres listed)":
        return ["(no genres listed)"]
    return [g.strip() for g in str(genres_val).split("|") if g.strip()]


class RecommendationService:
    """Service layer delegating to fitted Phase 3, Phase 4, and Phase 5 ML recommenders."""

    def __init__(
        self,
        content_recommender: Optional[ContentBasedRecommender],
        collaborative_recommender: Optional[ItemBasedCollaborativeRecommender],
        hybrid_recommender: Optional[HybridRecommender],
        movies_df: Optional[pd.DataFrame]
    ):
        self.content_recommender = content_recommender
        self.collaborative_recommender = collaborative_recommender
        self.hybrid_recommender = hybrid_recommender
        self.movies_df = movies_df

    def _ensure_models_ready(self):
        if (
            self.content_recommender is None
            or not self.content_recommender.is_fitted
            or self.collaborative_recommender is None
            or not self.collaborative_recommender.is_fitted
            or self.hybrid_recommender is None
            or self.movies_df is None
        ):
            raise ModelNotReadyException()

    def get_movie_details(self, movie_id: int) -> MovieResponse:
        self._ensure_models_ready()
        match = self.movies_df[self.movies_df["movieId"] == movie_id]
        if match.empty:
            raise MovieNotFoundException(movie_id)

        row = match.iloc[0]
        year_val = int(row["year"]) if "year" in row and pd.notnull(row["year"]) else None

        return MovieResponse(
            movie_id=int(row["movieId"]),
            title=str(row["title"]),
            genres=parse_genres_list(row["genres"]),
            year=year_val
        )

    def search_movies(self, query: str, limit: int = 10) -> MovieSearchResponse:
        self._ensure_models_ready()
        if not query or not query.strip():
            raise InvalidRecommendationRequestException("Search query 'q' must not be empty.")

        query_clean = query.strip().lower()
        matches = self.movies_df[
            self.movies_df["title"].astype(str).str.lower().str.contains(query_clean, regex=False)
        ].copy()

        # Deterministic sort: title ASC, movieId ASC
        matches.sort_values(by=["title", "movieId"], inplace=True)
        total_matches = len(matches)
        capped_matches = matches.head(limit)

        movie_items = []
        for _, row in capped_matches.iterrows():
            year_val = int(row["year"]) if "year" in row and pd.notnull(row["year"]) else None
            movie_items.append(
                MovieResponse(
                    movie_id=int(row["movieId"]),
                    title=str(row["title"]),
                    genres=parse_genres_list(row["genres"]),
                    year=year_val
                )
            )

        return MovieSearchResponse(
            query=query,
            total_matches=total_matches,
            limit=limit,
            movies=movie_items
        )

    def get_content_recommendations(self, movie_id: int, top_k: int = 10) -> RecommendationResponse:
        self._ensure_models_ready()

        # Lookup movie title
        match = self.movies_df[self.movies_df["movieId"] == movie_id]
        if match.empty:
            raise MovieNotFoundException(movie_id)
        seed_title = str(match.iloc[0]["title"])

        try:
            raw_recs = self.content_recommender.recommend(seed_title, n=top_k)
        except ValueError as e:
            raise MovieNotFoundException(movie_id)

        items = []
        for r in raw_recs:
            explanation = None
            try:
                explanation = self.content_recommender.explain_recommendation(seed_title, r["title"])
            except Exception:
                explanation = None

            items.append(
                RecommendationItem(
                    movie_id=int(r["movieId"]),
                    title=str(r["title"]),
                    genres=parse_genres_list(r["genres"]),
                    score=float(r["similarity_score"]),
                    content_score=float(r["similarity_score"]),
                    collaborative_score=None,
                    normalized_content_score=float(r["similarity_score"]),
                    normalized_collaborative_score=None,
                    explanation=explanation
                )
            )

        return RecommendationResponse(
            model="content_based",
            seed_movie_id=movie_id,
            seed_movie_title=seed_title,
            recommendations=items
        )

    def get_collaborative_recommendations(self, user_id: int, top_k: int = 10) -> RecommendationResponse:
        self._ensure_models_ready()

        # Check user existence
        try:
            self.collaborative_recommender.get_user_index(user_id)
        except ValueError:
            raise UserNotFoundException(user_id)

        raw_recs = self.collaborative_recommender.recommend_for_user(user_id, n=top_k)

        items = []
        for r in raw_recs:
            explanation = None
            try:
                exp_dict = self.collaborative_recommender.explain_user_recommendation(user_id, r["movieId"])
                explanation = exp_dict.get("explanation")
            except Exception:
                explanation = None

            items.append(
                RecommendationItem(
                    movie_id=int(r["movieId"]),
                    title=str(r.get("title", f"Movie {r['movieId']}")),
                    genres=parse_genres_list(r.get("genres")),
                    score=float(r["score"]),
                    content_score=None,
                    collaborative_score=float(r["score"]),
                    normalized_content_score=None,
                    normalized_collaborative_score=None,
                    explanation=explanation
                )
            )

        return RecommendationResponse(
            model="collaborative_filtering",
            user_id=user_id,
            recommendations=items
        )

    def get_hybrid_recommendations(
        self,
        user_id: Optional[int] = None,
        movie_id: Optional[int] = None,
        top_k: int = 10,
        content_weight: float = 0.5,
        collaborative_weight: float = 0.5
    ) -> RecommendationResponse:
        self._ensure_models_ready()

        # Re-initialize hybrid recommender if custom weights requested
        if (content_weight, collaborative_weight) != (
            self.hybrid_recommender.content_weight,
            self.hybrid_recommender.collaborative_weight
        ):
            try:
                active_hybrid = HybridRecommender(
                    content_recommender=self.content_recommender,
                    collaborative_recommender=self.collaborative_recommender,
                    content_weight=content_weight,
                    collaborative_weight=collaborative_weight
                )
            except ValueError as e:
                raise InvalidRecommendationRequestException(str(e))
        else:
            active_hybrid = self.hybrid_recommender

        seed_title = None
        if movie_id is not None:
            match = self.movies_df[self.movies_df["movieId"] == movie_id]
            if match.empty:
                raise MovieNotFoundException(movie_id)
            seed_title = str(match.iloc[0]["title"])

        try:
            raw_recs = active_hybrid.recommend_for_user(
                user_id=user_id,
                n=top_k,
                candidate_pool_size=100,
                seed_movie_title=seed_title
            )
        except ValueError as e:
            err_msg = str(e)
            if "not found in training dataset" in err_msg:
                raise UserNotFoundException(user_id)
            elif "not found in dataset catalog" in err_msg:
                raise MovieNotFoundException(movie_id if movie_id else seed_title)
            else:
                raise InvalidRecommendationRequestException(err_msg)

        items = []
        for r in raw_recs:
            explanation = None
            try:
                exp_dict = active_hybrid.explain_hybrid_recommendation(
                    user_id=user_id,
                    movie_id=r["movieId"],
                    seed_movie_title=seed_title
                )
                explanation = exp_dict.get("explanation")
            except Exception:
                explanation = None

            c_raw = float(r["content_score"]) if r.get("content_score") is not None else None
            cf_raw = float(r["collaborative_score"]) if r.get("collaborative_score") is not None else None
            c_norm = float(r["normalized_content_score"]) if r.get("normalized_content_score") is not None else None
            cf_norm = float(r["normalized_collaborative_score"]) if r.get("normalized_collaborative_score") is not None else None

            items.append(
                RecommendationItem(
                    movie_id=int(r["movieId"]),
                    title=str(r["title"]),
                    genres=parse_genres_list(r["genres"]),
                    score=float(r["hybrid_score"]),
                    content_score=c_raw,
                    collaborative_score=cf_raw,
                    normalized_content_score=c_norm,
                    normalized_collaborative_score=cf_norm,
                    explanation=explanation
                )
            )

        return RecommendationResponse(
            model="hybrid",
            user_id=user_id,
            seed_movie_id=movie_id,
            seed_movie_title=seed_title,
            content_weight=content_weight,
            collaborative_weight=collaborative_weight,
            recommendations=items
        )
