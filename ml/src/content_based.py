"""Content-Based Recommendation System module using TF-IDF and Cosine Similarity."""

import logging
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def prepare_content_text(row: pd.Series) -> str:
    """Construct a clean, combined textual representation from movie metadata.

    Combines title (with special character cleanup) and genres string.

    Example:
        Title: 'Toy Story (1995)'
        Genres: 'Adventure|Animation|Children|Comedy|Fantasy'
        Output: 'toy story adventure animation children comedy fantasy'

    Args:
        row: A pandas Series representing a movie row (with 'title' and 'genres').

    Returns:
        Combined textual representation string.
    """
    title = str(row.get("title", "")) if pd.notnull(row.get("title")) else ""
    genres = str(row.get("genres", "")) if pd.notnull(row.get("genres")) else ""

    # Clean title text (remove parentheses around release year for clean tokenization if present)
    title_clean = re.sub(r"\(.*?\)", "", title).strip()

    # Process genres: replace '|' with space, ignore '(no genres listed)'
    if genres == "(no genres listed)" or not genres:
        genres_clean = ""
    else:
        genres_clean = genres.replace("|", " ").strip()

    combined = f"{title_clean} {genres_clean}".strip().lower()
    return combined if combined else "unknown"


class ContentBasedRecommender:
    """Content-based movie recommendation engine using TF-IDF and Cosine Similarity."""

    def __init__(self, stop_words: str | list[str] | None = "english", ngram_range: tuple[int, int] = (1, 2)):
        """Initialize ContentBasedRecommender.

        Args:
            stop_words: Stop words configuration for TfidfVectorizer. Defaults to 'english'.
            ngram_range: Token n-gram range for TfidfVectorizer. Defaults to (1, 2).
        """
        self.stop_words = stop_words
        self.ngram_range = ngram_range
        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix: np.ndarray | None = None
        self.similarity_matrix: np.ndarray | None = None
        self.movies_df: pd.DataFrame | None = None
        self.title_to_index: dict[str, int] = {}
        self.is_fitted: bool = False

    def fit(self, movies_df: pd.DataFrame) -> "ContentBasedRecommender":
        """Fit the TF-IDF vectorizer and compute pairwise cosine similarity matrix.

        Args:
            movies_df: Pandas DataFrame containing movie catalog metadata ('movieId', 'title', 'genres').

        Returns:
            Fitted instance of ContentBasedRecommender.

        Raises:
            ValueError: If movies_df is empty or missing required columns.
        """
        if movies_df is None or movies_df.empty:
            raise ValueError("Input movies_df cannot be empty or None.")

        if "title" not in movies_df.columns:
            raise ValueError("Input movies_df must contain a 'title' column.")

        # Make an isolated copy and reset index for deterministic integer positioning
        self.movies_df = movies_df.copy().reset_index(drop=True)

        # Build normalized title lookup mapping (deterministic: first occurrence wins)
        self.title_to_index = {}
        for idx, raw_title in enumerate(self.movies_df["title"]):
            if pd.notnull(raw_title) and str(raw_title).strip():
                norm_title = str(raw_title).strip().lower()
                if norm_title not in self.title_to_index:
                    self.title_to_index[norm_title] = idx

        # Generate combined content text series for TF-IDF vectorization
        content_series = self.movies_df.apply(prepare_content_text, axis=1)

        # Build TF-IDF representation
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=self.stop_words,
            ngram_range=self.ngram_range,
            token_pattern=r"(?u)\b\w+\b"
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(content_series)

        # Compute full pairwise cosine similarity matrix
        # Dense float32 representation for MovieLens dataset (~9,742 movies) is efficient and fast
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix).astype(np.float32)

        self.is_fitted = True
        logger.info(
            "ContentBasedRecommender fitted successfully on %d movies (TF-IDF shape: %s).",
            len(self.movies_df),
            self.tfidf_matrix.shape
        )
        return self

    def get_movie_index(self, movie_title: str) -> int:
        """Find the row index of a movie by title (case-insensitive and whitespace-normalized).

        Args:
            movie_title: Movie title to search.

        Returns:
            Row index integer of the movie in the internal catalog.

        Raises:
            ValueError: If recommender is not fitted, title is empty, or title is not found.
        """
        if not self.is_fitted:
            raise ValueError("Recommender is not fitted yet. Call fit() first.")

        if not isinstance(movie_title, str) or not movie_title.strip():
            raise ValueError("Movie title must be a non-empty string.")

        norm_title = movie_title.strip().lower()
        if norm_title not in self.title_to_index:
            raise ValueError(f"Movie '{movie_title}' not found in dataset catalog.")

        return self.title_to_index[norm_title]

    def get_similarity(self, movie_title_a: str, movie_title_b: str) -> float:
        """Calculate the pairwise cosine similarity score between two movies.

        Args:
            movie_title_a: Title of first movie.
            movie_title_b: Title of second movie.

        Returns:
            Cosine similarity score as float in range [0.0, 1.0].
        """
        idx_a = self.get_movie_index(movie_title_a)
        idx_b = self.get_movie_index(movie_title_b)

        score = float(self.similarity_matrix[idx_a, idx_b])
        return float(np.clip(score, 0.0, 1.0))

    def recommend(self, movie_title: str, n: int = 10) -> list[dict]:
        """Generate top N content-based movie recommendations for a given movie.

        Args:
            movie_title: Source movie title.
            n: Number of recommendations to return (must be positive integer).

        Returns:
            List of dictionaries containing recommendation details:
            [
                {
                    "movieId": int,
                    "title": str,
                    "genres": str,
                    "similarity_score": float
                }, ...
            ]

        Raises:
            ValueError: If n <= 0, movie_title is invalid, or movie is not found.
        """
        if n <= 0 or not isinstance(n, int):
            raise ValueError(f"n must be a positive integer, got {n}.")

        source_idx = self.get_movie_index(movie_title)
        sim_scores = self.similarity_matrix[source_idx]

        # Candidate items (excluding source movie itself)
        candidates = []
        for idx, score in enumerate(sim_scores):
            if idx == source_idx:
                continue
            movie_id = int(self.movies_df.iloc[idx]["movieId"])
            candidates.append((idx, float(score), movie_id))

        # Primary sort: similarity_score DESC; Secondary sort (tie-breaker): movieId ASC
        candidates.sort(key=lambda x: (-x[1], x[2]))

        # Cap recommendations to requested n or available candidate count
        top_candidates = candidates[:n]

        results = []
        for idx, score, _ in top_candidates:
            row = self.movies_df.iloc[idx]
            results.append({
                "movieId": int(row["movieId"]),
                "title": str(row["title"]),
                "genres": str(row["genres"]) if pd.notnull(row["genres"]) else "(no genres listed)",
                "similarity_score": round(float(np.clip(score, 0.0, 1.0)), 4)
            })

        return results

    def explain_recommendation(self, source_movie: str, recommended_movie: str) -> str:
        """Provide a factual, metadata-grounded explanation for why a movie was recommended.

        Args:
            source_movie: Source movie title.
            recommended_movie: Recommended movie title.

        Returns:
            Factual explanation string detailing shared genres and similarity score.
        """
        source_idx = self.get_movie_index(source_movie)
        rec_idx = self.get_movie_index(recommended_movie)

        source_row = self.movies_df.iloc[source_idx]
        rec_row = self.movies_df.iloc[rec_idx]

        sim_score = self.get_similarity(source_movie, recommended_movie)

        # Extract genres lists
        def parse_genres(genres_val) -> set[str]:
            if not isinstance(genres_val, str) or genres_val == "(no genres listed)" or not genres_val.strip():
                return set()
            return {g.strip() for g in genres_val.split("|") if g.strip()}

        source_genres = parse_genres(source_row["genres"])
        rec_genres = parse_genres(rec_row["genres"])
        shared_genres = sorted(list(source_genres & rec_genres))

        if shared_genres:
            genres_formatted = ", ".join([f"'{g}'" for g in shared_genres])
            explanation = (
                f"Recommended '{rec_row['title']}' because it shares {len(shared_genres)} genre(s) "
                f"({genres_formatted}) with '{source_row['title']}'. "
                f"Content similarity score: {sim_score:.4f}."
            )
        else:
            explanation = (
                f"Recommended '{rec_row['title']}' based on textual title/content metadata similarity "
                f"with '{source_row['title']}'. Content similarity score: {sim_score:.4f}."
            )

        return explanation
