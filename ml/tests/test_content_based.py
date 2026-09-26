"""Unit tests for ContentBasedRecommender module."""

import pytest
import numpy as np
import pandas as pd

from ml.src.content_based import ContentBasedRecommender, prepare_content_text


@pytest.fixture
def sample_movies_df() -> pd.DataFrame:
    """Create a sample movies catalog for testing."""
    return pd.DataFrame({
        "movieId": [1, 2, 3, 4, 5],
        "title": [
            "Toy Story (1995)",
            "Jumanji (1995)",
            "Grumpier Old Men (1995)",
            "Toy Story 2 (1999)",
            "No Genre Movie (2000)"
        ],
        "genres": [
            "Adventure|Animation|Children|Comedy|Fantasy",
            "Adventure|Children|Fantasy",
            "Comedy|Romance",
            "Adventure|Animation|Children|Comedy|Fantasy",
            "(no genres listed)"
        ]
    })


def test_prepare_content_text():
    """Test combined textual representation generation."""
    row = pd.Series({"title": "Toy Story (1995)", "genres": "Adventure|Animation|Comedy"})
    text = prepare_content_text(row)
    assert "toy story" in text
    assert "adventure animation comedy" in text


def test_fit_success(sample_movies_df):
    """Test 1: Recommender can fit successfully."""
    recommender = ContentBasedRecommender()
    recommender.fit(sample_movies_df)
    assert recommender.is_fitted is True


def test_tfidf_matrix_rows(sample_movies_df):
    """Test 2: TF-IDF matrix has the same number of rows as movies."""
    recommender = ContentBasedRecommender()
    recommender.fit(sample_movies_df)
    assert recommender.tfidf_matrix.shape[0] == len(sample_movies_df)


def test_tfidf_matrix_features(sample_movies_df):
    """Test 3: TF-IDF matrix has a non-zero feature dimension."""
    recommender = ContentBasedRecommender()
    recommender.fit(sample_movies_df)
    assert recommender.tfidf_matrix.shape[1] > 0


def test_exact_title_lookup(sample_movies_df):
    """Test 4: Exact title lookup works."""
    recommender = ContentBasedRecommender()
    recommender.fit(sample_movies_df)
    idx = recommender.get_movie_index("Toy Story (1995)")
    assert idx == 0


def test_case_normalization(sample_movies_df):
    """Test 5: Case normalization works for movie title lookup."""
    recommender = ContentBasedRecommender()
    recommender.fit(sample_movies_df)
    idx = recommender.get_movie_index("toy story (1995)")
    assert idx == 0


def test_whitespace_normalization(sample_movies_df):
    """Test 6: Whitespace normalization works for movie title lookup."""
    recommender = ContentBasedRecommender()
    recommender.fit(sample_movies_df)
    idx = recommender.get_movie_index("  Toy Story (1995)   ")
    assert idx == 0


def test_unknown_movie_raises_error(sample_movies_df):
    """Test 7: Unknown movie title raises a meaningful ValueError."""
    recommender = ContentBasedRecommender()
    recommender.fit(sample_movies_df)
    with pytest.raises(ValueError, match="not found in dataset catalog"):
        recommender.get_movie_index("Non Existent Movie (2025)")


def test_similarity_calculation(sample_movies_df):
    """Test 8: Similarity calculation works between two valid movies."""
    recommender = ContentBasedRecommender()
    recommender.fit(sample_movies_df)
    sim = recommender.get_similarity("Toy Story (1995)", "Toy Story 2 (1999)")
    assert isinstance(sim, float)


def test_similarity_bounds(sample_movies_df):
    """Test 9: Similarity is between 0.0 and 1.0."""
    recommender = ContentBasedRecommender()
    recommender.fit(sample_movies_df)
    sim = recommender.get_similarity("Toy Story (1995)", "Grumpier Old Men (1995)")
    assert 0.0 <= sim <= 1.0


def test_same_movie_similarity(sample_movies_df):
    """Test 10: Same movie has similarity approximately 1.0."""
    recommender = ContentBasedRecommender()
    recommender.fit(sample_movies_df)
    sim = recommender.get_similarity("Toy Story (1995)", "Toy Story (1995)")
    assert pytest.approx(sim, 0.001) == 1.0


def test_recommendation_count(sample_movies_df):
    """Test 11: Recommendations return requested number when enough movies exist."""
    recommender = ContentBasedRecommender()
    recommender.fit(sample_movies_df)
    recs = recommender.recommend("Toy Story (1995)", n=2)
    assert len(recs) == 2


def test_source_movie_excluded(sample_movies_df):
    """Test 12: Source movie itself is excluded from recommendations."""
    recommender = ContentBasedRecommender()
    recommender.fit(sample_movies_df)
    recs = recommender.recommend("Toy Story (1995)", n=4)
    rec_ids = [item["movieId"] for item in recs]
    assert 1 not in rec_ids  # movieId=1 is Toy Story (1995)


def test_scores_ordered_descending(sample_movies_df):
    """Test 13: Recommendation similarity scores are ordered descending."""
    recommender = ContentBasedRecommender()
    recommender.fit(sample_movies_df)
    recs = recommender.recommend("Toy Story (1995)", n=3)
    scores = [item["similarity_score"] for item in recs]
    assert scores == sorted(scores, reverse=True)


def test_recommendations_deterministic(sample_movies_df):
    """Test 14: Recommendation order is deterministic across multiple calls."""
    recommender = ContentBasedRecommender()
    recommender.fit(sample_movies_df)
    recs_call_1 = recommender.recommend("Toy Story (1995)", n=3)
    recs_call_2 = recommender.recommend("Toy Story (1995)", n=3)
    assert recs_call_1 == recs_call_2


def test_invalid_n_raises_error(sample_movies_df):
    """Test 15: n <= 0 raises a meaningful ValueError."""
    recommender = ContentBasedRecommender()
    recommender.fit(sample_movies_df)
    with pytest.raises(ValueError, match="positive integer"):
        recommender.recommend("Toy Story (1995)", n=0)

    with pytest.raises(ValueError, match="positive integer"):
        recommender.recommend("Toy Story (1995)", n=-5)


def test_n_larger_than_catalog(sample_movies_df):
    """Test 16: n larger than catalog size is handled safely (caps result length)."""
    recommender = ContentBasedRecommender()
    recommender.fit(sample_movies_df)
    recs = recommender.recommend("Toy Story (1995)", n=100)
    assert len(recs) == 4  # Total catalog size is 5, excluding self leaves 4 candidates


def test_missing_genre_does_not_crash():
    """Test 17: Missing genre data (None / NaN) does not crash recommender."""
    df_missing_genre = pd.DataFrame({
        "movieId": [10, 20],
        "title": ["Movie A (2020)", "Movie B (2021)"],
        "genres": [None, "Action"]
    })
    recommender = ContentBasedRecommender()
    recommender.fit(df_missing_genre)
    recs = recommender.recommend("Movie A (2020)", n=1)
    assert len(recs) == 1


def test_empty_genre_does_not_crash(sample_movies_df):
    """Test 18: Empty/'(no genres listed)' genre data does not crash recommender."""
    recommender = ContentBasedRecommender()
    recommender.fit(sample_movies_df)
    recs = recommender.recommend("No Genre Movie (2000)", n=2)
    assert len(recs) == 2


def test_empty_title_raises_error(sample_movies_df):
    """Test 19: Missing or empty title raises ValueError."""
    recommender = ContentBasedRecommender()
    recommender.fit(sample_movies_df)
    with pytest.raises(ValueError, match="non-empty string"):
        recommender.get_movie_index("   ")


def test_duplicate_normalized_titles():
    """Test 20: Duplicate normalized titles are handled deterministically (first occurrence wins)."""
    df_duplicates = pd.DataFrame({
        "movieId": [100, 200],
        "title": ["Duplicate Title (2020)", "DUPLICATE TITLE (2020)"],
        "genres": ["Action", "Drama"]
    })
    recommender = ContentBasedRecommender()
    recommender.fit(df_duplicates)
    idx = recommender.get_movie_index("duplicate title (2020)")
    assert idx == 0  # First occurrence (movieId 100) is selected deterministically


def test_explanation_contains_actual_metadata(sample_movies_df):
    """Test 21: Explanation contains actual shared metadata genres."""
    recommender = ContentBasedRecommender()
    recommender.fit(sample_movies_df)
    explanation = recommender.explain_recommendation("Toy Story (1995)", "Toy Story 2 (1999)")
    assert "Adventure" in explanation
    assert "Animation" in explanation


def test_explanation_contains_actual_similarity_score(sample_movies_df):
    """Test 22: Explanation contains the actual similarity score."""
    recommender = ContentBasedRecommender()
    recommender.fit(sample_movies_df)
    explanation = recommender.explain_recommendation("Toy Story (1995)", "Toy Story 2 (1999)")
    sim = recommender.get_similarity("Toy Story (1995)", "Toy Story 2 (1999)")
    assert f"{sim:.4f}" in explanation
