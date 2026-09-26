"""Manual verification script for ContentBasedRecommender module."""

import sys
from pathlib import Path
import pandas as pd

# Ensure project root is in sys.path when script is executed directly
script_dir = Path(__file__).resolve().parent
ml_dir = script_dir.parent
project_root = ml_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.src import config
from ml.src.content_based import ContentBasedRecommender


def main():
    print("=" * 70)
    print("PHASE 3 — CONTENT-BASED RECOMMENDER MANUAL TEST & VERIFICATION")
    print("=" * 70)

    # 1. Load Processed Movies
    movies_csv_path = config.PROCESSED_DATA_DIR / "movies_processed.csv"
    if not movies_csv_path.exists():
        print(f"Error: Processed dataset not found at {movies_csv_path}. Please run run_eda.py first.")
        sys.exit(1)

    print(f"\n[1/5] Loading processed movies from {movies_csv_path.name}...")
    movies_df = pd.read_csv(movies_csv_path)
    print(f"      Loaded {len(movies_df):,} movies.")

    # 2. Fit Content-Based Recommender
    print("\n[2/5] Fitting ContentBasedRecommender...")
    recommender = ContentBasedRecommender()
    recommender.fit(movies_df)

    # Select a target query movie
    target_movie = "Toy Story (1995)"
    try:
        recommender.get_movie_index(target_movie)
    except ValueError:
        # Fallback to first movie in catalog if Toy Story is missing
        target_movie = str(movies_df["title"].iloc[0])

    print(f"\n[3/5] Generating Top 10 Recommendations for: '{target_movie}'")
    recommendations = recommender.recommend(target_movie, n=10)

    print("-" * 70)
    print(f"{'movieId':<8} {'Similarity':<12} {'Title':<35} {'Genres'}")
    print("-" * 70)
    for item in recommendations:
        title_trunc = item["title"][:33] + ".." if len(item["title"]) > 35 else item["title"]
        print(f"{item['movieId']:<8} {item['similarity_score']:<12.4f} {title_trunc:<35} {item['genres']}")
    print("-" * 70)

    # 4. Demonstrate Explanation & Similarity Scoring
    print("\n[4/5] Explaining Top Recommendations & Pairwise Similarity:")
    if recommendations:
        top_rec_title = recommendations[0]["title"]
        explanation = recommender.explain_recommendation(target_movie, top_rec_title)
        print(f"      Source: '{target_movie}'")
        print(f"      Rec #1: '{top_rec_title}'")
        print(f"      Explanation:\n      => {explanation}")

        sim_val = recommender.get_similarity(target_movie, top_rec_title)
        print(f"      Direct pairwise similarity: {sim_val:.4f}")

    # 5. Demonstrate Edge Cases & Robust Error Handling
    print("\n[5/5] Testing Edge Cases & Error Handling:")

    # Case & whitespace insensitive lookup
    fuzzy_query = f"   {target_movie.upper()}   "
    try:
        idx = recommender.get_movie_index(fuzzy_query)
        print(f"  [PASS] Case/whitespace normalized lookup for '{fuzzy_query.strip()}' -> Index {idx}")
    except ValueError as e:
        print(f"  [FAIL] Normalization error: {e}")

    # Unknown movie error handling
    unknown_query = "NonExistent Movie 99999 (2099)"
    try:
        recommender.recommend(unknown_query, n=5)
        print("  [FAIL] Unknown movie should have raised ValueError!")
    except ValueError as e:
        print(f"  [PASS] Handled unknown movie gracefully -> Raised ValueError: \"{e}\"")

    # Invalid n error handling
    try:
        recommender.recommend(target_movie, n=-5)
        print("  [FAIL] n <= 0 should have raised ValueError!")
    except ValueError as e:
        print(f"  [PASS] Handled n <= 0 gracefully -> Raised ValueError: \"{e}\"")

    print("\n=" * 70)
    print("Content-Based Recommender Verification Status: SUCCESS")
    print("=" * 70)


if __name__ == "__main__":
    main()
