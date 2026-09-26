"""Manual verification script for ItemBasedCollaborativeRecommender module."""

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
from ml.src.collaborative_filtering import ItemBasedCollaborativeRecommender


def main():
    print("=" * 75)
    print("PHASE 4 — COLLABORATIVE FILTERING RECOMMENDER MANUAL TEST & VERIFICATION")
    print("=" * 75)

    # 1. Load Processed Datasets
    ratings_csv_path = config.PROCESSED_DATA_DIR / "ratings_processed.csv"
    movies_csv_path = config.PROCESSED_DATA_DIR / "movies_processed.csv"

    if not ratings_csv_path.exists() or not movies_csv_path.exists():
        print("Error: Processed datasets missing. Please run run_eda.py first.")
        sys.exit(1)

    print(f"\n[1/6] Loading processed dataset...")
    ratings_df = pd.read_csv(ratings_csv_path)
    movies_df = pd.read_csv(movies_csv_path)
    print(f"      Loaded {len(ratings_df):,} ratings and {len(movies_df):,} catalog movies.")

    # 2. Fit Collaborative Recommender
    print("\n[2/6] Fitting ItemBasedCollaborativeRecommender...")
    recommender = ItemBasedCollaborativeRecommender()
    recommender.fit(ratings_df, movies_df=movies_df)

    # Select target user (User 1)
    target_user_id = 1
    rated_movie_ids = recommender.user_rated_movies[target_user_id]
    print(f"      Target User: {target_user_id} (rated {len(rated_movie_ids)} movies in dataset)")

    # 3. Generate User Recommendations
    print(f"\n[3/6] Top 10 Personalized Collaborative Recommendations for User {target_user_id}:")
    user_recs = recommender.recommend_for_user(target_user_id, n=10)

    print("-" * 75)
    print(f"{'movieId':<8} {'Score':<10} {'Title':<35} {'Genres'}")
    print("-" * 75)
    rec_movie_ids = []
    for item in user_recs:
        rec_movie_ids.append(item["movieId"])
        title_trunc = item.get("title", "")[:33] + ".." if len(item.get("title", "")) > 35 else item.get("title", "")
        genres_val = item.get("genres", "")
        print(f"{item['movieId']:<8} {item['score']:<10.4f} {title_trunc:<35} {genres_val}")
    print("-" * 75)

    # Verify Already-Rated Movies Exclusion
    already_rated_overlap = set(rec_movie_ids) & rated_movie_ids
    print(f"      Already-rated movie overlap count: {len(already_rated_overlap)} (Must be 0)")
    assert len(already_rated_overlap) == 0, "EXCLUSION FAILURE: Already rated movies appeared in recommendations!"

    # 4. Demonstrate Collaborative Explanation
    print(f"\n[4/6] Collaborative Explanation for Recommendation #1:")
    if user_recs:
        top_rec_mid = user_recs[0]["movieId"]
        exp = recommender.explain_user_recommendation(target_user_id, top_rec_mid)
        print(f"      => {exp['explanation']}")

    # 5. Demonstrate Similar Movies via Item-Item Rating Vectors
    target_movie_id = 1  # Toy Story (1995)
    print(f"\n[5/6] Top 5 Collaboratively Similar Movies to Movie {target_movie_id} (Toy Story):")
    similar_movies = recommender.recommend_similar_movies(target_movie_id, n=5)

    print("-" * 75)
    print(f"{'movieId':<8} {'Similarity':<12} {'Title':<35} {'Genres'}")
    print("-" * 75)
    for item in similar_movies:
        title_trunc = item.get("title", "")[:33] + ".." if len(item.get("title", "")) > 35 else item.get("title", "")
        print(f"{item['movieId']:<8} {item['similarity_score']:<12.4f} {title_trunc:<35} {item.get('genres', '')}")
    print("-" * 75)

    # 6. Test Error Handling & Edge Cases
    print("\n[6/6] Testing Cold-Start & Unknown ID Error Handling:")

    # Unknown User ID
    try:
        recommender.recommend_for_user(999999, n=5)
        print("  [FAIL] Unknown user ID should have raised ValueError!")
    except ValueError as e:
        print(f"  [PASS] Handled unknown user gracefully -> Raised ValueError: \"{e}\"")

    # Unknown Movie ID
    try:
        recommender.recommend_similar_movies(999999, n=5)
        print("  [FAIL] Unknown movie ID should have raised ValueError!")
    except ValueError as e:
        print(f"  [PASS] Handled unknown movie gracefully -> Raised ValueError: \"{e}\"")

    # Invalid n <= 0
    try:
        recommender.recommend_for_user(target_user_id, n=0)
        print("  [FAIL] n <= 0 should have raised ValueError!")
    except ValueError as e:
        print(f"  [PASS] Handled n <= 0 gracefully -> Raised ValueError: \"{e}\"")

    print("\n=" * 75)
    print("Collaborative Filtering Recommender Verification Status: SUCCESS")
    print("=" * 75)


if __name__ == "__main__":
    main()
