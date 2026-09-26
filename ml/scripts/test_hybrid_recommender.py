"""Manual verification script for HybridRecommender module."""

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
from ml.src.collaborative_filtering import ItemBasedCollaborativeRecommender
from ml.src.hybrid_recommender import HybridRecommender


def main():
    print("=" * 95)
    print("PHASE 5 — HYBRID RECOMMENDER MANUAL TEST & VERIFICATION")
    print("=" * 95)

    # 1. Load Processed Datasets
    movies_csv_path = config.PROCESSED_DATA_DIR / "movies_processed.csv"
    ratings_csv_path = config.PROCESSED_DATA_DIR / "ratings_processed.csv"

    if not movies_csv_path.exists() or not ratings_csv_path.exists():
        print("Error: Processed datasets missing. Please run run_eda.py first.")
        sys.exit(1)

    print("\n[1/7] Loading processed datasets...")
    movies_df = pd.read_csv(movies_csv_path)
    ratings_df = pd.read_csv(ratings_csv_path)
    print(f"      Loaded {len(movies_df):,} movies and {len(ratings_df):,} ratings.")

    # 2. Fit Component Models
    print("\n[2/7] Fitting Phase 3 Content-Based and Phase 4 Collaborative Recommenders...")
    content_rec = ContentBasedRecommender()
    content_rec.fit(movies_df)

    collab_rec = ItemBasedCollaborativeRecommender()
    collab_rec.fit(ratings_df, movies_df=movies_df)
    print("      Phase 3 & Phase 4 component models fitted successfully.")

    # 3. Initialize Hybrid Engine (Default 50/50 weights)
    print("\n[3/7] Initializing HybridRecommender (content_weight=0.5, collaborative_weight=0.5)...")
    hybrid = HybridRecommender(
        content_recommender=content_rec,
        collaborative_recommender=collab_rec,
        content_weight=0.5,
        collaborative_weight=0.5
    )

    # 4. Generate Recommendations for User 1
    target_user_id = 1
    user_rated_mids = collab_rec.user_rated_movies[target_user_id]
    print(f"\n[4/7] Generating Top 10 Hybrid Recommendations for User {target_user_id}:")
    recs = hybrid.recommend_for_user(user_id=target_user_id, n=10)

    print("-" * 95)
    print(f"{'movieId':<8} {'Title':<28} {'Raw Content':<13} {'Raw Collab':<13} {'Norm C':<8} {'Norm CF':<8} {'Hybrid Score'}")
    print("-" * 95)
    rec_mids = []
    for r in recs:
        rec_mids.append(r["movieId"])
        title_trunc = r["title"][:26] + ".." if len(r["title"]) > 28 else r["title"]
        c_raw = f"{r['content_score']:.4f}" if r['content_score'] is not None else "N/A"
        cf_raw = f"{r['collaborative_score']:.4f}" if r['collaborative_score'] is not None else "N/A"
        print(f"{r['movieId']:<8} {title_trunc:<28} {c_raw:<13} {cf_raw:<13} {r['normalized_content_score']:<8.4f} {r['normalized_collaborative_score']:<8.4f} {r['hybrid_score']:.4f}")
    print("-" * 95)

    # 5. Verify Already-Rated Filtering Invariant
    already_rated_overlap = set(rec_mids) & user_rated_mids
    print(f"      Already-rated movie overlap count: {len(already_rated_overlap)}")
    if len(already_rated_overlap) == 0:
        print("  [PASS] Mandatory Already-Rated Filtering invariant verified.")
    else:
        print("  [FAIL] Already-rated movies detected in recommendations!")
        sys.exit(1)

    # 6. Display Explanation
    print("\n[5/7] Demonstrating Hybrid Explanation:")
    if recs:
        top_mid = recs[0]["movieId"]
        exp = hybrid.explain_hybrid_recommendation(target_user_id, top_mid)
        print(f"      => {exp['explanation']}")

    # 7. Test Alternative Weight Configurations (0.2 Content / 0.8 Collaborative)
    print("\n[6/8] Testing Alternative Weight Configuration (0.2 Content / 0.8 Collaborative):")
    hybrid_custom = HybridRecommender(
        content_recommender=content_rec,
        collaborative_recommender=collab_rec,
        content_weight=0.2,
        collaborative_weight=0.8
    )
    recs_custom = hybrid_custom.recommend_for_user(user_id=target_user_id, n=5)
    print(f"      Top recommendation with 0.2/0.8 weights: '{recs_custom[0]['title']}' (Score: {recs_custom[0]['hybrid_score']:.4f})")
    print("  [PASS] Alternative weight configuration executed successfully.")

    # 8. True Two-Source Fusion Demonstration (Content Signal + Collaborative Signal)
    print("\n[7/8] Verifying True Two-Source Fusion (Content + Collaborative signals):")
    dual_recs = hybrid.recommend_for_user(
        user_id=target_user_id,
        n=20,
        candidate_pool_size=500,
        seed_movie_title="Toy Story (1995)"
    )
    dual_signal_candidates = [
        r for r in dual_recs if r["content_score"] is not None and r["collaborative_score"] is not None
    ]
    print(f"      Found {len(dual_signal_candidates)} candidates with BOTH Content and Collaborative signals.")
    if dual_signal_candidates:
        sample_cand = dual_signal_candidates[0]
        expected_score = round(0.5 * sample_cand["normalized_content_score"] + 0.5 * sample_cand["normalized_collaborative_score"], 4)
        print(f"      Sample Candidate: '{sample_cand['title']}' (movieId: {sample_cand['movieId']})")
        print(f"      - Raw Content: {sample_cand['content_score']:.4f}, Norm Content: {sample_cand['normalized_content_score']:.4f}")
        print(f"      - Raw Collab: {sample_cand['collaborative_score']:.4f}, Norm Collab: {sample_cand['normalized_collaborative_score']:.4f}")
        print(f"      - Calculated Hybrid Score: {sample_cand['hybrid_score']:.4f} (Expected: {expected_score:.4f})")
        assert sample_cand["hybrid_score"] == expected_score, "Hybrid score math discrepancy!"
        print("  [PASS] True Two-Source Fusion verified.")
    else:
        print("  [INFO] No dual-signal candidates found in top pool.")

    # 8. Cold-Start Error Handling Verification
    print("\n[8/8] Verifying Cold-Start Rules & Error Handling:")

    # Unknown user without seed_movie_title -> Must raise ValueError
    try:
        hybrid.recommend_for_user(user_id=999999, n=5, seed_movie_title=None)
        print("  [FAIL] Unknown user without seed title should have raised ValueError!")
    except ValueError as e:
        print(f"  [PASS] Handled unknown user without seed title -> Raised ValueError: \"{e}\"")

    # Unknown user WITH seed_movie_title -> Content fallback
    try:
        fallback_recs = hybrid.recommend_for_user(user_id=999999, n=5, seed_movie_title="Toy Story (1995)")
        print(f"  [PASS] Handled unknown user with seed title -> Fallback generated {len(fallback_recs)} content recommendations.")
    except ValueError as e:
        print(f"  [FAIL] Seed movie fallback failed: {e}")

    # Unknown seed movie title -> Must raise ValueError
    try:
        hybrid.recommend_for_user(user_id=target_user_id, n=5, seed_movie_title="NonExistent Movie 9999")
        print("  [FAIL] Unknown seed movie title should have raised ValueError!")
    except ValueError as e:
        print(f"  [PASS] Handled unknown seed movie title -> Raised ValueError: \"{e}\"")

    print("\n=" * 95)
    print("Hybrid Recommender Verification Status: SUCCESS")
    print("=" * 95)


if __name__ == "__main__":
    main()
