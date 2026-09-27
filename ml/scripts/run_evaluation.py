"""CLI script to execute Phase 6 Recommendation Model Evaluation."""

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
from ml.src.evaluation import ModelEvaluator


def main():
    print("=" * 105)
    print("PHASE 6 — RECOMMENDATION MODEL EVALUATION PIPELINE")
    print("=" * 105)

    movies_csv_path = config.PROCESSED_DATA_DIR / "movies_processed.csv"
    ratings_csv_path = config.PROCESSED_DATA_DIR / "ratings_processed.csv"

    if not movies_csv_path.exists() or not ratings_csv_path.exists():
        print(f"Error: Processed datasets missing at {config.PROCESSED_DATA_DIR}. Please run run_eda.py first.")
        sys.exit(1)

    print("\n[1/5] Loading processed MovieLens datasets...")
    movies_df = pd.read_csv(movies_csv_path)
    ratings_df = pd.read_csv(ratings_csv_path)
    print(f"      Loaded {len(movies_df):,} movies and {len(ratings_df):,} ratings.")

    # Validate required columns
    req_movie = set(config.REQUIRED_MOVIE_COLUMNS)
    req_rating = set(config.REQUIRED_RATING_COLUMNS)
    if not req_movie.issubset(movies_df.columns) or not req_rating.issubset(ratings_df.columns):
        print("Error: Missing required schema columns in processed CSV files.")
        sys.exit(1)

    # Initialize ModelEvaluator with standard configuration
    print("\n[2/5] Initializing ModelEvaluator configuration...")
    evaluator = ModelEvaluator(
        test_ratio=0.20,
        min_ratings=5,
        relevance_threshold=4.0,
        k_values=[5, 10, 20],
        candidate_pool_size=100,
        content_weight=0.5,
        collaborative_weight=0.5,
        seed=42
    )

    print(f"      - Test Ratio: {evaluator.test_ratio}")
    print(f"      - Min Ratings per User: {evaluator.min_ratings}")
    print(f"      - Relevance Threshold: >= {evaluator.relevance_threshold}")
    print(f"      - Cutoff Ranks (K): {evaluator.k_values}")
    print(f"      - Hybrid Component Weights: Content={evaluator.content_weight}, Collaborative={evaluator.collaborative_weight}")

    print("\n[3/5] Performing Temporal Train/Test Split & Fitting Models on Training Split ONLY...")
    results = evaluator.run_evaluation(movies_df=movies_df, ratings_df=ratings_df)

    meta = results["metadata"]
    models_summary = results["models"]

    print("\n[4/5] Data Split & User Eligibility Metadata:")
    print(f"      - Total Dataset Users: {meta['total_users']}")
    print(f"      - Eligible Test Users: {meta['eligible_users']}")
    print(f"      - Excluded Users: {meta['excluded_users']}")
    print(f"      - Training Interactions: {meta['train_interactions']:,}")
    print(f"      - Test Interactions: {meta['test_interactions']:,}")
    print(f"      - Users with 0 Relevant Test Items: {meta['users_with_no_relevant_test_items']}")

    print("\n[5/5] Evaluation Model Comparison Summary:")
    print("-" * 105)
    header = f"{'Model':<25} {'P@5':<8} {'R@5':<8} {'NDCG@5':<8} {'HR@5':<8} {'P@10':<8} {'R@10':<8} {'NDCG@10':<8} {'HR@10':<8} {'Cov@10':<8}"
    print(header)
    print("-" * 105)

    for m in models_summary:
        print(
            f"{m['Model']:<25} "
            f"{m['Precision@5']:<8.4f} {m['Recall@5']:<8.4f} {m['NDCG@5']:<8.4f} {m['Hit Rate@5']:<8.4f} "
            f"{m['Precision@10']:<8.4f} {m['Recall@10']:<8.4f} {m['NDCG@10']:<8.4f} {m['Hit Rate@10']:<8.4f} "
            f"{m['Coverage@10']:<8.4f}"
        )
    print("-" * 105)

    print(f"\nResults successfully exported to:")
    print(f"  - Metrics CSV:  {evaluator.metrics_dir / 'model_comparison.csv'}")
    print(f"  - Metrics JSON: {evaluator.metrics_dir / 'model_comparison.json'}")
    print(f"  - Per-User CSV: {evaluator.metrics_dir / 'per_user_metrics.csv'}")
    print(f"  - Plot Charts:  {evaluator.plots_dir}/")

    print("\n=" * 105)
    print("PHASE 6 RECOMMENDATION MODEL EVALUATION STATUS: SUCCESS")
    print("=" * 105)


if __name__ == "__main__":
    main()
