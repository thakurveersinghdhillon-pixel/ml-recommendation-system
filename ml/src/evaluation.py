"""Recommendation System Model Evaluator Module for Phase 6."""

import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml.src import config
from ml.src.content_based import ContentBasedRecommender
from ml.src.collaborative_filtering import ItemBasedCollaborativeRecommender
from ml.src.hybrid_recommender import HybridRecommender
from ml.src.evaluation_utils import (
    temporal_train_test_split,
    get_user_training_content_seed,
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    hit_rate_at_k,
    catalog_coverage
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Rigorous, leakage-free evaluation pipeline for recommendation models."""

    def __init__(
        self,
        test_ratio: float = 0.20,
        min_ratings: int = 5,
        relevance_threshold: float = 4.0,
        k_values: list[int] | None = None,
        candidate_pool_size: int = 100,
        content_weight: float = 0.5,
        collaborative_weight: float = 0.5,
        seed: int = 42,
        output_dir: Path | None = None
    ):
        """Initialize ModelEvaluator with evaluation parameters.

        Args:
            test_ratio: Proportion of recent interactions allocated to test set.
            min_ratings: Minimum total interactions required per user for evaluation eligibility.
            relevance_threshold: Minimum rating threshold for ground-truth test relevance.
            k_values: List of top K recommendation cutoff ranks (defaults to [5, 10, 20]).
            candidate_pool_size: Candidate retrieval pool size for hybrid recommender.
            content_weight: Weight for content component in hybrid model.
            collaborative_weight: Weight for collaborative component in hybrid model.
            seed: Random seed for reproducibility.
            output_dir: Directory to store generated metric CSV/JSON and plots.
        """
        self.test_ratio = test_ratio
        self.min_ratings = min_ratings
        self.relevance_threshold = relevance_threshold
        self.k_values = sorted(k_values) if k_values else [5, 10, 20]
        self.candidate_pool_size = candidate_pool_size
        self.content_weight = content_weight
        self.collaborative_weight = collaborative_weight
        self.seed = seed

        if output_dir is None:
            self.output_dir = config.ML_DIR / "data" / "evaluation"
        else:
            self.output_dir = Path(output_dir)

        self.metrics_dir = self.output_dir / "metrics"
        self.plots_dir = self.output_dir / "plots"

        # Ensure output directories exist
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def run_evaluation(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame) -> dict:
        """Execute leakage-free temporal evaluation across Content, Collaborative, and Hybrid models.

        Args:
            movies_df: Movie catalog DataFrame ['movieId', 'title', 'genres'].
            ratings_df: User ratings DataFrame ['userId', 'movieId', 'rating', 'timestamp'].

        Returns:
            Dictionary containing evaluation summary results and metadata.
        """
        logger.info("Starting Phase 6 Recommendation Model Evaluation...")

        # 1. Temporal Train/Test Split (Strictly before model fitting)
        train_df, test_df, split_info = temporal_train_test_split(
            ratings_df=ratings_df,
            test_ratio=self.test_ratio,
            min_ratings=self.min_ratings,
            seed=self.seed
        )

        # 2. Fit Models Exclusively on Training Data
        logger.info("Fitting Content-Based Recommender on movie catalog...")
        content_rec = ContentBasedRecommender()
        content_rec.fit(movies_df)

        logger.info("Fitting Item-Based Collaborative Recommender on train_df ONLY...")
        collab_rec = ItemBasedCollaborativeRecommender()
        collab_rec.fit(train_df, movies_df=movies_df)

        logger.info("Initializing Hybrid Recommender with fitted component models...")
        hybrid_rec = HybridRecommender(
            content_recommender=content_rec,
            collaborative_recommender=collab_rec,
            content_weight=self.content_weight,
            collaborative_weight=self.collaborative_weight
        )

        # 3. Identify Eligible Test Users
        test_user_ids = sorted(test_df["userId"].unique().tolist())
        catalog_movie_ids = set(movies_df["movieId"].unique())

        max_k = max(self.k_values)
        model_names = ["Content-Based", "Collaborative Filtering", "Hybrid"]

        # Per-user metric storage
        per_user_records = []
        # Model -> K -> List of recommended movie IDs across all users for coverage calculation
        model_k_recommendations: dict[str, dict[int, list[int]]] = {
            m: {k: [] for k in self.k_values} for m in model_names
        }

        # User relevance tracking
        users_evaluated_count = len(test_user_ids)
        users_no_relevant_items_count = 0

        logger.info("Evaluating %d test users across K values %s...", users_evaluated_count, self.k_values)

        for uid in test_user_ids:
            # User training ratings & rated movies (for already-rated filtering)
            user_train_ratings = train_df[train_df["userId"] == uid]
            train_rated_mids = set(user_train_ratings["movieId"].astype(int))

            # User test ratings
            user_test_ratings = test_df[test_df["userId"] == uid]
            relevant_test_mids = set(
                user_test_ratings[user_test_ratings["rating"] >= self.relevance_threshold]["movieId"].astype(int)
            )

            has_relevant = len(relevant_test_mids) > 0
            if not has_relevant:
                users_no_relevant_items_count += 1

            # Select content seed from user's TRAINING history ONLY
            seed_title = get_user_training_content_seed(uid, user_train_ratings, movies_df)

            # Generate Recommendations for User from each Model
            # A. Collaborative Filtering Recommendations
            try:
                collab_recs_raw = collab_rec.recommend_for_user(uid, n=max_k)
                collab_rec_mids = [r["movieId"] for r in collab_recs_raw]
            except ValueError:
                collab_rec_mids = []

            # B. Content-Based Recommendations
            content_rec_mids = []
            if seed_title is not None:
                try:
                    # Fetch extra candidates to account for filtering already-rated training movies
                    raw_c_recs = content_rec.recommend(seed_title, n=max_k + len(train_rated_mids))
                    for r in raw_c_recs:
                        mid = r["movieId"]
                        if mid not in train_rated_mids:
                            content_rec_mids.append(mid)
                            if len(content_rec_mids) == max_k:
                                break
                except ValueError:
                    content_rec_mids = []

            # C. Hybrid Recommendations
            try:
                hybrid_recs_raw = hybrid_rec.recommend_for_user(
                    user_id=uid,
                    n=max_k,
                    candidate_pool_size=self.candidate_pool_size,
                    seed_movie_title=seed_title
                )
                hybrid_rec_mids = [r["movieId"] for r in hybrid_recs_raw]
            except ValueError:
                hybrid_rec_mids = []

            model_rec_map = {
                "Content-Based": content_rec_mids,
                "Collaborative Filtering": collab_rec_mids,
                "Hybrid": hybrid_rec_mids
            }

            # Collect recommendations for catalog coverage and calculate per-user metrics
            for m_name, rec_mids in model_rec_map.items():
                for k in self.k_values:
                    k_mids = rec_mids[:k]
                    model_k_recommendations[m_name][k].extend(k_mids)

                    p = precision_at_k(rec_mids, relevant_test_mids, k)
                    r = recall_at_k(rec_mids, relevant_test_mids, k)
                    nd = ndcg_at_k(rec_mids, relevant_test_mids, k)
                    hr = hit_rate_at_k(rec_mids, relevant_test_mids, k)

                    per_user_records.append({
                        "userId": uid,
                        "model": m_name,
                        "k": k,
                        "has_relevant_items": has_relevant,
                        "relevant_count": len(relevant_test_mids),
                        "precision": p,
                        "recall": r,
                        "ndcg": nd,
                        "hit_rate": hr
                    })

        per_user_df = pd.DataFrame(per_user_records)

        # 4. Aggregate Macro Metrics Across Users
        summary_rows = []
        for m_name in model_names:
            row_dict = {"Model": m_name}
            for k in self.k_values:
                sub_df = per_user_df[(per_user_df["model"] == m_name) & (per_user_df["k"] == k)]

                # Precision denominator: all evaluated test users
                mean_p = sub_df["precision"].mean() if not sub_df.empty else 0.0

                # Recall, NDCG, Hit Rate denominator: users with at least 1 relevant test item
                relevant_sub_df = sub_df[sub_df["has_relevant_items"]]
                mean_r = relevant_sub_df["recall"].mean() if not relevant_sub_df.empty else 0.0
                mean_nd = relevant_sub_df["ndcg"].mean() if not relevant_sub_df.empty else 0.0
                mean_hr = relevant_sub_df["hit_rate"].mean() if not relevant_sub_df.empty else 0.0

                # Catalog Coverage calculation
                cov = catalog_coverage(model_k_recommendations[m_name][k], catalog_movie_ids)

                row_dict[f"Precision@{k}"] = round(float(mean_p), 4)
                row_dict[f"Recall@{k}"] = round(float(mean_r), 4)
                row_dict[f"NDCG@{k}"] = round(float(mean_nd), 4)
                row_dict[f"Hit Rate@{k}"] = round(float(mean_hr), 4)
                row_dict[f"Coverage@{k}"] = round(float(cov), 4)

            summary_rows.append(row_dict)

        summary_df = pd.DataFrame(summary_rows)

        # 5. Export Results
        # Export per-user metrics
        per_user_path = self.metrics_dir / "per_user_metrics.csv"
        per_user_df.to_csv(per_user_path, index=False)

        # Export comparison CSV
        comparison_csv_path = self.metrics_dir / "model_comparison.csv"
        summary_df.to_csv(comparison_csv_path, index=False)

        # Build comprehensive JSON payload
        full_results = {
            "metadata": {
                "test_ratio": self.test_ratio,
                "min_ratings": self.min_ratings,
                "relevance_threshold": self.relevance_threshold,
                "k_values": self.k_values,
                "candidate_pool_size": self.candidate_pool_size,
                "hybrid_weights": {
                    "content_weight": self.content_weight,
                    "collaborative_weight": self.collaborative_weight
                },
                "seed": self.seed,
                "total_users": split_info["total_users"],
                "eligible_users": split_info["eligible_users"],
                "excluded_users": split_info["excluded_users"],
                "users_evaluated": users_evaluated_count,
                "users_with_no_relevant_test_items": users_no_relevant_items_count,
                "catalog_size": len(catalog_movie_ids),
                "train_interactions": split_info["train_interactions"],
                "test_interactions": split_info["test_interactions"]
            },
            "models": summary_df.to_dict(orient="records")
        }

        comparison_json_path = self.metrics_dir / "model_comparison.json"
        with open(comparison_json_path, "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=2)

        # 6. Generate Comparison Visualizations
        self._generate_plots(summary_df)

        logger.info("Phase 6 Model Evaluation complete. Results saved to %s.", self.output_dir)

        return full_results

    def _generate_plots(self, summary_df: pd.DataFrame) -> None:
        """Generate Matplotlib plots for Precision, Recall, NDCG, Hit Rate, and Coverage."""
        models = summary_df["Model"].tolist()
        x = np.arange(len(self.k_values))
        width = 0.25

        color_map = {
            "Content-Based": "#1f77b4",        # Blue
            "Collaborative Filtering": "#ff7f0e", # Orange
            "Hybrid": "#2ca02c"                # Green
        }

        # Plot 1: Precision@K
        plt.figure(figsize=(8, 5))
        for idx, m in enumerate(models):
            vals = [summary_df.loc[summary_df["Model"] == m, f"Precision@{k}"].values[0] for k in self.k_values]
            plt.bar(x + (idx - 1) * width, vals, width, label=m, color=color_map.get(m, "#333333"))
        plt.xlabel("Cutoff Rank (K)")
        plt.ylabel("Precision@K")
        plt.title(f"Precision@K Comparison (Relevance Threshold >= {self.relevance_threshold})")
        plt.xticks(x, [f"K={k}" for k in self.k_values])
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "precision_at_k.png", dpi=300)
        plt.close()

        # Plot 2: Recall@K
        plt.figure(figsize=(8, 5))
        for idx, m in enumerate(models):
            vals = [summary_df.loc[summary_df["Model"] == m, f"Recall@{k}"].values[0] for k in self.k_values]
            plt.bar(x + (idx - 1) * width, vals, width, label=m, color=color_map.get(m, "#333333"))
        plt.xlabel("Cutoff Rank (K)")
        plt.ylabel("Recall@K")
        plt.title(f"Recall@K Comparison (Relevance Threshold >= {self.relevance_threshold})")
        plt.xticks(x, [f"K={k}" for k in self.k_values])
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "recall_at_k.png", dpi=300)
        plt.close()

        # Plot 3: NDCG@K
        plt.figure(figsize=(8, 5))
        for idx, m in enumerate(models):
            vals = [summary_df.loc[summary_df["Model"] == m, f"NDCG@{k}"].values[0] for k in self.k_values]
            plt.bar(x + (idx - 1) * width, vals, width, label=m, color=color_map.get(m, "#333333"))
        plt.xlabel("Cutoff Rank (K)")
        plt.ylabel("NDCG@K")
        plt.title(f"NDCG@K Comparison (Relevance Threshold >= {self.relevance_threshold})")
        plt.xticks(x, [f"K={k}" for k in self.k_values])
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "ndcg_at_k.png", dpi=300)
        plt.close()

        # Plot 4: Hit Rate@K
        plt.figure(figsize=(8, 5))
        for idx, m in enumerate(models):
            vals = [summary_df.loc[summary_df["Model"] == m, f"Hit Rate@{k}"].values[0] for k in self.k_values]
            plt.bar(x + (idx - 1) * width, vals, width, label=m, color=color_map.get(m, "#333333"))
        plt.xlabel("Cutoff Rank (K)")
        plt.ylabel("Hit Rate@K")
        plt.title(f"Hit Rate@K Comparison (Relevance Threshold >= {self.relevance_threshold})")
        plt.xticks(x, [f"K={k}" for k in self.k_values])
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "hit_rate_at_k.png", dpi=300)
        plt.close()

        # Plot 5: Catalog Coverage Comparison
        plt.figure(figsize=(8, 5))
        for idx, m in enumerate(models):
            vals = [summary_df.loc[summary_df["Model"] == m, f"Coverage@{k}"].values[0] for k in self.k_values]
            plt.bar(x + (idx - 1) * width, vals, width, label=m, color=color_map.get(m, "#333333"))
        plt.xlabel("Cutoff Rank (K)")
        plt.ylabel("Catalog Coverage Proportion")
        plt.title("Catalog Coverage Comparison Across Models")
        plt.xticks(x, [f"K={k}" for k in self.k_values])
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "catalog_coverage.png", dpi=300)
        plt.close()
