"""
main.py — Amazon Product Recommendation System Pipeline

Loads and filters the Amazon Electronics ratings dataset, trains four
recommendation models, evaluates them, and prints a comparison table.

Usage
-----
    python main.py                     # full run (may load up to 500k rows)
    python main.py --nrows 100000      # faster dev run

Outputs
-------
    reports/figures/           — all EDA and comparison plots
    reports/model_results.csv  — model evaluation metrics
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from src.preprocessing import run_preprocessing_pipeline
from src.eda import run_full_eda
from src.recommenders import (
    get_rank_based_recommendations,
    train_user_user_cf,
    train_item_item_cf,
    train_svd,
    get_recommendations_for_user,
)
from src.evaluation import evaluate_model, plot_model_comparison

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

# NOTE: Full dataset has 7.8M rows. We default to 500k for a reasonable runtime.
DEFAULT_NROWS = 500_000


def parse_args():
    parser = argparse.ArgumentParser(description="Amazon Recommendation System")
    parser.add_argument("--nrows", type=int, default=DEFAULT_NROWS,
                        help=f"Rows to load from CSV (default {DEFAULT_NROWS:,}). "
                             "Set to 0 for full dataset.")
    parser.add_argument("--no-tune", action="store_true",
                        help="Skip hyperparameter tuning (faster run).")
    return parser.parse_args()


def main():
    args = parse_args()
    nrows = args.nrows if args.nrows > 0 else None

    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 65)
    print("  Amazon Electronics Recommendation System Pipeline")
    print("=" * 65)
    if nrows:
        print(f"\n  NOTE: Loading first {nrows:,} rows for speed.")
        print(f"  Set --nrows 0 to load the full 7.8M row dataset.\n")

    # ── 1. Preprocessing ──────────────────────────────────────────────────────
    print("[1/5] Loading and filtering data …")
    pipeline_data = run_preprocessing_pipeline(nrows=nrows)
    df_raw = pipeline_data["df_raw"]
    df = pipeline_data["df_filtered"]
    trainset = pipeline_data["trainset"]
    testset = pipeline_data["testset"]

    raw_stats = pipeline_data["stats_raw"]
    filt_stats = pipeline_data["stats_filtered"]

    print(f"      Raw dataset:      {raw_stats['n_interactions']:>10,} interactions  "
          f"| {raw_stats['n_users']:>7,} users | {raw_stats['n_products']:>7,} products")
    print(f"      Filtered dataset: {filt_stats['n_interactions']:>10,} interactions  "
          f"| {filt_stats['n_users']:>7,} users | {filt_stats['n_products']:>7,} products")
    print(f"      Sparsity: {filt_stats['sparsity']:.2f}%  |  Avg rating: {filt_stats['avg_rating']:.3f}")

    # ── 2. EDA ────────────────────────────────────────────────────────────────
    print("\n[2/5] Running exploratory data analysis …")
    run_full_eda(df_raw, df, save=True, path=FIGURES_DIR)
    print("      Plots saved to reports/figures/")

    # ── 3. Rank-Based Baseline ────────────────────────────────────────────────
    print("\n[3/5] Computing rank-based recommendations (baseline) …")
    top_products = get_rank_based_recommendations(df, min_interactions=50, top_n=10)
    print("      Top 10 popular products:")
    print(top_products[["rank", "productId", "avg_rating", "num_ratings"]].to_string(index=False))

    # ── 4. Train CF Models ────────────────────────────────────────────────────
    print("\n[4/5] Training collaborative filtering models …")

    print("      Training User-User CF …")
    uu_model = train_user_user_cf(trainset)
    uu_results = evaluate_model(uu_model, testset, "User-User CF")

    print("      Training Item-Item CF …")
    ii_model = train_item_item_cf(trainset)
    ii_results = evaluate_model(ii_model, testset, "Item-Item CF")

    print("      Training SVD …")
    svd_model = train_svd(trainset)
    svd_results = evaluate_model(svd_model, testset, "SVD")

    # ── 5. Comparison ─────────────────────────────────────────────────────────
    print("\n[5/5] Comparing models and saving results …")
    results_list = [uu_results, ii_results, svd_results]
    cmp_df = plot_model_comparison(results_list, save=True, path=FIGURES_DIR)
    cmp_df.to_csv(os.path.join(REPORTS_DIR, "model_results.csv"))

    print("\n── Model Comparison ───────────────────────────────────────")
    display = cmp_df.drop(columns=["predictions"], errors="ignore")
    print(cmp_df.to_string())

    # ── Sample Recommendations ────────────────────────────────────────────────
    sample_user = df["userId"].value_counts().index[0]  # most active user
    print(f"\n── Top-5 Recommendations for user '{sample_user}' (SVD) ──")
    recs = get_recommendations_for_user(svd_model, sample_user, df, n=5)
    if not recs.empty:
        print(recs.to_string(index=False))
    else:
        print("      (No recommendations above threshold for this user)")

    print(f"\n Done!  Results saved to {REPORTS_DIR}/")


if __name__ == "__main__":
    main()
