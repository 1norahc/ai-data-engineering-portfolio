"""
main.py — Customer Personality Segmentation Pipeline

Runs the full analysis: data loading, EDA, clustering, and reporting.

Usage
-----
    python main.py

Outputs
-------
    reports/customer_segments.csv   — customers with cluster assignments
    reports/figures/                — all generated plots
"""

import os
import sys

# ── make sure src/ is importable ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from src.preprocessing import run_preprocessing_pipeline
from src.eda import run_full_eda
from src.clustering import (
    compute_elbow_metrics,
    fit_kmeans,
    assign_clusters,
    get_cluster_profiles,
    get_cluster_summary,
)
from src.visualization import (
    plot_elbow_curve,
    plot_cluster_distribution,
    plot_cluster_income_spending,
    plot_cluster_profiles_heatmap,
    plot_cluster_spending_boxplots,
)

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
N_CLUSTERS = 5


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 60)
    print("  Customer Personality Segmentation Pipeline")
    print("=" * 60)

    # ── 1. Preprocessing ──────────────────────────────────────────────────────
    print("\n[1/4] Loading and preprocessing data …")
    df, X_scaled, scaler = run_preprocessing_pipeline()
    print(f"      Rows: {len(df):,}  |  Features: {X_scaled.shape[1]}")

    # ── 2. Exploratory Data Analysis ──────────────────────────────────────────
    print("\n[2/4] Running exploratory data analysis …")
    run_full_eda(df, save=True, path=FIGURES_DIR)
    print("      Plots saved to reports/figures/")

    # ── 3. Clustering ─────────────────────────────────────────────────────────
    print("\n[3/4] Selecting optimal k and training K-Means …")
    metrics = compute_elbow_metrics(X_scaled, max_k=10)
    plot_elbow_curve(
        metrics, save=True,
        path=os.path.join(FIGURES_DIR, "elbow_curve.png")
    )

    model = fit_kmeans(X_scaled, n_clusters=N_CLUSTERS)
    df = assign_clusters(df, model, X_scaled)
    print(f"      Inertia:           {model.inertia_:,.2f}")

    # ── 4. Visualisation & reporting ──────────────────────────────────────────
    print("\n[4/4] Generating cluster visualisations and saving results …")
    plot_cluster_distribution(
        df, save=True,
        path=os.path.join(FIGURES_DIR, "cluster_distribution.png")
    )
    plot_cluster_income_spending(
        df, save=True,
        path=os.path.join(FIGURES_DIR, "income_vs_spending.png")
    )
    plot_cluster_profiles_heatmap(
        df, save=True,
        path=os.path.join(FIGURES_DIR, "cluster_profiles.png")
    )
    plot_cluster_spending_boxplots(
        df, save=True,
        path=os.path.join(FIGURES_DIR, "cluster_spending_boxplots.png")
    )

    # Save segmented dataset
    out_csv = os.path.join(REPORTS_DIR, "customer_segments.csv")
    df.to_csv(out_csv, index=False)

    # Print summary table
    print("\n── Cluster Summary ──────────────────────────────────────")
    summary = get_cluster_summary(df)
    print(summary.to_string(index=False))

    print("\n── Cluster Profiles (top metrics) ───────────────────────")
    profile_cols = ["Income", "Age", "TotalSpending", "MntWines", "MntMeatProducts"]
    available = [c for c in profile_cols if c in df.columns]
    print(get_cluster_profiles(df)[available].to_string())

    print(f"\n Done!  Results saved to {REPORTS_DIR}/")


if __name__ == "__main__":
    main()
