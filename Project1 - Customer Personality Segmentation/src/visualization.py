"""
visualization.py — Customer Personality Segmentation

Post-clustering visualizations: elbow curves, cluster distributions,
scatter plots, profile heatmaps, and spending boxplots.
"""

import os
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np

matplotlib.use("Agg")

FIGURES_PATH = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")

CLUSTER_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

_SPENDING_COLS = [
    "MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts", "MntGoldProds",
]


def _save_or_show(fig: plt.Figure, save: bool, filename: str, path: str = None):
    if save:
        out_dir = path or os.path.dirname(
            os.path.join(FIGURES_PATH, filename)
        ) if os.path.dirname(filename) else FIGURES_PATH
        # If path is a full file path, use its directory
        if path and os.path.splitext(path)[1]:
            out_dir = os.path.dirname(path)
            filename = os.path.basename(path)
        else:
            out_dir = path or FIGURES_PATH

        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, filename), bbox_inches="tight", dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_elbow_curve(metrics: dict, save: bool = False, path: str = None):
    """
    Side-by-side line plots of inertia and silhouette score vs number of clusters.

    Parameters
    ----------
    metrics : dict
        Output of ``clustering.compute_elbow_metrics``.
    save : bool
        Write to disk if True.
    path : str, optional
        Output file path (full path) or directory.
    """
    k = metrics["k_values"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(k, metrics["inertias"], "o-", color="#3498db", linewidth=2)
    ax1.set_title("Elbow Method (Inertia)", fontweight="bold")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia (WCSS)")
    ax1.grid(alpha=0.3)

    ax2.plot(k, metrics["silhouette_scores"], "s-", color="#e74c3c", linewidth=2)
    ax2.set_title("Silhouette Score vs k", fontweight="bold")
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.grid(alpha=0.3)

    fig.suptitle("Optimal Cluster Selection", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_or_show(fig, save, "elbow_curve.png", path)


def plot_cluster_distribution(df: pd.DataFrame, save: bool = False, path: str = None):
    """
    Horizontal bar chart showing customer count per cluster.
    """
    counts = df.groupby(["Cluster", "Cluster_Name"]).size().reset_index(name="Count")
    counts = counts.sort_values("Cluster")

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(
        counts["Cluster_Name"], counts["Count"],
        color=[CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in counts["Cluster"]]
    )
    ax.bar_label(bars, padding=4)
    ax.set_title("Customer Count per Cluster", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Customers")
    fig.tight_layout()
    _save_or_show(fig, save, "cluster_distribution.png", path)


def plot_cluster_income_spending(df: pd.DataFrame, save: bool = False, path: str = None):
    """
    Scatter plot of Income vs TotalSpending coloured by cluster.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    for cluster_id in sorted(df["Cluster"].unique()):
        subset = df[df["Cluster"] == cluster_id]
        name = subset["Cluster_Name"].iloc[0]
        ax.scatter(
            subset["Income"], subset["TotalSpending"],
            label=f"{cluster_id}: {name}",
            color=CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)],
            alpha=0.5, s=25
        )

    ax.set_xlabel("Annual Income ($)")
    ax.set_ylabel("Total 2-Year Spending ($)")
    ax.set_title("Income vs Total Spending by Cluster", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    _save_or_show(fig, save, "income_vs_spending.png", path)


def plot_cluster_profiles_heatmap(df: pd.DataFrame, save: bool = False, path: str = None):
    """
    Heatmap of z-score normalised cluster profile means.
    Rows = clusters, columns = key metrics.
    """
    profile_cols = [
        c for c in [
            "Income", "Age", "TotalSpending", "TotalPurchases",
            "TotalCampaigns", "NumWebPurchases", "MntWines", "MntMeatProducts",
        ] if c in df.columns
    ]
    profile = df.groupby("Cluster_Name")[profile_cols].mean()
    profile_norm = (profile - profile.mean()) / (profile.std() + 1e-9)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(
        profile_norm, annot=True, fmt=".2f", cmap="RdYlGn",
        linewidths=0.5, ax=ax, cbar_kws={"label": "Z-Score"}
    )
    ax.set_title("Cluster Profile Heatmap (Z-Score Normalised)", fontsize=13, fontweight="bold")
    ax.set_ylabel("")
    fig.tight_layout()
    _save_or_show(fig, save, "cluster_profiles.png", path)


def plot_cluster_spending_boxplots(df: pd.DataFrame, save: bool = False, path: str = None):
    """
    Box plots comparing spending categories across clusters.
    """
    spending_cols = [c for c in _SPENDING_COLS if c in df.columns]
    n = len(spending_cols)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, col in enumerate(spending_cols):
        df.boxplot(column=col, by="Cluster_Name", ax=axes[i],
                   patch_artist=True, notch=False)
        axes[i].set_title(col.replace("Mnt", ""), fontsize=10)
        axes[i].set_xlabel("")
        axes[i].tick_params(axis="x", rotation=30)
        plt.sca(axes[i])
        plt.title(col.replace("Mnt", "Spending: "))

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Spending Distribution by Cluster", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, save, "cluster_spending_boxplots.png", path)
