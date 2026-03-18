"""
eda.py — Customer Personality Segmentation

Exploratory Data Analysis: distributions, correlations, spending patterns,
and campaign response rates. All functions accept a ``save`` flag and optional
``path`` to write figures to disk.
"""

import os
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np

matplotlib.use("Agg")  # non-interactive backend safe for scripts

FIGURES_PATH = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")

_MNT_COLS = [
    "MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts", "MntGoldProds",
]

_CAMPAIGN_COLS = [
    "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3",
    "AcceptedCmp4", "AcceptedCmp5", "Response",
]


def _save_or_show(fig: plt.Figure, save: bool, filename: str, path: str = None):
    if save:
        out_dir = path or FIGURES_PATH
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, filename), bbox_inches="tight", dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_missing_values(df: pd.DataFrame, save: bool = False, path: str = None):
    """
    Bar chart of missing value counts per column (only shows columns with nulls).
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("No missing values found.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    missing.sort_values().plot(kind="barh", color="#e74c3c", ax=ax)
    ax.set_title("Missing Values per Column", fontsize=14, fontweight="bold")
    ax.set_xlabel("Count")
    fig.tight_layout()
    _save_or_show(fig, save, "missing_values.png", path)


def plot_income_distribution(df: pd.DataFrame, save: bool = False, path: str = None):
    """
    Histogram + KDE of customer income distribution.
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(df["Income"], bins=40, kde=True, color="#3498db", ax=ax)
    ax.axvline(df["Income"].median(), color="#e74c3c", linestyle="--",
               label=f"Median: ${df['Income'].median():,.0f}")
    ax.set_title("Income Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Annual Income ($)")
    ax.legend()
    fig.tight_layout()
    _save_or_show(fig, save, "income_distribution.png", path)


def plot_spending_distributions(df: pd.DataFrame, save: bool = False, path: str = None):
    """
    Grid of histograms for each spending category (MntWines, MntFruits, …).
    """
    cols = [c for c in _MNT_COLS if c in df.columns]
    n = len(cols)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.flatten()
    colors = ["#e74c3c", "#2ecc71", "#e67e22", "#3498db", "#9b59b6", "#f1c40f"]

    for i, col in enumerate(cols):
        sns.histplot(df[col], bins=30, kde=True, color=colors[i], ax=axes[i])
        axes[i].set_title(col.replace("Mnt", "Spending: "), fontsize=11)
        axes[i].set_xlabel("Amount ($)")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Spending Category Distributions", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save_or_show(fig, save, "spending_distributions.png", path)


def plot_correlation_heatmap(df: pd.DataFrame, save: bool = False, path: str = None):
    """
    Seaborn heatmap of numeric feature correlations (annotated, masked upper triangle).
    """
    num_df = df.select_dtypes(include=np.number)
    corr = num_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        vmin=-1, vmax=1, linewidths=0.5, ax=ax
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_or_show(fig, save, "correlation_heatmap.png", path)


def plot_age_vs_income(df: pd.DataFrame, save: bool = False, path: str = None):
    """
    Scatter plot of Age vs Income, coloured by Education level.
    """
    if "Age" not in df.columns or "Income" not in df.columns:
        print("Age / Income columns not found — run engineer_features() first.")
        return

    hue_col = "Education" if "Education" in df.columns else None
    fig, ax = plt.subplots(figsize=(9, 5))
    scatter = ax.scatter(
        df["Age"], df["Income"],
        c=pd.factorize(df[hue_col])[0] if hue_col else "#3498db",
        alpha=0.4, s=20, cmap="tab10"
    )
    ax.set_xlabel("Age")
    ax.set_ylabel("Annual Income ($)")
    ax.set_title("Age vs Income" + (f" (coloured by {hue_col})" if hue_col else ""),
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_or_show(fig, save, "age_vs_income.png", path)


def plot_campaign_response_rates(df: pd.DataFrame, save: bool = False, path: str = None):
    """
    Bar chart of acceptance rates for each of the 5 campaigns + final response.
    """
    cols = [c for c in _CAMPAIGN_COLS if c in df.columns]
    rates = df[cols].mean().sort_values()

    fig, ax = plt.subplots(figsize=(8, 4))
    rates.plot(kind="barh", color="#9b59b6", ax=ax)
    ax.set_title("Campaign Acceptance Rates", fontsize=13, fontweight="bold")
    ax.set_xlabel("Acceptance Rate")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
    fig.tight_layout()
    _save_or_show(fig, save, "campaign_response_rates.png", path)


def run_full_eda(df: pd.DataFrame, save: bool = False, path: str = None):
    """
    Execute all EDA plots in sequence.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned and engineered DataFrame.
    save : bool
        If True, write PNG files to ``path`` (or default FIGURES_PATH).
    path : str, optional
        Directory for saved figures.
    """
    plot_missing_values(df, save=save, path=path)
    plot_income_distribution(df, save=save, path=path)
    plot_spending_distributions(df, save=save, path=path)
    plot_correlation_heatmap(df, save=save, path=path)
    plot_age_vs_income(df, save=save, path=path)
    plot_campaign_response_rates(df, save=save, path=path)
