"""
eda.py — Lead Conversion Prediction (ExtraaLearn)

Exploratory Data Analysis: class balance, numeric distributions by conversion
status, categorical conversion rates, and correlation heatmap.
"""

import os
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np

matplotlib.use("Agg")

FIGURES_PATH = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")

_NUMERIC_COLS = ["age", "website_visits", "time_spent_on_website", "page_views_per_visit"]
_CATEGORICAL_COLS = [
    "current_occupation", "first_interaction", "last_activity", "profile_completed"
]
TARGET = "status"


def _save_or_show(fig: plt.Figure, save: bool, filename: str, path: str = None):
    if save:
        out_dir = path or FIGURES_PATH
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, filename), bbox_inches="tight", dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_class_distribution(df: pd.DataFrame, save: bool = False, path: str = None):
    """
    Side-by-side pie and bar chart showing converted vs not-converted lead counts.
    """
    counts = df[TARGET].value_counts()
    labels = ["Not Converted", "Converted"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.pie(counts, labels=labels, autopct="%1.1f%%",
            colors=["#3498db", "#e74c3c"], startangle=90)
    ax1.set_title("Lead Conversion Split", fontweight="bold")

    ax2.bar(labels, counts.values, color=["#3498db", "#e74c3c"])
    ax2.set_title("Conversion Counts", fontweight="bold")
    ax2.set_ylabel("Number of Leads")
    for i, v in enumerate(counts.values):
        ax2.text(i, v + 20, str(v), ha="center", fontweight="bold")

    fig.tight_layout()
    _save_or_show(fig, save, "class_distribution.png", path)


def plot_numeric_distributions(df: pd.DataFrame, save: bool = False, path: str = None):
    """
    KDE plots for numeric features, split by conversion status.
    """
    cols = [c for c in _NUMERIC_COLS if c in df.columns]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    colors = {"not_converted": "#3498db", "converted": "#e74c3c"}

    for i, col in enumerate(cols):
        for label, color in zip([0, 1], ["#3498db", "#e74c3c"]):
            subset = df[df[TARGET] == label][col]
            sns.kdeplot(subset, ax=axes[i], fill=True, alpha=0.3,
                        color=color, label="Converted" if label else "Not Converted")
        axes[i].set_title(col.replace("_", " ").title(), fontsize=11, fontweight="bold")
        axes[i].legend()

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Numeric Feature Distributions by Conversion Status",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_or_show(fig, save, "numeric_distributions.png", path)


def plot_categorical_conversion_rates(df: pd.DataFrame, save: bool = False, path: str = None):
    """
    Grouped bar charts showing conversion rate (%) for each categorical feature.
    """
    cols = [c for c in _CATEGORICAL_COLS if c in df.columns]
    n = len(cols)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        rate = df.groupby(col)[TARGET].mean().sort_values(ascending=True)
        rate.plot(kind="barh", color="#9b59b6", ax=axes[i])
        axes[i].set_title(f"Conversion Rate by {col.replace('_', ' ').title()}",
                          fontsize=10, fontweight="bold")
        axes[i].set_xlabel("Conversion Rate")
        axes[i].xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.0%}")
        )

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Conversion Rate by Category", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_or_show(fig, save, "categorical_conversion_rates.png", path)


def plot_correlation_heatmap(df: pd.DataFrame, save: bool = False, path: str = None):
    """
    Correlation heatmap of numeric features including the target variable.
    """
    num_df = df.select_dtypes(include=np.number)
    corr = num_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5, ax=ax
    )
    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_or_show(fig, save, "correlation_heatmap.png", path)


def plot_time_on_site_vs_conversion(df: pd.DataFrame, save: bool = False, path: str = None):
    """
    Box plot of time spent on website for converted vs not-converted leads.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    df.boxplot(column="time_spent_on_website", by=TARGET, ax=ax,
               patch_artist=True, notch=False)
    ax.set_title("Time on Website by Conversion Status", fontweight="bold")
    ax.set_xlabel("Conversion Status (0=No, 1=Yes)")
    ax.set_ylabel("Time Spent on Website (s)")
    plt.suptitle("")  # suppress auto-title
    fig.tight_layout()
    _save_or_show(fig, save, "time_on_site_vs_conversion.png", path)


def run_full_eda(df: pd.DataFrame, save: bool = False, path: str = None):
    """
    Run all EDA plots in sequence.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame.
    save : bool
        Write PNG files to disk if True.
    path : str, optional
        Output directory for figures.
    """
    plot_class_distribution(df, save=save, path=path)
    plot_numeric_distributions(df, save=save, path=path)
    plot_categorical_conversion_rates(df, save=save, path=path)
    plot_correlation_heatmap(df, save=save, path=path)
    plot_time_on_site_vs_conversion(df, save=save, path=path)
