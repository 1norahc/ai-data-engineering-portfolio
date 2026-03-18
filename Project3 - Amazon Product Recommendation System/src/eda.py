"""
eda.py — Amazon Product Recommendation System

Exploratory Data Analysis: rating distributions, user activity,
product popularity, and sparsity visualisation.
"""

import os
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np

matplotlib.use("Agg")

FIGURES_PATH = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")


def _save_or_show(fig: plt.Figure, save: bool, filename: str, path: str = None):
    if save:
        out_dir = path or FIGURES_PATH
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, filename), bbox_inches="tight", dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_rating_distribution(df: pd.DataFrame, title_suffix: str = "",
                              save: bool = False, path: str = None):
    """
    Bar chart of rating value counts (1–5 stars).

    Parameters
    ----------
    df : pd.DataFrame
        Ratings DataFrame.
    title_suffix : str
        Optional suffix for the plot title (e.g., "— Filtered Dataset").
    """
    counts = df["rating"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values, color="#3498db", edgecolor="white", width=0.6)
    ax.bar_label(bars, fmt="%d", padding=3)
    ax.set_title(f"Rating Distribution {title_suffix}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Rating (Stars)")
    ax.set_ylabel("Number of Ratings")
    ax.set_xticks([1, 2, 3, 4, 5])
    fig.tight_layout()
    _save_or_show(fig, save, "rating_distribution.png", path)


def plot_user_activity(df: pd.DataFrame, top_n: int = 20,
                       save: bool = False, path: str = None):
    """
    Bar chart of the top N most active users by number of ratings given.
    """
    top_users = df["userId"].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(10, 5))
    top_users.plot(kind="bar", color="#9b59b6", ax=ax)
    ax.set_title(f"Top {top_n} Most Active Users", fontsize=13, fontweight="bold")
    ax.set_xlabel("User ID")
    ax.set_ylabel("Number of Ratings")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    _save_or_show(fig, save, "user_activity.png", path)


def plot_product_popularity(df: pd.DataFrame, top_n: int = 20,
                             save: bool = False, path: str = None):
    """
    Bar chart of the top N most-rated products.
    """
    top_products = df["productId"].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(10, 5))
    top_products.plot(kind="bar", color="#e74c3c", ax=ax)
    ax.set_title(f"Top {top_n} Most Rated Products", fontsize=13, fontweight="bold")
    ax.set_xlabel("Product ID")
    ax.set_ylabel("Number of Ratings")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    _save_or_show(fig, save, "product_popularity.png", path)


def plot_ratings_per_user_distribution(df: pd.DataFrame,
                                        save: bool = False, path: str = None):
    """
    Log-scale histogram of the number of ratings per user.
    Shows the long-tail distribution typical in recommendation datasets.
    """
    user_counts = df["userId"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(user_counts, bins=50, color="#2ecc71", log=True, edgecolor="white")
    ax.set_title("Ratings per User Distribution (log scale)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Ratings")
    ax.set_ylabel("Number of Users (log scale)")
    fig.tight_layout()
    _save_or_show(fig, save, "ratings_per_user.png", path)


def plot_ratings_per_product_distribution(df: pd.DataFrame,
                                           save: bool = False, path: str = None):
    """
    Log-scale histogram of the number of ratings per product.
    """
    product_counts = df["productId"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(product_counts, bins=50, color="#f39c12", log=True, edgecolor="white")
    ax.set_title("Ratings per Product Distribution (log scale)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Ratings")
    ax.set_ylabel("Number of Products (log scale)")
    fig.tight_layout()
    _save_or_show(fig, save, "ratings_per_product.png", path)


def plot_before_after_filtering(df_raw: pd.DataFrame, df_filtered: pd.DataFrame,
                                  save: bool = False, path: str = None):
    """
    Side-by-side bar charts comparing dataset size before and after filtering.
    """
    labels = ["Users", "Products", "Interactions"]
    before = [
        df_raw["userId"].nunique(),
        df_raw["productId"].nunique(),
        len(df_raw),
    ]
    after = [
        df_filtered["userId"].nunique(),
        df_filtered["productId"].nunique(),
        len(df_filtered),
    ]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, before, width, label="Raw", color="#3498db", alpha=0.8)
    ax.bar(x + width / 2, after, width, label="Filtered", color="#e74c3c", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Dataset Size: Before vs After Filtering", fontsize=13, fontweight="bold")
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_yscale("log")
    fig.tight_layout()
    _save_or_show(fig, save, "before_after_filtering.png", path)


def run_full_eda(df_raw: pd.DataFrame, df_filtered: pd.DataFrame,
                 save: bool = False, path: str = None):
    """
    Run all EDA plots.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Unfiltered ratings.
    df_filtered : pd.DataFrame
        Filtered ratings (active users, popular products).
    save : bool
        Write PNG files if True.
    path : str, optional
        Output directory.
    """
    plot_rating_distribution(df_filtered, "— Filtered Dataset", save=save, path=path)
    plot_user_activity(df_filtered, save=save, path=path)
    plot_product_popularity(df_filtered, save=save, path=path)
    plot_ratings_per_user_distribution(df_filtered, save=save, path=path)
    plot_ratings_per_product_distribution(df_filtered, save=save, path=path)
    plot_before_after_filtering(df_raw, df_filtered, save=save, path=path)
