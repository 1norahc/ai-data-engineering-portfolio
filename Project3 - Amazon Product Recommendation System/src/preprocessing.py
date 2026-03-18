"""
preprocessing.py — Amazon Product Recommendation System

Loads the large Amazon Electronics ratings dataset, filters for active users
and popular products, and prepares data for the Surprise recommendation library.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "ratings_Electronics.csv"


def load_data(path: Path = DATA_PATH, nrows: int = None) -> pd.DataFrame:
    """
    Load the Amazon Electronics ratings dataset.

    The raw CSV has no header row. Columns are assigned as:
    userId, productId, rating, timestamp.

    Parameters
    ----------
    path : Path
        Path to the CSV file.
    nrows : int, optional
        Limit rows loaded (useful for development). Default: load all.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: userId, productId, rating, timestamp.
    """
    df = pd.read_csv(
        path,
        header=None,
        names=["userId", "productId", "rating", "timestamp"],
        nrows=nrows,
    )
    df["rating"] = df["rating"].astype(float)
    return df


def filter_active_users(df: pd.DataFrame, min_ratings: int = 50) -> pd.DataFrame:
    """
    Remove users with fewer than ``min_ratings`` interactions.

    Addresses data sparsity: keeping only active users improves collaborative
    filtering quality by ensuring enough signal per user.

    Parameters
    ----------
    df : pd.DataFrame
        Raw ratings DataFrame.
    min_ratings : int
        Minimum number of ratings a user must have to be retained.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    user_counts = df["userId"].value_counts()
    active_users = user_counts[user_counts >= min_ratings].index
    return df[df["userId"].isin(active_users)].reset_index(drop=True)


def filter_popular_products(df: pd.DataFrame, min_ratings: int = 5) -> pd.DataFrame:
    """
    Remove products with fewer than ``min_ratings`` interactions.

    Ensures every recommended product has enough community ratings to be
    meaningful.

    Parameters
    ----------
    df : pd.DataFrame
        Ratings DataFrame (ideally after user filtering).
    min_ratings : int
        Minimum number of ratings a product must have to be retained.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    product_counts = df["productId"].value_counts()
    popular_products = product_counts[product_counts >= min_ratings].index
    return df[df["productId"].isin(popular_products)].reset_index(drop=True)


def get_data_stats(df: pd.DataFrame) -> dict:
    """
    Compute key statistics about the ratings dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Ratings DataFrame.

    Returns
    -------
    dict
        n_users, n_products, n_interactions, sparsity (%),
        avg_rating, median_rating, rating_distribution.
    """
    n_users = df["userId"].nunique()
    n_products = df["productId"].nunique()
    n_interactions = len(df)
    possible_interactions = n_users * n_products
    sparsity = 1 - (n_interactions / possible_interactions) if possible_interactions > 0 else 1.0

    return {
        "n_users": n_users,
        "n_products": n_products,
        "n_interactions": n_interactions,
        "sparsity": round(sparsity * 100, 2),
        "avg_rating": round(df["rating"].mean(), 3),
        "median_rating": df["rating"].median(),
        "rating_distribution": df["rating"].value_counts().sort_index().to_dict(),
    }


def prepare_surprise_data(df: pd.DataFrame, rating_scale: tuple = (1, 5)):
    """
    Convert a pandas DataFrame into a Surprise Dataset object.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with userId, productId, rating columns.
    rating_scale : tuple
        Min and max possible rating values.

    Returns
    -------
    tuple
        (surprise.Dataset, surprise.Reader)
    """
    from surprise import Dataset, Reader

    reader = Reader(rating_scale=rating_scale)
    data = Dataset.load_from_df(df[["userId", "productId", "rating"]], reader)
    return data, reader


def split_data(data, test_size: float = 0.2, random_state: int = 42):
    """
    Split a Surprise Dataset into trainset and testset.

    Parameters
    ----------
    data : surprise.Dataset
        Full dataset.
    test_size : float
        Fraction for testing.
    random_state : int
        Random seed.

    Returns
    -------
    tuple
        (trainset, testset)
    """
    from surprise.model_selection import train_test_split

    return train_test_split(data, test_size=test_size, random_state=random_state)


def run_preprocessing_pipeline(
    path: Path = DATA_PATH,
    min_user_ratings: int = 50,
    min_product_ratings: int = 5,
    nrows: int = None,
) -> dict:
    """
    Execute the full preprocessing pipeline.

    Steps
    -----
    load → filter active users → filter popular products →
    prepare Surprise data → train/test split

    Parameters
    ----------
    path : Path
        CSV file path.
    min_user_ratings : int
        Minimum user interaction threshold (default 50).
    min_product_ratings : int
        Minimum product interaction threshold (default 5).
    nrows : int, optional
        Row limit for faster development runs.

    Returns
    -------
    dict
        Keys: df_raw, df_filtered, stats_raw, stats_filtered,
        data, trainset, testset.
    """
    df_raw = load_data(path, nrows=nrows)
    stats_raw = get_data_stats(df_raw)

    df_filtered = filter_active_users(df_raw, min_ratings=min_user_ratings)
    df_filtered = filter_popular_products(df_filtered, min_ratings=min_product_ratings)
    stats_filtered = get_data_stats(df_filtered)

    data, reader = prepare_surprise_data(df_filtered)
    trainset, testset = split_data(data)

    return {
        "df_raw": df_raw,
        "df_filtered": df_filtered,
        "stats_raw": stats_raw,
        "stats_filtered": stats_filtered,
        "data": data,
        "trainset": trainset,
        "testset": testset,
    }
