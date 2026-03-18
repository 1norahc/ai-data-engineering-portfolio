"""
recommenders.py — Amazon Product Recommendation System

Four recommendation approaches:
1. Rank-Based    — popularity baseline (no personalisation)
2. User-User CF  — KNN collaborative filtering (user similarity)
3. Item-Item CF  — KNN collaborative filtering (item similarity)
4. SVD           — matrix factorisation (latent factors)

All CF models use the Surprise library.
"""

import pandas as pd
import numpy as np
from surprise import KNNBasic, SVD


# ── 1. Rank-Based Recommender ──────────────────────────────────────────────────

def get_rank_based_recommendations(
    df: pd.DataFrame,
    min_interactions: int = 50,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Recommend the most popular products by average rating.

    Products with fewer than ``min_interactions`` ratings are excluded to
    avoid high ratings from very few users skewing results.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered ratings DataFrame with userId, productId, rating columns.
    min_interactions : int
        Minimum number of ratings a product must have (default 50).
    top_n : int
        Number of recommendations to return.

    Returns
    -------
    pd.DataFrame
        Top-N products with columns: productId, avg_rating, num_ratings, rank.
    """
    product_stats = (
        df.groupby("productId")
        .agg(avg_rating=("rating", "mean"), num_ratings=("rating", "count"))
        .reset_index()
    )
    popular = product_stats[product_stats["num_ratings"] >= min_interactions]
    top = popular.sort_values("avg_rating", ascending=False).head(top_n).copy()
    top["rank"] = range(1, len(top) + 1)
    top["avg_rating"] = top["avg_rating"].round(3)
    return top.reset_index(drop=True)


# ── 2. User-User Collaborative Filtering ──────────────────────────────────────

def train_user_user_cf(
    trainset,
    k: int = 60,
    min_k: int = 5,
) -> KNNBasic:
    """
    Train a User-User Collaborative Filtering model using cosine similarity.

    Finds similar users and recommends products they liked.
    Best hyperparameters found via GridSearchCV: k=60, min_k=5.

    Parameters
    ----------
    trainset : surprise.Trainset
        Training data in Surprise format.
    k : int
        Number of nearest neighbours to consider.
    min_k : int
        Minimum number of neighbours required for a prediction.

    Returns
    -------
    KNNBasic
        Fitted User-User CF model.
    """
    sim_options = {"name": "cosine", "user_based": True}
    algo = KNNBasic(k=k, min_k=min_k, sim_options=sim_options, verbose=False)
    algo.fit(trainset)
    return algo


# ── 3. Item-Item Collaborative Filtering ──────────────────────────────────────

def train_item_item_cf(
    trainset,
    k: int = 30,
    min_k: int = 6,
) -> KNNBasic:
    """
    Train an Item-Item Collaborative Filtering model using cosine similarity.

    Finds similar products to those the user has already rated.
    Best hyperparameters: k=30, min_k=6 (GridSearchCV).

    Parameters
    ----------
    trainset : surprise.Trainset
        Training data in Surprise format.
    k : int
        Number of nearest neighbour items.
    min_k : int
        Minimum neighbours for a valid prediction.

    Returns
    -------
    KNNBasic
        Fitted Item-Item CF model.
    """
    sim_options = {"name": "cosine", "user_based": False}
    algo = KNNBasic(k=k, min_k=min_k, sim_options=sim_options, verbose=False)
    algo.fit(trainset)
    return algo


# ── 4. SVD Matrix Factorisation ───────────────────────────────────────────────

def train_svd(
    trainset,
    n_epochs: int = 30,
    lr_all: float = 0.005,
    reg_all: float = 0.02,
) -> SVD:
    """
    Train an SVD Matrix Factorisation model.

    Decomposes the user-item matrix into latent factor vectors, learning
    underlying user preferences and item characteristics.
    Best configuration (GridSearchCV): n_epochs=30, lr_all=0.005, reg_all=0.02.
    Achieves RMSE 0.9039 — best among all models.

    Parameters
    ----------
    trainset : surprise.Trainset
        Training data.
    n_epochs : int
        Number of SGD iterations.
    lr_all : float
        Learning rate for all parameters.
    reg_all : float
        L2 regularisation factor.

    Returns
    -------
    SVD
        Fitted SVD model.
    """
    algo = SVD(n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)
    algo.fit(trainset)
    return algo


# ── Hyperparameter Tuning ──────────────────────────────────────────────────────

def tune_user_user_cf(data) -> dict:
    """
    Grid-search optimal hyperparameters for User-User CF.

    Search space
    ------------
    k: [20, 40, 60], min_k: [1, 3, 5], similarity: [cosine, pearson]

    Returns
    -------
    dict
        best_params, best_rmse
    """
    from surprise.model_selection import GridSearchCV as SurpriseGridSearchCV

    param_grid = {
        "k": [20, 40, 60],
        "min_k": [1, 3, 5],
        "sim_options": {
            "name": ["cosine", "pearson"],
            "user_based": [True],
        },
    }
    gs = SurpriseGridSearchCV(KNNBasic, param_grid, measures=["rmse"], cv=3, n_jobs=-1)
    gs.fit(data)
    return {"best_params": gs.best_params["rmse"], "best_rmse": gs.best_score["rmse"]}


def tune_item_item_cf(data) -> dict:
    """
    Grid-search optimal hyperparameters for Item-Item CF.

    Search space
    ------------
    k: [10, 20, 30], min_k: [3, 6, 9], similarity: [cosine, msd]

    Returns
    -------
    dict
        best_params, best_rmse
    """
    from surprise.model_selection import GridSearchCV as SurpriseGridSearchCV

    param_grid = {
        "k": [10, 20, 30],
        "min_k": [3, 6, 9],
        "sim_options": {
            "name": ["cosine", "msd"],
            "user_based": [False],
        },
    }
    gs = SurpriseGridSearchCV(KNNBasic, param_grid, measures=["rmse"], cv=3, n_jobs=-1)
    gs.fit(data)
    return {"best_params": gs.best_params["rmse"], "best_rmse": gs.best_score["rmse"]}


def tune_svd(data) -> dict:
    """
    Grid-search optimal hyperparameters for SVD.

    Search space
    ------------
    n_epochs: [20, 30, 40], lr_all: [0.002, 0.005, 0.008], reg_all: [0.02, 0.05, 0.1]

    Returns
    -------
    dict
        best_params, best_rmse
    """
    from surprise.model_selection import GridSearchCV as SurpriseGridSearchCV

    param_grid = {
        "n_epochs": [20, 30, 40],
        "lr_all": [0.002, 0.005, 0.008],
        "reg_all": [0.02, 0.05, 0.1],
    }
    gs = SurpriseGridSearchCV(SVD, param_grid, measures=["rmse"], cv=3, n_jobs=-1)
    gs.fit(data)
    return {"best_params": gs.best_params["rmse"], "best_rmse": gs.best_score["rmse"]}


# ── Recommendation Generation ──────────────────────────────────────────────────

def get_recommendations_for_user(
    algo,
    user_id: str,
    df: pd.DataFrame,
    n: int = 10,
    threshold: float = 3.5,
) -> pd.DataFrame:
    """
    Generate top-N product recommendations for a specific user.

    Only considers products the user has NOT already rated.
    Filters by a minimum estimated rating threshold.

    Parameters
    ----------
    algo : Surprise algorithm
        Fitted recommendation model (KNNBasic or SVD).
    user_id : str
        Target user identifier.
    df : pd.DataFrame
        Filtered ratings DataFrame (to know what user has already rated).
    n : int
        Number of recommendations to return.
    threshold : float
        Minimum estimated rating to include in recommendations (default 3.5).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: productId, estimated_rating, rank.
        Returns empty DataFrame if user not in training data.
    """
    all_products = df["productId"].unique()
    rated_products = set(df[df["userId"] == user_id]["productId"].tolist())
    unrated_products = [p for p in all_products if p not in rated_products]

    if not unrated_products:
        return pd.DataFrame(columns=["productId", "estimated_rating", "rank"])

    predictions = [algo.predict(user_id, product_id) for product_id in unrated_products]
    recs = [
        (pred.iid, round(pred.est, 3))
        for pred in predictions
        if pred.est >= threshold
    ]
    recs.sort(key=lambda x: x[1], reverse=True)
    recs = recs[:n]

    result = pd.DataFrame(recs, columns=["productId", "estimated_rating"])
    result["rank"] = range(1, len(result) + 1)
    return result
