"""
evaluation.py — Amazon Product Recommendation System

Evaluation metrics for recommendation models:
- RMSE (rating prediction accuracy)
- Precision@K, Recall@K, F1@K (recommendation quality)
- Model comparison charts
"""

import os
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np
from collections import defaultdict
from surprise import accuracy

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


def compute_rmse(predictions) -> float:
    """
    Compute Root Mean Squared Error from a Surprise predictions list.

    Parameters
    ----------
    predictions : list
        Output of ``algo.test(testset)``.

    Returns
    -------
    float
        RMSE value.
    """
    return accuracy.rmse(predictions, verbose=False)


def precision_recall_at_k(
    predictions,
    k: int = 10,
    threshold: float = 3.5,
) -> tuple:
    """
    Compute Precision@K, Recall@K, and F1@K averaged over all users.

    A rating is considered **relevant** if the true rating >= threshold.
    A rating is considered **recommended** if the estimated rating >= threshold.

    Parameters
    ----------
    predictions : list
        List of Surprise Prediction named-tuples.
    k : int
        Number of top predictions per user to consider.
    threshold : float
        Minimum rating to consider relevant / recommended.

    Returns
    -------
    tuple
        (precision@k, recall@k, f1@k) — all floats.
    """
    user_est_true = defaultdict(list)
    for pred in predictions:
        user_est_true[pred.uid].append((pred.est, pred.r_ui))

    precisions = []
    recalls = []

    for uid, user_ratings in user_est_true.items():
        # Sort by estimated rating descending → take top-k
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]

        n_relevant = sum(1 for _, true_r in user_ratings if true_r >= threshold)
        n_recommended_and_relevant = sum(
            1 for est, true_r in top_k if est >= threshold and true_r >= threshold
        )
        n_recommended = sum(1 for est, _ in top_k if est >= threshold)

        precision = n_recommended_and_relevant / n_recommended if n_recommended > 0 else 0.0
        recall = n_recommended_and_relevant / n_relevant if n_relevant > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)

    avg_precision = float(np.mean(precisions)) if precisions else 0.0
    avg_recall = float(np.mean(recalls)) if recalls else 0.0
    f1 = (
        2 * avg_precision * avg_recall / (avg_precision + avg_recall)
        if (avg_precision + avg_recall) > 0
        else 0.0
    )
    return avg_precision, avg_recall, f1


def evaluate_model(
    algo,
    testset,
    model_name: str = "Model",
    k: int = 10,
    threshold: float = 3.5,
) -> dict:
    """
    Full evaluation of a fitted Surprise algorithm on the test set.

    Parameters
    ----------
    algo : Surprise algorithm
        Fitted recommendation model.
    testset : list
        Surprise testset (output of train_test_split).
    model_name : str
        Label for display in comparison tables.
    k : int
        K for ranking metrics.
    threshold : float
        Relevance threshold for Precision/Recall.

    Returns
    -------
    dict
        model, rmse, precision@k, recall@k, f1@k.
    """
    predictions = algo.test(testset)
    rmse = compute_rmse(predictions)
    precision, recall, f1 = precision_recall_at_k(predictions, k=k, threshold=threshold)

    return {
        "model": model_name,
        "rmse": round(rmse, 4),
        f"precision@{k}": round(precision, 4),
        f"recall@{k}": round(recall, 4),
        f"f1@{k}": round(f1, 4),
        "predictions": predictions,
    }


def plot_model_comparison(
    results_list: list,
    k: int = 10,
    save: bool = False,
    path: str = None,
) -> pd.DataFrame:
    """
    Bar chart comparing RMSE, Precision@K, Recall@K, F1@K across models.

    Parameters
    ----------
    results_list : list
        List of result dicts from ``evaluate_model``.
    k : int
        K used in metric names (for column lookup).

    Returns
    -------
    pd.DataFrame
        Comparison DataFrame.
    """
    rows = []
    for r in results_list:
        rows.append({
            "Model": r["model"],
            "RMSE": r["rmse"],
            f"Precision@{k}": r[f"precision@{k}"],
            f"Recall@{k}": r[f"recall@{k}"],
            f"F1@{k}": r[f"f1@{k}"],
        })
    cmp_df = pd.DataFrame(rows).set_index("Model")

    # Two sub-plots: RMSE (lower=better) and Ranking metrics (higher=better)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    cmp_df[["RMSE"]].plot(kind="bar", ax=ax1, color="#e74c3c", legend=False)
    ax1.set_title("RMSE (lower is better)", fontweight="bold")
    ax1.set_ylabel("RMSE")
    ax1.tick_params(axis="x", rotation=15)
    ax1.set_ylim(0, cmp_df["RMSE"].max() * 1.15)

    ranking_cols = [c for c in cmp_df.columns if c != "RMSE"]
    cmp_df[ranking_cols].plot(kind="bar", ax=ax2, colormap="Set2", edgecolor="white")
    ax2.set_title(f"Ranking Metrics @ {k} (higher is better)", fontweight="bold")
    ax2.set_ylabel("Score")
    ax2.tick_params(axis="x", rotation=15)
    ax2.set_ylim(0, 1.0)
    ax2.legend(loc="lower right")

    fig.suptitle("Model Performance Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_or_show(fig, save, "model_comparison.png", path)

    return cmp_df
