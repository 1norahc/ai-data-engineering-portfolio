"""
clustering.py — Customer Personality Segmentation

K-Means clustering: optimal-k selection, model fitting, cluster assignment,
and cluster profiling.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Human-readable names assigned to the 5 discovered clusters
CLUSTER_NAMES = {
    0: "Affluent Traditionalists",
    1: "Low-Income Minimal Spenders",
    2: "Premium Heavy Spenders",
    3: "Dormant Families",
    4: "Digital-Savvy Spenders",
}

# Key numeric columns used in cluster profiling
_PROFILE_COLS = [
    "Income", "Age", "Tenure", "TotalSpending", "TotalPurchases",
    "TotalCampaigns", "MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts", "MntGoldProds",
    "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases",
    "HasChildren",
]


def compute_elbow_metrics(X: np.ndarray, max_k: int = 10,
                          random_state: int = 42) -> dict:
    """
    Compute inertia and silhouette score for k = 2 … max_k.

    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix.
    max_k : int
        Maximum number of clusters to evaluate.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Keys: ``k_values``, ``inertias``, ``silhouette_scores``.
    """
    k_values = list(range(2, max_k + 1))
    inertias = []
    silhouette_scores = []

    for k in k_values:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(X)
        inertias.append(model.inertia_)
        silhouette_scores.append(silhouette_score(X, labels))

    return {
        "k_values": k_values,
        "inertias": inertias,
        "silhouette_scores": silhouette_scores,
    }


def fit_kmeans(X: np.ndarray, n_clusters: int = 5,
               random_state: int = 42) -> KMeans:
    """
    Fit a K-Means model with the given number of clusters.

    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix.
    n_clusters : int
        Number of clusters (default 5, chosen via elbow analysis).
    random_state : int
        Random seed.

    Returns
    -------
    KMeans
        Fitted scikit-learn KMeans object.
    """
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    model.fit(X)
    return model


def assign_clusters(df: pd.DataFrame, model: KMeans,
                    X_scaled: np.ndarray) -> pd.DataFrame:
    """
    Add ``Cluster`` (int) and ``Cluster_Name`` (str) columns to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Engineered DataFrame (same row order as X_scaled).
    model : KMeans
        Fitted KMeans model.
    X_scaled : np.ndarray
        Scaled feature matrix.

    Returns
    -------
    pd.DataFrame
        DataFrame with two new columns: ``Cluster`` and ``Cluster_Name``.
    """
    df = df.copy()
    df["Cluster"] = model.predict(X_scaled)
    df["Cluster_Name"] = df["Cluster"].map(CLUSTER_NAMES)
    return df


def get_cluster_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean values of key metrics per cluster.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``Cluster`` column already assigned.

    Returns
    -------
    pd.DataFrame
        Mean profile per cluster (rows = clusters, columns = metrics).
    """
    cols = [c for c in _PROFILE_COLS if c in df.columns]
    return df.groupby("Cluster")[cols].mean().round(2)


def get_cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    High-level summary table: size, avg income, avg spending, avg age per cluster.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``Cluster`` and ``Cluster_Name`` columns.

    Returns
    -------
    pd.DataFrame
        Summary with columns: Cluster, Name, Size, AvgIncome, AvgSpending, AvgAge.
    """
    summary = (
        df.groupby(["Cluster", "Cluster_Name"])
        .agg(
            Size=("Cluster", "count"),
            AvgIncome=("Income", "mean"),
            AvgSpending=("TotalSpending", "mean"),
            AvgAge=("Age", "mean"),
            AvgCampaigns=("TotalCampaigns", "mean"),
        )
        .reset_index()
        .rename(columns={"Cluster_Name": "Name"})
        .round({"AvgIncome": 0, "AvgSpending": 1, "AvgAge": 1, "AvgCampaigns": 2})
    )
    return summary
