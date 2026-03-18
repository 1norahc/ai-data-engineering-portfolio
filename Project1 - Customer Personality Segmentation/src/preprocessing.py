"""
preprocessing.py — Customer Personality Segmentation

Handles all data loading, cleaning, and feature engineering steps
before clustering is applied.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Default path: CSV sits at the project root
DATA_PATH = Path(__file__).parent.parent / "Customer_Personality_Segmentation.csv"

# Columns that carry no information (zero variance)
_CONSTANT_COLS = ["Z_CostContact", "Z_Revenue"]

# Spending columns used in feature engineering
_MNT_COLS = ["MntWines", "MntFruits", "MntMeatProducts",
             "MntFishProducts", "MntSweetProducts", "MntGoldProds"]

# Purchase-channel columns
_PURCHASE_COLS = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases",
                  "NumDealsPurchases"]

# Campaign response columns
_CAMPAIGN_COLS = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3",
                  "AcceptedCmp4", "AcceptedCmp5", "Response"]

# Columns to drop before clustering (identifiers, dates, raw year)
_DROP_FOR_CLUSTERING = ["ID", "Dt_Customer", "Year_Birth"]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the raw customer personality dataset from a tab-separated file.

    Parameters
    ----------
    path : Path
        File path to the CSV/TSV dataset.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with all 29 original columns.
    """
    df = pd.read_csv(path, sep="\t")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataset:
    - Fill 24 missing ``Income`` values with the column median.
    - Remove rows with extreme outliers (Income > 200 000, Year_Birth < 1900).
    - Drop constant columns (Z_CostContact, Z_Revenue).
    - Reset the index.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    df = df.copy()

    # Fill missing income
    df["Income"] = df["Income"].fillna(df["Income"].median())

    # Remove extreme outliers
    df = df[df["Income"] <= 200_000]
    df = df[df["Year_Birth"] >= 1900]

    # Drop zero-variance columns
    cols_to_drop = [c for c in _CONSTANT_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    df = df.reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame, reference_year: int = 2024) -> pd.DataFrame:
    """
    Add derived features that improve cluster separation:

    - ``Age``                 : reference_year - Year_Birth
    - ``Tenure``              : days since Dt_Customer (as of reference_year-01-01)
    - ``TotalSpending``       : sum of all Mnt* columns
    - ``TotalPurchases``      : sum of all purchase-channel columns
    - ``TotalCampaigns``      : total campaigns accepted (AcceptedCmp1-5 + Response)
    - ``SpendingPerPurchase`` : TotalSpending / (TotalPurchases + 1)
    - ``HasChildren``         : 1 if Kidhome + Teenhome > 0 else 0

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame.
    reference_year : int
        Year used to compute Age and Tenure (default 2024).

    Returns
    -------
    pd.DataFrame
        DataFrame with additional engineered columns.
    """
    df = df.copy()

    df["Age"] = reference_year - df["Year_Birth"]
    df["Tenure"] = (
        pd.Timestamp(f"{reference_year}-01-01") - pd.to_datetime(df["Dt_Customer"])
    ).dt.days

    mnt_cols = [c for c in _MNT_COLS if c in df.columns]
    df["TotalSpending"] = df[mnt_cols].sum(axis=1)

    purchase_cols = [c for c in _PURCHASE_COLS if c in df.columns]
    df["TotalPurchases"] = df[purchase_cols].sum(axis=1)

    campaign_cols = [c for c in _CAMPAIGN_COLS if c in df.columns]
    df["TotalCampaigns"] = df[campaign_cols].sum(axis=1)

    df["SpendingPerPurchase"] = df["TotalSpending"] / (df["TotalPurchases"] + 1)
    df["HasChildren"] = ((df.get("Kidhome", 0) + df.get("Teenhome", 0)) > 0).astype(int)

    return df


def get_feature_matrix(df: pd.DataFrame) -> tuple:
    """
    Prepare the numeric feature matrix for clustering.
    Drops identifier, date, and raw-year columns, then applies StandardScaler.

    Parameters
    ----------
    df : pd.DataFrame
        Engineered DataFrame.

    Returns
    -------
    tuple
        (X_scaled : np.ndarray, scaler : StandardScaler, feature_names : list[str])
    """
    drop_cols = [c for c in _DROP_FOR_CLUSTERING if c in df.columns]
    # Also drop any object-type columns that remain
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    X = df.drop(columns=drop_cols + obj_cols, errors="ignore")
    feature_names = X.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler, feature_names


def run_preprocessing_pipeline(path: Path = DATA_PATH) -> tuple:
    """
    Execute the full preprocessing pipeline:
    load → clean → engineer features → scale.

    Parameters
    ----------
    path : Path
        Path to the raw CSV file.

    Returns
    -------
    tuple
        (df_engineered : pd.DataFrame, X_scaled : np.ndarray, scaler : StandardScaler)
    """
    df = load_data(path)
    df = clean_data(df)
    df = engineer_features(df)
    X_scaled, scaler, _ = get_feature_matrix(df)
    return df, X_scaled, scaler
