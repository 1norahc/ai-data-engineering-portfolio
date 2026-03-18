"""
preprocessing.py — Lead Conversion Prediction (ExtraaLearn)

Handles data loading, outlier treatment, feature encoding, and
train/test splitting for the lead conversion classification task.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "ExtraaLearn.csv"

# ── Feature groups ─────────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "age", "website_visits", "time_spent_on_website", "page_views_per_visit"
]
CATEGORICAL_FEATURES = [
    "current_occupation", "first_interaction", "last_activity"
]
BINARY_FEATURES = [
    "print_media_type1", "print_media_type2", "digital_media",
    "educational_channels", "referral",
]
ORDINAL_MAP = {"Low": 0, "Medium": 1, "High": 2}
TARGET = "status"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the ExtraaLearn lead dataset.

    Parameters
    ----------
    path : Path
        CSV file path.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with 4,612 rows and 15 columns.
    """
    df = pd.read_csv(path)
    return df


def clip_outliers(
    df: pd.DataFrame,
    columns: list = None,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> pd.DataFrame:
    """
    Clip numeric columns at the specified quantile bounds.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list, optional
        Columns to clip. Defaults to NUMERIC_FEATURES.
    lower_q : float
        Lower quantile bound (default 0.01).
    upper_q : float
        Upper quantile bound (default 0.99).

    Returns
    -------
    pd.DataFrame
        DataFrame with clipped values.
    """
    df = df.copy()
    cols = columns or NUMERIC_FEATURES
    for col in cols:
        if col in df.columns:
            lo = df[col].quantile(lower_q)
            hi = df[col].quantile(upper_q)
            df[col] = df[col].clip(lower=lo, upper=hi)
    return df


def encode_ordinal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply ordinal encoding to ``profile_completed`` and create a binary
    ``is_professional`` flag from ``current_occupation``.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``profile_completed_ord`` and ``is_professional`` columns.
    """
    df = df.copy()
    if "profile_completed" in df.columns:
        df["profile_completed_ord"] = df["profile_completed"].map(ORDINAL_MAP)
    if "current_occupation" in df.columns:
        df["is_professional"] = (df["current_occupation"] == "Professional").astype(int)
    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Construct a scikit-learn ColumnTransformer that:
    - StandardScales all numeric features + profile_completed_ord
    - OneHotEncodes (drop='first') categorical features
    - Passes binary and is_professional features through unchanged

    Returns
    -------
    ColumnTransformer
        Unfitted preprocessor ready for use inside a Pipeline.
    """
    numeric_cols = NUMERIC_FEATURES + ["profile_completed_ord"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="passthrough",  # keeps binary features + is_professional
    )
    return preprocessor


def split_features_target(df: pd.DataFrame) -> tuple:
    """
    Separate feature matrix X from target vector y.

    Parameters
    ----------
    df : pd.DataFrame
        Full DataFrame with target column.

    Returns
    -------
    tuple
        (X : pd.DataFrame, y : pd.Series)
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def run_preprocessing_pipeline(
    path: Path = DATA_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Execute the full preprocessing pipeline.

    Steps
    -----
    load → clip outliers → encode ordinal → train/test split

    Parameters
    ----------
    path : Path
        CSV file path.
    test_size : float
        Fraction of data held out for testing (default 0.2).
    random_state : int
        Reproducibility seed.

    Returns
    -------
    dict
        Keys: ``df``, ``X_train``, ``X_test``, ``y_train``, ``y_test``,
        ``preprocessor``.
    """
    df = load_data(path)
    df = clip_outliers(df)
    df = encode_ordinal(df)

    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = build_preprocessor()

    return {
        "df": df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
    }
