"""
models.py — Lead Conversion Prediction (ExtraaLearn)

Decision Tree (with cost-complexity pruning) and Random Forest
(with RandomizedSearchCV tuning) classifiers wrapped in sklearn Pipelines.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    RandomizedSearchCV,
    cross_val_score,
    StratifiedKFold,
)


# ── Decision Tree ──────────────────────────────────────────────────────────────

def build_decision_tree_pipeline(preprocessor) -> Pipeline:
    """
    Build an unfitted Decision Tree pipeline.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Pre-built sklearn preprocessor.

    Returns
    -------
    Pipeline
        sklearn Pipeline with preprocessor + DecisionTreeClassifier.
    """
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(random_state=42)),
    ])


def find_best_ccp_alpha(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
) -> float:
    """
    Determine the optimal cost-complexity pruning alpha via cross-validation.

    Fits the tree path, then selects the ccp_alpha that maximises mean CV
    accuracy on the training set.

    Parameters
    ----------
    pipeline : Pipeline
        Unfitted pipeline (preprocessor + DecisionTree).
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    cv : int
        Number of cross-validation folds.

    Returns
    -------
    float
        Best ccp_alpha value.
    """
    # Fit the preprocessor to get transformed data for pruning path
    from sklearn.base import clone
    pipe_clone = clone(pipeline)
    pipe_clone.named_steps["preprocessor"].fit(X_train)
    X_transformed = pipe_clone.named_steps["preprocessor"].transform(X_train)

    # Get pruning path
    dt = DecisionTreeClassifier(random_state=42)
    path = dt.cost_complexity_pruning_path(X_transformed, y_train)
    ccp_alphas = path.ccp_alphas[:-1]  # exclude the trivial last value

    # Cross-validate each alpha
    cv_scores = []
    kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    for alpha in ccp_alphas:
        dt_cv = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
        scores = cross_val_score(dt_cv, X_transformed, y_train, cv=kfold, scoring="roc_auc")
        cv_scores.append(scores.mean())

    best_alpha = ccp_alphas[np.argmax(cv_scores)]
    return float(best_alpha)


def train_decision_tree(
    preprocessor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    ccp_alpha: float = None,
) -> Pipeline:
    """
    Train a Decision Tree classifier. Finds optimal ccp_alpha via CV if not provided.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Pre-built sklearn preprocessor.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    ccp_alpha : float, optional
        Pruning alpha. If None, computed automatically.

    Returns
    -------
    Pipeline
        Fitted pipeline.
    """
    pipeline = build_decision_tree_pipeline(preprocessor)

    if ccp_alpha is None:
        ccp_alpha = find_best_ccp_alpha(pipeline, X_train, y_train)

    pipeline.set_params(classifier__ccp_alpha=ccp_alpha)
    pipeline.fit(X_train, y_train)
    return pipeline


# ── Random Forest ──────────────────────────────────────────────────────────────

def build_random_forest_pipeline(preprocessor) -> Pipeline:
    """
    Build an unfitted Random Forest pipeline.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Pre-built sklearn preprocessor.

    Returns
    -------
    Pipeline
        sklearn Pipeline with preprocessor + RandomForestClassifier.
    """
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=400, random_state=42, n_jobs=-1
        )),
    ])


def tune_random_forest(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 25,
    cv: int = 5,
    random_state: int = 42,
) -> Pipeline:
    """
    Tune Random Forest hyperparameters with RandomizedSearchCV.

    Search space
    ------------
    - max_depth       : [5, 10, 15, None]
    - min_samples_split : [2, 5, 10]
    - min_samples_leaf  : [1, 2, 4]
    - max_features      : ['sqrt', 'log2', 0.5]

    Parameters
    ----------
    pipeline : Pipeline
        Unfitted RF pipeline.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    n_iter : int
        Number of random search iterations (default 25).
    cv : int
        Cross-validation folds (default 5).
    random_state : int
        Random seed.

    Returns
    -------
    Pipeline
        Best fitted pipeline found by RandomizedSearchCV.
    """
    param_dist = {
        "classifier__max_depth": [5, 10, 15, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2", 0.5],
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state),
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_


def train_random_forest(
    preprocessor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tune: bool = True,
) -> Pipeline:
    """
    Train a Random Forest classifier, optionally with hyperparameter tuning.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Pre-built sklearn preprocessor.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    tune : bool
        If True, run RandomizedSearchCV (slower but better). Default True.

    Returns
    -------
    Pipeline
        Fitted pipeline.
    """
    pipeline = build_random_forest_pipeline(preprocessor)
    if tune:
        return tune_random_forest(pipeline, X_train, y_train)
    else:
        pipeline.fit(X_train, y_train)
        return pipeline
