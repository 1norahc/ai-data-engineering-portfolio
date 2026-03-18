"""
evaluation.py — Lead Conversion Prediction (ExtraaLearn)

Model evaluation utilities: metrics computation, confusion matrices,
ROC/PR curves, feature importance, and model comparison charts.
"""

import os
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    classification_report,
)

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


def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> dict:
    """
    Evaluate a fitted pipeline on the test set.

    Parameters
    ----------
    model : sklearn Pipeline
        Fitted model pipeline.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        True labels.
    model_name : str
        Label for display.

    Returns
    -------
    dict
        accuracy, roc_auc, pr_auc, precision, recall, f1,
        y_pred, y_proba, report.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "pr_auc": round(average_precision_score(y_test, y_proba), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "y_pred": y_pred,
        "y_proba": y_proba,
        "report": classification_report(y_test, y_pred, target_names=["Not Converted", "Converted"]),
    }
    return metrics


def plot_confusion_matrix(
    y_true, y_pred, model_name: str = "Model",
    save: bool = False, path: str = None
):
    """
    Seaborn heatmap of the confusion matrix.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    model_name : str
        Used in title.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Not Converted", "Converted"],
        yticklabels=["Not Converted", "Converted"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}", fontweight="bold")
    fig.tight_layout()

    filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    _save_or_show(fig, save, filename, path)


def plot_roc_curves(
    models_dict: dict, X_test, y_test,
    save: bool = False, path: str = None
):
    """
    Overlay ROC curves for multiple models on one axes.

    Parameters
    ----------
    models_dict : dict
        ``{'Model Name': fitted_pipeline, ...}``
    """
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    fig, ax = plt.subplots(figsize=(7, 6))

    for i, (name, model) in enumerate(models_dict.items()):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, color=colors[i % len(colors)],
                label=f"{name} (AUC={auc:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    _save_or_show(fig, save, "roc_curves.png", path)


def plot_precision_recall_curves(
    models_dict: dict, X_test, y_test,
    save: bool = False, path: str = None
):
    """
    Overlay Precision-Recall curves for multiple models.

    Parameters
    ----------
    models_dict : dict
        ``{'Model Name': fitted_pipeline, ...}``
    """
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    fig, ax = plt.subplots(figsize=(7, 6))

    for i, (name, model) in enumerate(models_dict.items()):
        y_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        ax.plot(recall, precision, color=colors[i % len(colors)],
                label=f"{name} (AP={ap:.3f})", linewidth=2)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Model Comparison", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    _save_or_show(fig, save, "pr_curves.png", path)


def plot_feature_importance(
    model_pipeline, feature_names: list = None,
    top_n: int = 15,
    save: bool = False, path: str = None
):
    """
    Horizontal bar chart of top-N feature importances from the classifier.

    Works with both DecisionTreeClassifier and RandomForestClassifier.

    Parameters
    ----------
    model_pipeline : Pipeline
        Fitted sklearn Pipeline.
    feature_names : list, optional
        Feature names after preprocessing. Auto-detected if None.
    top_n : int
        Number of top features to display.
    """
    clf = model_pipeline.named_steps["classifier"]
    importances = clf.feature_importances_

    if feature_names is None:
        preprocessor = model_pipeline.named_steps["preprocessor"]
        try:
            feature_names = preprocessor.get_feature_names_out().tolist()
        except Exception:
            feature_names = [f"feature_{i}" for i in range(len(importances))]

    feat_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(9, 6))
    feat_df.sort_values("Importance").plot(
        kind="barh", x="Feature", y="Importance",
        color="#9b59b6", legend=False, ax=ax
    )
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    fig.tight_layout()
    _save_or_show(fig, save, "feature_importance.png", path)


def compare_models(
    results_dict: dict,
    save: bool = False, path: str = None
) -> pd.DataFrame:
    """
    Print and optionally plot a model comparison table.

    Parameters
    ----------
    results_dict : dict
        ``{'Model Name': metrics_dict, ...}`` where metrics_dict has
        keys accuracy, roc_auc, pr_auc, precision, recall, f1.

    Returns
    -------
    pd.DataFrame
        Comparison DataFrame.
    """
    rows = []
    for name, m in results_dict.items():
        rows.append({
            "Model": name,
            "Accuracy": m["accuracy"],
            "ROC-AUC": m["roc_auc"],
            "PR-AUC": m["pr_auc"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1": m["f1"],
        })
    cmp_df = pd.DataFrame(rows).set_index("Model")

    # Bar chart
    metrics_to_plot = ["Accuracy", "ROC-AUC", "PR-AUC", "F1"]
    plot_df = cmp_df[metrics_to_plot]
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_df.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="white")
    ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0.5, 1.0)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save_or_show(fig, save, "model_comparison.png", path)

    return cmp_df
