"""
main.py — Lead Conversion Prediction Pipeline

Runs EDA, trains Decision Tree and Random Forest classifiers,
evaluates both, and prints a comparison table.

Usage
-----
    python main.py

Outputs
-------
    reports/figures/          — all generated plots
    reports/model_results.csv — model comparison metrics
"""

import os
import sys
import joblib

sys.path.insert(0, os.path.dirname(__file__))

from src.preprocessing import run_preprocessing_pipeline
from src.eda import run_full_eda
from src.models import train_decision_tree, train_random_forest
from src.evaluation import (
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_feature_importance,
    compare_models,
)

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
MODELS_DIR = os.path.join(REPORTS_DIR, "models")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 60)
    print("  Lead Conversion Prediction Pipeline — ExtraaLearn")
    print("=" * 60)

    # ── 1. Preprocessing ──────────────────────────────────────────────────────
    print("\n[1/5] Loading and preprocessing data …")
    data = run_preprocessing_pipeline()
    df = data["df"]
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    preprocessor = data["preprocessor"]
    print(f"      Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")
    print(f"      Conversion rate: {y_train.mean():.1%} (train)")

    # ── 2. EDA ────────────────────────────────────────────────────────────────
    print("\n[2/5] Running exploratory data analysis …")
    run_full_eda(df, save=True, path=FIGURES_DIR)
    print("      Plots saved to reports/figures/")

    # ── 3. Decision Tree ──────────────────────────────────────────────────────
    print("\n[3/5] Training Decision Tree (with cost-complexity pruning) …")
    dt_model = train_decision_tree(preprocessor, X_train, y_train)
    dt_metrics = evaluate_model(dt_model, X_test, y_test, "Decision Tree")
    plot_confusion_matrix(y_test, dt_metrics["y_pred"], "Decision Tree",
                          save=True, path=FIGURES_DIR)
    joblib.dump(dt_model, os.path.join(MODELS_DIR, "decision_tree.joblib"))
    print(f"      Accuracy: {dt_metrics['accuracy']:.4f}  |  ROC-AUC: {dt_metrics['roc_auc']:.4f}")

    # ── 4. Random Forest ──────────────────────────────────────────────────────
    print("\n[4/5] Training Random Forest (RandomizedSearchCV tuning) …")
    rf_model = train_random_forest(preprocessor, X_train, y_train, tune=True)
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    plot_confusion_matrix(y_test, rf_metrics["y_pred"], "Random Forest",
                          save=True, path=FIGURES_DIR)
    joblib.dump(rf_model, os.path.join(MODELS_DIR, "random_forest.joblib"))
    print(f"      Accuracy: {rf_metrics['accuracy']:.4f}  |  ROC-AUC: {rf_metrics['roc_auc']:.4f}")

    # ── 5. Comparison & reporting ─────────────────────────────────────────────
    print("\n[5/5] Generating comparison charts and saving results …")
    models = {"Decision Tree": dt_model, "Random Forest": rf_model}
    results = {"Decision Tree": dt_metrics, "Random Forest": rf_metrics}

    plot_roc_curves(models, X_test, y_test, save=True, path=FIGURES_DIR)
    plot_feature_importance(rf_model, save=True, path=FIGURES_DIR)

    cmp_df = compare_models(results, save=True, path=FIGURES_DIR)
    cmp_df.to_csv(os.path.join(REPORTS_DIR, "model_results.csv"))

    print("\n── Model Comparison ─────────────────────────────────────")
    print(cmp_df.to_string())

    print("\n── Decision Tree Classification Report ──────────────────")
    print(dt_metrics["report"])

    print("\n── Random Forest Classification Report ──────────────────")
    print(rf_metrics["report"])

    print(f"\n Done!  Results saved to {REPORTS_DIR}/")


if __name__ == "__main__":
    main()
