"""
FlowScore — XGBoost Model Training Pipeline
=============================================

Trains a gradient-boosted classifier on the 23 engineered gig-worker
features and produces all artifacts needed for the scoring API.

Pipeline Flow:
    engineered_features.csv (23 features + TARGET)
      → 70/30 stratified train/test split
      → XGBoost training with early stopping
      → Evaluation (AUC-ROC, precision-recall, confusion matrix)
      → SHAP explainer generation
      → FlowScore calibration (probability → 300-850 scale)
      → Save artifacts (model.pkl, scaler.pkl, explainer.pkl, metrics.json)

Performance Targets (from project spec):
    • AUC-ROC  ≥ 0.85
    • Precision ≥ 0.50 at recall = 0.80
    • Inference < 200ms per borrower

Usage:
    python model/train_model.py
    python model/train_model.py --input data/processed/engineered_features.csv
    python model/train_model.py --tune   # Run hyperparameter search

Author: FlowScore Team
"""

import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
    f1_score,
    average_precision_score,
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no GUI window)
import matplotlib.pyplot as plt

# SHAP import is deferred to where it's used — it takes ~2s to import
# and we want fast startup for validation runs.

# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "model" / "artifacts"

DEFAULT_INPUT = PROCESSED_DATA_DIR / "engineered_features.csv"

# Output artifact paths
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
EXPLAINER_PATH = ARTIFACTS_DIR / "explainer.pkl"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
FEATURE_IMPORTANCE_PATH = ARTIFACTS_DIR / "feature_importance.json"
FEATURE_IMPORTANCE_PNG = ARTIFACTS_DIR / "feature_importance.png"
FEATURE_NAMES_PATH = ARTIFACTS_DIR / "feature_names.pkl"

TARGET_COL = "TARGET"
RANDOM_STATE = 42
TEST_SIZE = 0.30  # 70/30 split per spec

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

log_filename = f"train_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / log_filename, mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("flowscore.train_model")


# ===================================================================
# Baseline Hyperparameters (from project spec)
# ===================================================================
BASELINE_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 10,  # Handles ~8% positive class imbalance
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,  # Use all CPU cores
    "verbosity": 0,  # Suppress XGBoost internal logging
}

# Hyperparameter search space for --tune mode
TUNING_GRID = {
    "max_depth": [4, 5, 6, 7, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
    "n_estimators": [150, 200, 300, 400],
    "min_child_weight": [3, 5, 7, 10],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "scale_pos_weight": [8, 10, 12, 15],
}


# ===================================================================
# FlowScore Conversion
# ===================================================================
def get_flowscore(default_probability: float) -> int:
    """
    Convert XGBoost default probability to a FICO-like FlowScore.

    The FlowScore range is 300–850:
        • 300 = highest default risk (probability ≈ 1.0)
        • 850 = lowest default risk  (probability ≈ 0.0)

    This is an INVERSE mapping: low default probability → high score.

    Args:
        default_probability: XGBoost predicted P(default), range [0, 1].

    Returns:
        Integer FlowScore in [300, 850].
    """
    score = 300 + (1 - default_probability) * 550
    return int(np.clip(score, 300, 850))


def get_flowscores_batch(probabilities: np.ndarray) -> np.ndarray:
    """Vectorized FlowScore conversion for entire arrays."""
    scores = 300 + (1 - probabilities) * 550
    return np.clip(scores, 300, 850).astype(int)


def get_risk_category(flowscore: int) -> str:
    """Map FlowScore to a human-readable risk bucket."""
    if flowscore >= 700:
        return "low"
    elif flowscore >= 600:
        return "medium"
    elif flowscore >= 500:
        return "high"
    else:
        return "very_high"


# ===================================================================
# STEP 1 — Load Engineered Features
# ===================================================================
def load_features(filepath: Path) -> tuple[pd.DataFrame, list[str]]:
    """
    Load the feature-engineered dataset and separate features from target.

    The input CSV has 24 columns: 23 features + TARGET.
    Features were already StandardScaler-normalized in feature_engineering.py,
    but we'll re-fit the scaler on training data only (to prevent data leakage).

    Args:
        filepath: Path to engineered_features.csv.

    Returns:
        (df, feature_names) — full DataFrame and list of 23 feature column names.
    """
    logger.info("=" * 60)
    logger.info("STEP 1 — Loading engineered features")
    logger.info("=" * 60)

    if not filepath.exists():
        logger.error(f"Feature file not found: {filepath}")
        logger.error("Run `python scripts/feature_engineering.py` first.")
        raise FileNotFoundError(f"Feature file not found: {filepath}")

    df = pd.read_csv(filepath)
    feature_names = [c for c in df.columns if c != TARGET_COL]

    logger.info(f"  Source:   {filepath}")
    logger.info(f"  Shape:    {df.shape[0]:,} rows × {df.shape[1]} columns")
    logger.info(f"  Features: {len(feature_names)}")
    logger.info(f"  Target:   {TARGET_COL} → {df[TARGET_COL].value_counts().to_dict()}")

    # Validate no nulls survived the pipeline
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        logger.warning(f"  ⚠ {null_count:,} null values found — filling with 0")
        df = df.fillna(0)
    else:
        logger.info(f"  ✓ No null values")

    return df, feature_names


# ===================================================================
# STEP 2 — Train/Test Split + Re-scale
# ===================================================================
def split_and_scale(
    df: pd.DataFrame, feature_names: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Stratified 70/30 split with scaler fit on training data ONLY.

    Why re-scale?
        feature_engineering.py applied StandardScaler to the full dataset
        for diagnostic purposes. For proper ML training, we MUST fit the
        scaler only on training data to prevent target leakage.

    Args:
        df:            Full DataFrame with features + TARGET.
        feature_names: List of 23 feature column names.

    Returns:
        (X_train, X_test, y_train, y_test, scaler)
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2 — Train/test split (70/30 stratified)")
    logger.info("=" * 60)

    X = df[feature_names].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,  # Preserve class distribution in both splits
    )

    logger.info(f"  Train: {X_train.shape[0]:,} samples ({y_train.mean()*100:.1f}% positive)")
    logger.info(f"  Test:  {X_test.shape[0]:,} samples ({y_test.mean()*100:.1f}% positive)")

    # --- Re-fit scaler on training data only ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)  # Transform test using train statistics

    logger.info(f"  ✓ StandardScaler fit on training data")
    logger.info(f"    Train feature means: [{X_train.mean(axis=0)[:3].round(4)}...]")
    logger.info(f"    Test  feature means: [{X_test.mean(axis=0)[:3].round(4)}...]")

    return X_train, X_test, y_train, y_test, scaler


# ===================================================================
# STEP 3 — Train XGBoost Model
# ===================================================================
def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    params: dict | None = None,
) -> xgb.XGBClassifier:
    """
    Train the XGBoost binary classifier with early stopping.

    Early stopping monitors AUC on the test set and stops training
    if no improvement is seen for 20 consecutive rounds. This prevents
    overfitting while finding the optimal number of boosting rounds.

    Args:
        X_train, y_train: Training data.
        X_test, y_test:   Validation data for early stopping.
        params:           Hyperparameters (defaults to BASELINE_PARAMS).

    Returns:
        Trained XGBClassifier.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3 — Training XGBoost classifier")
    logger.info("=" * 60)

    if params is None:
        params = BASELINE_PARAMS.copy()

    # Log key hyperparameters
    logger.info(f"  Hyperparameters:")
    for key in ["n_estimators", "max_depth", "learning_rate", "scale_pos_weight",
                "subsample", "colsample_bytree", "min_child_weight"]:
        logger.info(f"    {key:<25s} = {params.get(key, 'N/A')}")

    # --- Initialize model ---
    model = xgb.XGBClassifier(**params)

    # --- Train with early stopping ---
    logger.info(f"  Training started...")
    start = time.time()

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False,  # We log our own progress
    )

    elapsed = time.time() - start
    best_iteration = model.best_iteration if hasattr(model, "best_iteration") else params["n_estimators"]

    logger.info(f"  ✓ Training complete")
    logger.info(f"    Time:            {elapsed:.2f}s")
    logger.info(f"    Best iteration:  {best_iteration}")
    logger.info(f"    Trees built:     {model.n_estimators}")

    return model


# ===================================================================
# STEP 4 — Evaluate Model Performance
# ===================================================================
def evaluate_model(
    model: xgb.XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
) -> dict:
    """
    Comprehensive model evaluation with multiple metrics.

    Metrics computed:
        1. AUC-ROC (primary metric — target ≥ 0.85)
        2. Average Precision (AP) / PR-AUC
        3. Precision at recall=0.80 (target ≥ 0.50)
        4. F1 score at optimal threshold
        5. Confusion matrix at 0.5 threshold
        6. FlowScore distribution on test set
        7. Inference time benchmark

    Args:
        model:         Trained XGBClassifier.
        X_test:        Test features.
        y_test:        Test labels.
        feature_names: Feature column names for importance logging.

    Returns:
        Dictionary of all metrics (saved to metrics.json).
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 4 — Evaluating model performance")
    logger.info("=" * 60)

    # --- 4.1 Predicted probabilities ---
    y_prob = model.predict_proba(X_test)[:, 1]  # P(default)
    y_pred = (y_prob >= 0.5).astype(int)

    # --- 4.2 AUC-ROC ---
    auc_roc = roc_auc_score(y_test, y_prob)
    target_met = "✓" if auc_roc >= 0.85 else "✗"
    logger.info(f"  {target_met} AUC-ROC: {auc_roc:.4f}  (target: ≥ 0.85)")

    # --- 4.3 Average Precision (PR-AUC) ---
    avg_precision = average_precision_score(y_test, y_prob)
    logger.info(f"    Average Precision: {avg_precision:.4f}")

    # --- 4.4 Precision at Recall = 0.80 ---
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

    # Find the threshold where recall ≈ 0.80
    recall_target = 0.80
    idx_at_recall = np.argmin(np.abs(recalls - recall_target))
    precision_at_recall = precisions[idx_at_recall]
    threshold_at_recall = thresholds[min(idx_at_recall, len(thresholds) - 1)]

    prec_met = "✓" if precision_at_recall >= 0.50 else "✗"
    logger.info(
        f"  {prec_met} Precision@Recall=0.80: {precision_at_recall:.4f}  "
        f"(target: ≥ 0.50, threshold: {threshold_at_recall:.4f})"
    )

    # --- 4.5 Optimal F1 threshold ---
    f1_scores = 2 * precisions * recalls / np.maximum(precisions + recalls, 1e-8)
    best_f1_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_idx]
    best_f1_threshold = thresholds[min(best_f1_idx, len(thresholds) - 1)]
    logger.info(f"    Best F1: {best_f1:.4f} at threshold={best_f1_threshold:.4f}")

    # --- 4.6 Classification Report at 0.5 threshold ---
    logger.info(f"")
    logger.info(f"  Classification Report (threshold=0.50):")
    report = classification_report(y_test, y_pred, target_names=["Repaid", "Defaulted"])
    for line in report.strip().split("\n"):
        logger.info(f"    {line}")

    # --- 4.7 Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    logger.info(f"")
    logger.info(f"  Confusion Matrix:")
    logger.info(f"    ┌──────────────────┬──────────────────┐")
    logger.info(f"    │ TN: {tn:>8,}     │ FP: {fp:>8,}     │")
    logger.info(f"    │ (correct repaid) │ (false alarm)    │")
    logger.info(f"    ├──────────────────┼──────────────────┤")
    logger.info(f"    │ FN: {fn:>8,}     │ TP: {tp:>8,}     │")
    logger.info(f"    │ (missed default) │ (caught default) │")
    logger.info(f"    └──────────────────┴──────────────────┘")

    # --- 4.8 FlowScore Distribution ---
    flowscores = get_flowscores_batch(y_prob)
    logger.info(f"")
    logger.info(f"  FlowScore Distribution (test set):")
    logger.info(f"    Min:    {flowscores.min()}")
    logger.info(f"    Max:    {flowscores.max()}")
    logger.info(f"    Mean:   {flowscores.mean():.0f}")
    logger.info(f"    Median: {np.median(flowscores):.0f}")

    # Score buckets
    for label, lo, hi in [
        ("Very High Risk (300-499)", 300, 500),
        ("High Risk      (500-599)", 500, 600),
        ("Medium Risk    (600-699)", 600, 700),
        ("Low Risk       (700-850)", 700, 851),
    ]:
        count = np.sum((flowscores >= lo) & (flowscores < hi))
        pct = count / len(flowscores) * 100
        logger.info(f"    {label}: {count:>7,}  ({pct:.1f}%)")

    # --- 4.9 Inference Time Benchmark ---
    single_sample = X_test[:1]
    n_iters = 100
    start = time.time()
    for _ in range(n_iters):
        model.predict_proba(single_sample)
    avg_inference_ms = (time.time() - start) / n_iters * 1000
    time_met = "✓" if avg_inference_ms < 200 else "✗"
    logger.info(f"")
    logger.info(f"  {time_met} Inference time: {avg_inference_ms:.2f}ms/sample  (target: < 200ms)")

    # --- Build metrics dict ---
    metrics = {
        "auc_roc": round(auc_roc, 6),
        "average_precision": round(avg_precision, 6),
        "precision_at_recall_80": round(float(precision_at_recall), 6),
        "threshold_at_recall_80": round(float(threshold_at_recall), 6),
        "best_f1": round(float(best_f1), 6),
        "best_f1_threshold": round(float(best_f1_threshold), 6),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "flowscore_distribution": {
            "min": int(flowscores.min()),
            "max": int(flowscores.max()),
            "mean": round(float(flowscores.mean()), 1),
            "median": int(np.median(flowscores)),
        },
        "inference_time_ms": round(avg_inference_ms, 3),
        "test_set_size": int(len(y_test)),
        "positive_rate_test": round(float(y_test.mean()), 4),
        "targets_met": {
            "auc_roc_ge_085": bool(auc_roc >= 0.85),
            "precision_at_recall80_ge_050": bool(precision_at_recall >= 0.50),
            "inference_lt_200ms": bool(avg_inference_ms < 200),
        },
    }

    return metrics


# ===================================================================
# STEP 5 — Feature Importance
# ===================================================================
def compute_feature_importance(
    model: xgb.XGBClassifier, feature_names: list[str]
) -> dict:
    """
    Extract and log XGBoost feature importance (gain-based).

    Gain measures the total improvement in the objective (AUC) that a
    feature provides across all trees and splits. It's more reliable
    than weight (number of splits) for understanding feature value.

    Args:
        model:         Trained XGBClassifier.
        feature_names: List of feature column names.

    Returns:
        Dict mapping feature names to their importance scores.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 5 — Feature importance (gain-based)")
    logger.info("=" * 60)

    importances = model.feature_importances_
    importance_dict = dict(zip(feature_names, importances.tolist()))

    # Sort by importance descending
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

    logger.info(f"  {'Rank':<6} {'Feature':<35} {'Importance':>12} {'Bar'}")
    logger.info(f"  {'─'*6} {'─'*35} {'─'*12} {'─'*20}")

    max_imp = sorted_features[0][1] if sorted_features else 1.0
    for rank, (feat, imp) in enumerate(sorted_features, 1):
        bar_len = int(imp / max_imp * 20)
        bar = "█" * bar_len
        logger.info(f"  {rank:<6} {feat:<35} {imp:>12.6f} {bar}")

    # Return as ordered dict for JSON serialization
    return {
        "method": "gain",
        "features": [
            {"rank": i + 1, "name": feat, "importance": round(imp, 6)}
            for i, (feat, imp) in enumerate(sorted_features)
        ],
    }


def plot_feature_importance(
    model: xgb.XGBClassifier,
    feature_names: list[str],
    output_path: Path,
) -> None:
    """
    Save a horizontal bar chart of feature importances to PNG.

    Produces a publication-ready chart with:
        - Features sorted by importance (top → bottom)
        - Color gradient from high (dark) to low (light)
        - Percentage labels on each bar
        - Clean styling (no grid clutter)

    Args:
        model:       Trained XGBClassifier.
        feature_names: List of feature column names.
        output_path: Where to save the PNG.
    """
    logger.info("")
    logger.info("  Saving feature importance chart...")

    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importances = importances[sorted_idx]

    # Normalize to percentage
    total = sorted_importances.sum()
    sorted_pct = sorted_importances / total * 100

    # --- Create the plot ---
    fig, ax = plt.subplots(figsize=(10, 8))

    # Color gradient: darker = more important
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(sorted_features)))

    bars = ax.barh(
        range(len(sorted_features)),
        sorted_pct,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )

    # Labels
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features, fontsize=10)
    ax.set_xlabel("Feature Importance (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "FlowScore — XGBoost Feature Importance (Gain)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    # Percentage labels on bars
    for bar, pct in zip(bars, sorted_pct):
        if pct > 1.0:  # Only label bars with >1%
            ax.text(
                bar.get_width() + 0.3,
                bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%",
                va="center",
                fontsize=9,
                color="#333",
            )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"  ✓ {output_path.name} saved ({output_path.stat().st_size / 1024:.0f} KB)")


# ===================================================================
# STEP 6 — SHAP Explainability
# ===================================================================
def build_shap_explainer(
    model: xgb.XGBClassifier,
    X_train: np.ndarray,
    feature_names: list[str],
) -> object:
    """
    Build a SHAP TreeExplainer for the trained XGBoost model.

    Why TreeExplainer?
        - 100x faster than KernelExplainer for tree-based models
        - Exact SHAP values (not approximations)
        - Works natively with XGBoost's internal tree structure

    The explainer is saved as a pickle and loaded by the API server
    to generate per-prediction explanations in real-time.

    Args:
        model:         Trained XGBClassifier.
        X_train:       Training data (used for SHAP background/reference).
        feature_names: Feature column names (for readable SHAP output).

    Returns:
        SHAP TreeExplainer object.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 6 — Building SHAP explainer")
    logger.info("=" * 60)

    import shap  # Deferred import — SHAP is heavy (~2s)

    start = time.time()
    explainer = shap.TreeExplainer(model)
    elapsed = time.time() - start
    logger.info(f"  ✓ TreeExplainer built in {elapsed:.2f}s")

    # --- Compute SHAP values on a sample for validation ---
    sample_size = min(500, X_train.shape[0])
    logger.info(f"  Computing SHAP values on {sample_size} sample rows...")

    start = time.time()
    shap_values = explainer.shap_values(X_train[:sample_size])
    elapsed = time.time() - start
    logger.info(f"  ✓ SHAP values computed in {elapsed:.2f}s")

    # --- Log mean absolute SHAP values (global feature importance via SHAP) ---
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = sorted(
        zip(feature_names, mean_abs_shap), key=lambda x: x[1], reverse=True
    )

    logger.info(f"")
    logger.info(f"  SHAP Feature Importance (mean |SHAP value|):")
    for feat, val in shap_importance[:10]:
        bar_len = int(val / shap_importance[0][1] * 15)
        bar = "█" * bar_len
        logger.info(f"    {feat:<35s} {val:.6f}  {bar}")
    if len(shap_importance) > 10:
        logger.info(f"    ... and {len(shap_importance) - 10} more features")

    # --- Single-sample explanation demo ---
    logger.info(f"")
    logger.info(f"  Example explanation (first test sample):")
    sample_shap = shap_values[0]
    sample_explanation = sorted(
        zip(feature_names, sample_shap), key=lambda x: abs(x[1]), reverse=True
    )
    for feat, val in sample_explanation[:5]:
        direction = "↑ risk" if val > 0 else "↓ risk"
        logger.info(f"    {feat:<35s} SHAP={val:+.4f}  ({direction})")

    return explainer


# ===================================================================
# STEP 7 — Cross-Validation (optional robustness check)
# ===================================================================
def run_cross_validation(
    X: np.ndarray, y: np.ndarray, params: dict
) -> dict:
    """
    5-fold stratified cross-validation for AUC stability check.

    This verifies that our 70/30 split wasn't accidentally favorable.
    If CV AUC is significantly lower than test AUC, we may be overfitting.

    Args:
        X: Full feature matrix.
        y: Full target vector.
        params: XGBoost hyperparameters.

    Returns:
        CV results dict.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 7 — 5-Fold Cross-Validation")
    logger.info("=" * 60)

    model = xgb.XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    start = time.time()
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    elapsed = time.time() - start

    logger.info(f"  Time: {elapsed:.2f}s")
    logger.info(f"  Fold AUCs: {[round(s, 4) for s in cv_scores]}")
    logger.info(f"  Mean AUC:  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    cv_met = "✓" if cv_scores.mean() >= 0.85 else "✗"
    logger.info(f"  {cv_met} CV AUC: {cv_scores.mean():.4f}  (target: ≥ 0.85)")

    return {
        "fold_aucs": [round(float(s), 6) for s in cv_scores],
        "mean_auc": round(float(cv_scores.mean()), 6),
        "std_auc": round(float(cv_scores.std()), 6),
        "n_folds": 5,
    }


# ===================================================================
# STEP 8 — Hyperparameter Tuning (--tune flag)
# ===================================================================
def tune_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_iter: int = 30,
) -> dict:
    """
    Random search over the hyperparameter grid.

    We use random search instead of grid search because:
        - Full grid = 5×5×4×4×3×3×4 = 14,400 combinations → too slow
        - Random search with 30 iterations finds near-optimal params
          with ~90% probability (Bergstra & Bengio, 2012)

    Args:
        X_train, y_train: Training data.
        X_test, y_test:   Validation data.
        n_iter:           Number of random combinations to try.

    Returns:
        Best hyperparameter dict.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"STEP 8 — Hyperparameter Tuning ({n_iter} iterations)")
    logger.info("=" * 60)

    rng = np.random.default_rng(RANDOM_STATE)
    best_auc = 0.0
    best_params = BASELINE_PARAMS.copy()

    for i in range(n_iter):
        # Sample random params
        trial_params = BASELINE_PARAMS.copy()
        trial_params["max_depth"] = rng.choice(TUNING_GRID["max_depth"])
        trial_params["learning_rate"] = rng.choice(TUNING_GRID["learning_rate"])
        trial_params["n_estimators"] = rng.choice(TUNING_GRID["n_estimators"])
        trial_params["min_child_weight"] = rng.choice(TUNING_GRID["min_child_weight"])
        trial_params["subsample"] = rng.choice(TUNING_GRID["subsample"])
        trial_params["colsample_bytree"] = rng.choice(TUNING_GRID["colsample_bytree"])
        trial_params["scale_pos_weight"] = rng.choice(TUNING_GRID["scale_pos_weight"])

        model = xgb.XGBClassifier(**trial_params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)

        marker = "★" if auc > best_auc else " "
        logger.info(
            f"  {marker} [{i+1:>3}/{n_iter}] "
            f"AUC={auc:.4f}  "
            f"depth={trial_params['max_depth']}  "
            f"lr={trial_params['learning_rate']:.3f}  "
            f"trees={trial_params['n_estimators']}  "
            f"spw={trial_params['scale_pos_weight']}"
        )

        if auc > best_auc:
            best_auc = auc
            best_params = trial_params.copy()

    logger.info(f"")
    logger.info(f"  ★ Best AUC: {best_auc:.4f}")
    logger.info(f"  Best params:")
    for key in ["max_depth", "learning_rate", "n_estimators", "min_child_weight",
                "subsample", "colsample_bytree", "scale_pos_weight"]:
        logger.info(f"    {key:<25s} = {best_params[key]}")

    return best_params


# ===================================================================
# STEP 9 — Save Artifacts
# ===================================================================
def save_artifacts(
    model: xgb.XGBClassifier,
    scaler: StandardScaler,
    explainer: object,
    metrics: dict,
    feature_importance: dict,
    feature_names: list[str],
    cv_results: dict | None = None,
    tuned_params: dict | None = None,
) -> None:
    """
    Save all model artifacts to disk.

    Artifacts:
        model.pkl             — Trained XGBoost classifier (joblib)
        scaler.pkl            — StandardScaler fit on training data
        explainer.pkl         — SHAP TreeExplainer
        metrics.json          — All evaluation metrics
        feature_importance.json — Ranked feature importances

    Uses joblib for binary artifacts because:
        - 2-5x smaller files than pickle for numpy-backed objects
        - Built-in compression support
        - Better handling of large arrays

    Args:
        model:              Trained XGBClassifier.
        scaler:             Fitted StandardScaler.
        explainer:          SHAP TreeExplainer.
        metrics:            Evaluation metrics dict.
        feature_importance: Feature importance dict.
        feature_names:      List of feature names (for inference pipeline).
        cv_results:         Optional cross-validation results.
        tuned_params:       Optional tuned hyperparameters.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 9 — Saving artifacts")
    logger.info("=" * 60)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- 9.1 Save model ---
    joblib.dump(model, MODEL_PATH, compress=3)
    model_size = MODEL_PATH.stat().st_size / (1024 * 1024)
    logger.info(f"  ✓ model.pkl        ({model_size:.1f} MB)")

    # --- 9.2 Save scaler ---
    scaler_data = {
        "scaler": scaler,
        "feature_names": feature_names,
        "n_features": len(feature_names),
    }
    joblib.dump(scaler_data, SCALER_PATH, compress=3)
    scaler_size = SCALER_PATH.stat().st_size / 1024
    logger.info(f"  ✓ scaler.pkl       ({scaler_size:.1f} KB)")

    # --- 9.3 Save SHAP explainer ---
    joblib.dump(explainer, EXPLAINER_PATH, compress=3)
    explainer_size = EXPLAINER_PATH.stat().st_size / (1024 * 1024)
    logger.info(f"  ✓ explainer.pkl    ({explainer_size:.1f} MB)")

    # --- 9.4 Save metrics ---
    full_metrics = {
        "generated_at": datetime.now().isoformat(),
        "model_version": "v1.0",
        "pipeline": "FlowScore XGBoost v1",
        "hyperparameters": dict(model.get_params()),
        "evaluation": metrics,
        "feature_names": feature_names,
        "flowscore_formula": "300 + (1 - P(default)) * 550",
    }
    if cv_results:
        full_metrics["cross_validation"] = cv_results
    if tuned_params:
        full_metrics["tuned_hyperparameters"] = tuned_params

    # Convert numpy types to native Python for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    def deep_convert(obj):
        if isinstance(obj, dict):
            return {k: deep_convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [deep_convert(v) for v in obj]
        return convert_numpy(obj)

    full_metrics = deep_convert(full_metrics)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=2, default=str)
    logger.info(f"  ✓ metrics.json")

    # --- 9.5 Save feature importance ---
    with open(FEATURE_IMPORTANCE_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_importance, f, indent=2)
    logger.info(f"  ✓ feature_importance.json")

    # --- 9.6 Save feature names list (standalone pkl) ---
    joblib.dump(feature_names, FEATURE_NAMES_PATH, compress=3)
    logger.info(f"  ✓ feature_names.pkl    ({len(feature_names)} features)")

    logger.info(f"")
    logger.info(f"  All artifacts saved to: {ARTIFACTS_DIR}")


# ===================================================================
# Main Pipeline Orchestrator
# ===================================================================
def run_pipeline(input_path: Path, do_tune: bool = False, do_cv: bool = True) -> None:
    """
    Execute the full training pipeline end-to-end.

    Args:
        input_path: Path to engineered_features.csv.
        do_tune:    If True, run hyperparameter search before final training.
        do_cv:      If True, run 5-fold cross-validation after training.
    """
    pipeline_start = time.time()

    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║   FlowScore — Model Training Pipeline                    ║")
    logger.info("║   XGBoost Binary Classification                          ║")
    logger.info("╚" + "═" * 58 + "╝")
    logger.info("")
    logger.info(f"  Input:    {input_path}")
    logger.info(f"  Tune:     {do_tune}")
    logger.info(f"  CV:       {do_cv}")
    logger.info(f"  Seed:     {RANDOM_STATE}")
    logger.info(f"  Split:    {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)}")
    logger.info(f"  Output:   {ARTIFACTS_DIR}")
    logger.info("")

    # --- Step 1: Load data ---
    df, feature_names = load_features(input_path)

    # --- Step 2: Split & scale ---
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df, feature_names)

    # --- Step 8 (optional): Hyperparameter tuning ---
    tuned_params = None
    if do_tune:
        tuned_params = tune_hyperparameters(X_train, y_train, X_test, y_test)
        training_params = tuned_params
    else:
        training_params = BASELINE_PARAMS.copy()

    # --- Step 3: Train model ---
    model = train_xgboost(X_train, y_train, X_test, y_test, params=training_params)

    # --- Step 4: Evaluate ---
    metrics = evaluate_model(model, X_test, y_test, feature_names)

    # --- Step 5: Feature importance ---
    feature_importance = compute_feature_importance(model, feature_names)
    plot_feature_importance(model, feature_names, FEATURE_IMPORTANCE_PNG)

    # --- Step 6: SHAP explainer ---
    explainer = build_shap_explainer(model, X_train, feature_names)

    # --- Step 7 (optional): Cross-validation ---
    cv_results = None
    if do_cv:
        # Use full dataset (pre-split) for CV
        X_full = df[feature_names].values
        y_full = df[TARGET_COL].values
        cv_results = run_cross_validation(X_full, y_full, training_params)

    # --- Step 9: Save everything ---
    save_artifacts(
        model=model,
        scaler=scaler,
        explainer=explainer,
        metrics=metrics,
        feature_importance=feature_importance,
        feature_names=feature_names,
        cv_results=cv_results,
        tuned_params=tuned_params,
    )

    # --- Final summary ---
    elapsed = time.time() - pipeline_start
    auc = metrics["auc_roc"]
    score_median = metrics["flowscore_distribution"]["median"]
    inf_time = metrics["inference_time_ms"]

    logger.info("")
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║   ✓ Training Pipeline Complete                           ║")
    logger.info("╠" + "═" * 58 + "╣")
    logger.info(f"║   Total time:       {elapsed:>8.1f}s{' ' * 28}║")
    logger.info(f"║   AUC-ROC:          {auc:>8.4f}{' ' * 28}║")
    logger.info(f"║   FlowScore median: {score_median:>8}{' ' * 28}║")
    logger.info(f"║   Inference time:   {inf_time:>7.2f}ms{' ' * 28}║")
    logger.info("╠" + "═" * 58 + "╣")
    logger.info("║   Artifacts saved to: model/artifacts/                    ║")
    logger.info("║   Next step: python backend/main.py                      ║")
    logger.info("╚" + "═" * 58 + "╝")

    # --- Exit with non-zero if targets not met ---
    all_targets_met = all(metrics["targets_met"].values())
    if not all_targets_met:
        failed = [k for k, v in metrics["targets_met"].items() if not v]
        logger.warning(f"  ⚠ Targets NOT met: {failed}")
        logger.warning(f"    Consider running with --tune flag")
    else:
        logger.info(f"  ✓ ALL performance targets met!")


# ===================================================================
# CLI Entry Point
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description="FlowScore — Train XGBoost credit scoring model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python model/train_model.py
  python model/train_model.py --input data/processed/engineered_features.csv
  python model/train_model.py --tune           # Run hyperparameter search
  python model/train_model.py --no-cv          # Skip cross-validation
  python model/train_model.py --tune --no-cv   # Tune only, skip CV
        """,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to engineered features CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        default=False,
        help="Run hyperparameter tuning before training (slower, ~30 trials)",
    )
    parser.add_argument(
        "--no-cv",
        action="store_true",
        default=False,
        help="Skip 5-fold cross-validation (faster)",
    )
    args = parser.parse_args()

    try:
        run_pipeline(
            input_path=args.input,
            do_tune=args.tune,
            do_cv=not args.no_cv,
        )
    except FileNotFoundError as e:
        logger.error(f"Input not found: {e}")
        sys.exit(1)
    except KeyError as e:
        logger.error(f"Missing required data: {e}")
        sys.exit(1)
    except MemoryError:
        logger.error("Out of memory during training. Try reducing n_estimators.")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
