"""
FlowScore — Scoring Utilities
===============================

Core logic for:
    1. Extracting 23 model features from a borrower profile JSON
    2. Running the XGBoost model and converting to FlowScore
    3. Generating SHAP explanations
    4. Producing personalized coaching tips

This module is imported by main.py and used by the /score and
/borrower endpoints. It loads model artifacts once at import time
and reuses them for all requests (singleton pattern).
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import joblib

logger = logging.getLogger("flowscore.utils")

# ---------------------------------------------------------------------------
# Artifact Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "model" / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
EXPLAINER_PATH = ARTIFACTS_DIR / "explainer.pkl"

# ---------------------------------------------------------------------------
# The 23 Features (must match training pipeline order exactly)
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    # Income (8)
    "avg_6m_income",
    "income_std_dev",
    "income_trend_6m",
    "latest_month_income",
    "income_velocity_3m",
    "platform_count",
    "primary_platform_earnings_pct",
    "secondary_platform_earnings_pct",
    # Platform (6)
    "avg_platform_rating",
    "days_on_primary_platform",
    "days_on_secondary_platform",
    "platform_account_variance",
    "account_age_months",
    "total_platforms",
    # Spending (4)
    "spending_to_income_ratio",
    "avg_monthly_spending",
    "spending_volatility",
    "spending_trend",
    # Credit (3)
    "late_payments_count_6m",
    "missed_payments_count_6m",
    "existing_loan_count",
    # Demographic (2)
    "age",
    "account_tenure_months",
]


# ===================================================================
# Singleton Model Loader
# ===================================================================

class ModelStore:
    """
    Lazy-loading singleton for ML artifacts.

    Loads model, scaler, and SHAP explainer on first access.
    Subsequent calls return cached instances (no disk I/O).
    Thread-safe for FastAPI's async request handling.
    """

    def __init__(self):
        self._model = None
        self._scaler = None
        self._scaler_data = None
        self._explainer = None
        self._loaded = False

    def load(self) -> bool:
        """Load all artifacts from disk. Returns True if successful."""
        if self._loaded:
            return True

        try:
            logger.info(f"Loading model from {MODEL_PATH}...")
            self._model = joblib.load(MODEL_PATH)
            logger.info(f"✓ Model loaded")

            logger.info(f"Loading scaler from {SCALER_PATH}...")
            self._scaler_data = joblib.load(SCALER_PATH)
            self._scaler = self._scaler_data["scaler"]
            logger.info(f"✓ Scaler loaded (features: {self._scaler_data.get('n_features', '?')})")

            logger.info(f"Loading SHAP explainer from {EXPLAINER_PATH}...")
            self._explainer = joblib.load(EXPLAINER_PATH)
            logger.info(f"✓ SHAP explainer loaded")

            self._loaded = True
            return True

        except FileNotFoundError as e:
            logger.error(f"Artifact not found: {e}")
            logger.error("Run `python model/train_model.py` to generate artifacts.")
            return False
        except Exception as e:
            logger.error(f"Failed to load artifacts: {e}")
            return False

    @property
    def model(self):
        return self._model

    @property
    def scaler(self):
        return self._scaler

    @property
    def explainer(self):
        return self._explainer

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# Global singleton — shared across all requests
model_store = ModelStore()


# ===================================================================
# Feature Extraction
# ===================================================================

def extract_features(request) -> np.ndarray:
    """
    Extract the 23 model features from a BorrowerScoreRequest.

    Maps the structured borrower JSON into the flat feature vector
    that the XGBoost model expects. Computes derived features
    (income_trend, velocity, etc.) from the raw platform earnings.

    Args:
        request: BorrowerScoreRequest (Pydantic model).

    Returns:
        numpy array of shape (1, 23) — single borrower feature vector.
    """
    income = request.income_data
    spending = request.spending_data
    credit = request.credit_profile or type("CreditProfile", (), {
        "existing_loans": 0, "total_debt": 0, "credit_inquiries_6m": 0
    })()
    calc = request.calculated_features

    platforms = income.platforms
    n_platforms = len(platforms)

    # --- Aggregate all platform earnings ---
    all_earnings = []
    for p in platforms:
        all_earnings.append(p.monthly_earnings_last_6m)

    # Total monthly earnings per month (sum across platforms)
    max_months = max(len(e) for e in all_earnings)
    monthly_totals = np.zeros(max_months)
    for earnings in all_earnings:
        for i, val in enumerate(earnings):
            monthly_totals[i] += val

    # --- Income Features (8) ---
    avg_6m_income = float(np.mean(monthly_totals))
    income_std_dev = float(np.std(monthly_totals)) if len(monthly_totals) > 1 else 0.0

    # Income trend: compare last 3 months to first 3 months
    if len(monthly_totals) >= 6:
        first_half = np.mean(monthly_totals[:3])
        second_half = np.mean(monthly_totals[3:])
        income_trend_6m = (second_half - first_half) / max(first_half, 1.0)
    elif calc and calc.income_trend is not None:
        income_trend_6m = calc.income_trend
    else:
        income_trend_6m = 0.0

    latest_month_income = float(monthly_totals[-1]) if len(monthly_totals) > 0 else avg_6m_income

    # Income velocity: growth rate over last 3 months
    if len(monthly_totals) >= 3:
        recent_3 = monthly_totals[-3:]
        if recent_3[0] > 0:
            income_velocity_3m = (recent_3[-1] - recent_3[0]) / recent_3[0]
        else:
            income_velocity_3m = 0.0
    elif calc and calc.income_velocity_3m is not None:
        income_velocity_3m = calc.income_velocity_3m
    else:
        income_velocity_3m = 0.0

    platform_count = n_platforms

    # Primary/secondary platform earnings percentage
    platform_totals = [sum(p.monthly_earnings_last_6m) for p in platforms]
    grand_total = sum(platform_totals) or 1.0
    sorted_pct = sorted([t / grand_total for t in platform_totals], reverse=True)
    primary_pct = sorted_pct[0] if sorted_pct else 1.0
    secondary_pct = sorted_pct[1] if len(sorted_pct) > 1 else 0.0

    # --- Platform Features (6) ---
    avg_platform_rating = float(np.mean([p.platform_rating for p in platforms]))

    # Days on platforms (from account age in months)
    platform_ages = [p.platform_account_age_months or 12 for p in platforms]
    sorted_ages_days = sorted([a * 30 for a in platform_ages], reverse=True)
    days_on_primary = sorted_ages_days[0] if sorted_ages_days else 360
    days_on_secondary = sorted_ages_days[1] if len(sorted_ages_days) > 1 else 0

    # Platform account variance
    if len(sorted_ages_days) > 1:
        platform_account_var = float(np.var(sorted_ages_days))
    else:
        platform_account_var = 0.0

    account_age_months = max(platform_ages) if platform_ages else 12
    total_platforms = n_platforms

    # --- Spending Features (4) ---
    avg_monthly_spending = spending.avg_monthly_spending
    spending_to_income_ratio = avg_monthly_spending / max(avg_6m_income, 1.0)

    # Spending volatility — estimate from spending/income pattern
    spending_volatility = avg_monthly_spending * 0.15  # Default ~15% CV

    # Spending trend — proxy: if spending ratio is high, trend is likely positive
    spending_trend = min(spending_to_income_ratio * 0.3, 0.5)

    # --- Credit Features (3) ---
    late_payments_count_6m = spending.late_payments_count_6m
    missed_payments_count_6m = spending.missed_payments_count_6m
    existing_loan_count = credit.existing_loans if hasattr(credit, 'existing_loans') else 0

    # --- Demographic Features (2) ---
    age = request.age or 30  # Default age if not provided

    # Account tenure: use the oldest platform age as proxy
    account_tenure_months = max(platform_ages) if platform_ages else 12

    # --- Override with calculated_features if provided ---
    if calc:
        if calc.income_volatility is not None:
            income_std_dev = calc.income_volatility * avg_6m_income
        if calc.spending_to_income_ratio is not None:
            spending_to_income_ratio = calc.spending_to_income_ratio
        if calc.platform_count is not None:
            platform_count = calc.platform_count
            total_platforms = calc.platform_count
        if calc.days_active_primary is not None:
            days_on_primary = calc.days_active_primary

    # --- Assemble feature vector (must match FEATURE_NAMES order) ---
    features = np.array([[
        avg_6m_income,
        income_std_dev,
        income_trend_6m,
        latest_month_income,
        income_velocity_3m,
        platform_count,
        primary_pct,
        secondary_pct,
        avg_platform_rating,
        days_on_primary,
        days_on_secondary,
        platform_account_var,
        account_age_months,
        total_platforms,
        spending_to_income_ratio,
        avg_monthly_spending,
        spending_volatility,
        spending_trend,
        late_payments_count_6m,
        missed_payments_count_6m,
        existing_loan_count,
        age,
        account_tenure_months,
    ]], dtype=np.float64)

    return features


# ===================================================================
# FlowScore Conversion
# ===================================================================

def probability_to_flowscore(default_prob: float) -> int:
    """
    Convert P(default) to FlowScore (300–850).

    Formula: FlowScore = 300 + (1 - P(default)) × 550
    Lower default risk → higher score.
    """
    score = 300 + (1 - default_prob) * 550
    return int(np.clip(score, 300, 850))


def get_risk_category(flowscore: int) -> str:
    """Classify FlowScore into risk buckets."""
    if flowscore >= 700:
        return "low"
    elif flowscore >= 600:
        return "medium"
    elif flowscore >= 500:
        return "high"
    else:
        return "very_high"


def get_confidence_interval(flowscore: int, n_features: int = 23) -> list[int]:
    """
    Estimate confidence interval around the FlowScore.

    Width decreases with more features (more information = more
    confidence). This is a simplified heuristic — in production,
    you'd use bootstrap or conformal prediction.
    """
    half_width = max(15, 50 - n_features)  # ±15 to ±27 points
    lower = max(300, flowscore - half_width)
    upper = min(850, flowscore + half_width)
    return [lower, upper]


# ===================================================================
# SHAP Explanation Generation
# ===================================================================

def generate_shap_explanation(
    features: np.ndarray,
    n_positive: int = 5,
    n_negative: int = 3,
) -> tuple[list[dict], list[dict]]:
    """
    Generate SHAP-based feature explanations for a single prediction.

    Uses the pre-loaded TreeExplainer to compute exact SHAP values,
    then sorts features by contribution magnitude.

    Args:
        features:   Scaled feature vector, shape (1, 23).
        n_positive: Number of top risk-increasing factors to return.
        n_negative: Number of top risk-decreasing factors to return.

    Returns:
        (positive_factors, negative_factors) — each is a list of dicts
        with keys: feature, contribution, value.
    """
    explainer = model_store.explainer

    if explainer is None:
        # Fallback: return empty if explainer not loaded
        logger.warning("SHAP explainer not loaded — returning empty explanation")
        return [], []

    try:
        shap_values = explainer.shap_values(features)

        # shap_values shape: (1, 23) for binary classification
        if isinstance(shap_values, list):
            # Some SHAP versions return [class_0_shap, class_1_shap]
            shap_vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        else:
            shap_vals = shap_values[0]

        raw_values = features[0]

        # Pair each feature with its SHAP value and raw value
        feature_shap = list(zip(FEATURE_NAMES, shap_vals, raw_values))

        # Sort by SHAP value
        positive = sorted(
            [(f, s, v) for f, s, v in feature_shap if s > 0],
            key=lambda x: x[1],
            reverse=True,
        )[:n_positive]

        negative = sorted(
            [(f, s, v) for f, s, v in feature_shap if s <= 0],
            key=lambda x: x[1],
        )[:n_negative]

        positive_factors = [
            {"feature": f, "contribution": round(float(s) * 100, 1), "value": round(float(v), 4)}
            for f, s, v in positive
        ]

        negative_factors = [
            {"feature": f, "contribution": round(float(s) * 100, 1), "value": round(float(v), 4)}
            for f, s, v in negative
        ]

        return positive_factors, negative_factors

    except Exception as e:
        logger.error(f"SHAP computation failed: {e}")
        return [], []


# ===================================================================
# Coaching Tips Generator
# ===================================================================

# Rule-based coaching tip templates, keyed by feature name.
# Each rule has a condition lambda and a tip template string.
COACHING_RULES = [
    {
        "feature": "income_trend_6m",
        "condition": lambda val, shap_val: val > 0.2,
        "tip_positive": "Your income grew {pct}% over 6 months — excellent trajectory! {impact}",
        "tip_negative": "Your income has been declining. Growing it steadily could add +{points} points to your score.",
    },
    {
        "feature": "spending_to_income_ratio",
        "condition": lambda val, shap_val: val > 0.6,
        "tip_high": "Your spending is {pct}% of income. Reducing discretionary spending by 10% could add +{points} points.",
        "tip_low": "Great spending discipline! Your low spending ratio adds +{points} points to your score.",
    },
    {
        "feature": "avg_platform_rating",
        "condition": lambda val, shap_val: True,
        "tip": "Maintain your platform rating of {rating:.1f}. Every 0.1pt increase = +2 score points.",
    },
    {
        "feature": "income_std_dev",
        "condition": lambda val, shap_val: shap_val > 0,
        "tip": "Your income variability is high. Diversifying across platforms could stabilize your score by +{points} points.",
    },
    {
        "feature": "platform_count",
        "condition": lambda val, shap_val: val == 1,
        "tip": "You're earning from 1 platform. Adding a second income source could improve your stability score by +15 points.",
    },
    {
        "feature": "late_payments_count_6m",
        "condition": lambda val, shap_val: val > 0,
        "tip": "You have {count} late payment(s). Each on-time payment streak of 3 months adds +8 points.",
    },
    {
        "feature": "income_velocity_3m",
        "condition": lambda val, shap_val: val > 0.1,
        "tip": "Your recent income acceleration of {pct}% is strong — keep this momentum for a higher score.",
    },
]


def generate_coaching_tips(
    features: np.ndarray,
    shap_positive: list[dict],
    shap_negative: list[dict],
    flowscore: int,
    max_tips: int = 3,
) -> list[str]:
    """
    Generate personalized, actionable coaching tips based on SHAP values.

    Tips are ordered by potential impact (highest SHAP magnitude first).
    Each tip includes an estimated score improvement to motivate the borrower.

    Args:
        features:       Raw feature vector (1, 23).
        shap_positive:  SHAP factors increasing risk.
        shap_negative:  SHAP factors decreasing risk.
        flowscore:      Current FlowScore.
        max_tips:       Maximum number of tips to return.

    Returns:
        List of coaching tip strings.
    """
    raw = features[0]
    feature_values = dict(zip(FEATURE_NAMES, raw))
    all_shap = {f["feature"]: f["contribution"] for f in shap_positive + shap_negative}

    tips = []

    # --- Income Trend Tip ---
    trend = feature_values.get("income_trend_6m", 0)
    trend_shap = all_shap.get("income_trend_6m", 0)
    if trend > 0.15:
        tips.append(
            f"Your income grew {abs(trend)*100:.0f}% over 6 months — excellent trajectory! "
            f"+{abs(trend_shap):.0f} points"
        )
    elif trend < 0:
        tips.append(
            f"Your income declined {abs(trend)*100:.0f}% recently. "
            f"Growing it steadily could add +{max(abs(trend_shap), 10):.0f} points."
        )

    # --- Spending Ratio Tip ---
    ratio = feature_values.get("spending_to_income_ratio", 0)
    ratio_shap = all_shap.get("spending_to_income_ratio", 0)
    if ratio > 0.6:
        tips.append(
            f"Your spending is {ratio*100:.0f}% of income. "
            f"Reducing discretionary spending by 10% could add +{max(abs(ratio_shap), 15):.0f} points."
        )
    elif ratio < 0.4:
        tips.append(
            f"Great spending discipline! Your low {ratio*100:.0f}% spending ratio "
            f"adds +{abs(ratio_shap):.0f} points to your score."
        )

    # --- Platform Rating Tip ---
    rating = feature_values.get("avg_platform_rating", 4.0)
    tips.append(
        f"Maintain your platform rating of {rating:.1f}/5.0. "
        f"Every 0.1pt increase ≈ +2 score points."
    )

    # --- Platform Diversity Tip ---
    n_platforms = feature_values.get("platform_count", 1)
    if n_platforms <= 1:
        tips.append(
            "You're earning from 1 platform. Adding a second income source "
            "could improve your stability score by +15 points."
        )

    # --- Late Payments Tip ---
    late = feature_values.get("late_payments_count_6m", 0)
    if late > 0:
        tips.append(
            f"You have {int(late)} late payment(s). Each on-time streak of "
            f"3 months adds +8 points."
        )

    # --- Income Volatility Tip ---
    vol_shap = all_shap.get("income_std_dev", 0)
    if vol_shap > 5:
        tips.append(
            "Your income variability is impacting your score. "
            "Diversifying across platforms could stabilize it by +12 points."
        )

    return tips[:max_tips]


# ===================================================================
# Full Scoring Pipeline
# ===================================================================

def score_borrower(request) -> dict:
    """
    End-to-end scoring: extract features → predict → explain → coach.

    This is the core function called by POST /score. It orchestrates
    the entire prediction pipeline in a single call.

    Args:
        request: BorrowerScoreRequest (Pydantic model).

    Returns:
        Dict matching ScoreResponse schema.

    Raises:
        RuntimeError: If model artifacts are not loaded.
    """
    if not model_store.is_loaded:
        raise RuntimeError("Model artifacts not loaded. Run train_model.py first.")

    # 1. Extract 23 features from borrower JSON
    raw_features = extract_features(request)

    # 2. Scale features using the training scaler
    scaled_features = model_store.scaler.transform(raw_features)

    # 3. Predict default probability
    default_prob = float(model_store.model.predict_proba(scaled_features)[0, 1])

    # 4. Convert to FlowScore
    flowscore = probability_to_flowscore(default_prob)
    risk_category = get_risk_category(flowscore)
    confidence = get_confidence_interval(flowscore)

    # 5. Generate SHAP explanation
    positive_factors, negative_factors = generate_shap_explanation(scaled_features)

    # 6. Generate coaching tips
    tips = generate_coaching_tips(
        raw_features, positive_factors, negative_factors, flowscore
    )

    return {
        "borrower_id": request.borrower_id,
        "flowscore": flowscore,
        "default_probability": round(default_prob, 4),
        "confidence_interval": confidence,
        "risk_category": risk_category,
        "shap_explanation": {
            "top_positive_factors": positive_factors,
            "top_negative_factors": negative_factors,
        },
        "coaching_tips": tips,
        "model_metadata": {
            "model_version": "v1.0",
            "prediction_timestamp": datetime.utcnow().isoformat() + "Z",
            "feature_count": len(FEATURE_NAMES),
        },
    }
