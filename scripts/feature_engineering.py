"""
FlowScore — Feature Engineering Pipeline
==========================================

Transforms the cleaned Home Credit dataset into 23 gig-worker credit
features used by the XGBoost scoring model.

The Problem:
    Home Credit has 100+ raw banking columns (AMT_INCOME_TOTAL, DAYS_EMPLOYED,
    EXT_SOURCE_*, etc.), but FlowScore scores gig workers on income velocity,
    platform diversity, spending discipline, etc. This script bridges the gap.

Strategy:
    1. DIRECT MAPPING — Where a Home Credit column directly represents a
       gig concept (e.g., AMT_INCOME_TOTAL → avg_6m_income)
    2. DERIVED FEATURES — Computed from existing columns (e.g., income
       volatility from credit bureau inquiry patterns)
    3. SYNTHETIC PROXIES — When no direct mapping exists, we generate
       realistic proxies using domain-informed transformations and
       controlled noise (seeded for reproducibility)

The 23 Output Features (grouped by category):
    ┌──────────────────────────────────────────────────────────────┐
    │ Income (8)    │ avg_6m_income, income_std_dev,              │
    │               │ income_trend_6m, latest_month_income,       │
    │               │ income_velocity_3m, platform_count,         │
    │               │ primary_platform_earnings_pct,              │
    │               │ secondary_platform_earnings_pct             │
    ├───────────────┼────────────────────────────────────────────-─┤
    │ Platform (6)  │ avg_platform_rating, days_on_primary,       │
    │               │ days_on_secondary, platform_account_var,    │
    │               │ account_age_months, total_platforms         │
    ├───────────────┼─────────────────────────────────────────────┤
    │ Spending (4)  │ spending_to_income_ratio,                   │
    │               │ avg_monthly_spending, spending_volatility,  │
    │               │ spending_trend                              │
    ├───────────────┼─────────────────────────────────────────────┤
    │ Credit (3)    │ late_payments_count_6m,                     │
    │               │ missed_payments_count_6m,                   │
    │               │ existing_loan_count                         │
    ├───────────────┼─────────────────────────────────────────────┤
    │ Demographic(2)│ age, account_tenure_months                  │
    └───────────────┴─────────────────────────────────────────────┘

Usage:
    python scripts/feature_engineering.py
    python scripts/feature_engineering.py --input data/processed/cleaned_data.csv
    python scripts/feature_engineering.py --seed 42

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
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_INPUT = PROCESSED_DATA_DIR / "cleaned_data.csv"
OUTPUT_FEATURES_CSV = PROCESSED_DATA_DIR / "engineered_features.csv"
OUTPUT_SCALER_META = PROCESSED_DATA_DIR / "scaler_metadata.json"
OUTPUT_FEATURE_MAP = PROCESSED_DATA_DIR / "feature_mapping.json"

# Target column preserved through the pipeline
TARGET_COL = "TARGET"

# Reproducibility seed — controls all synthetic proxy generation
DEFAULT_SEED = 42

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

log_filename = f"feature_eng_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / log_filename, mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("flowscore.feature_engineering")


# ===================================================================
# Feature Definition Registry
# ===================================================================
# Central definition of all 23 features with their category, source
# mapping, and description. This registry drives the entire pipeline.

FEATURE_REGISTRY = [
    # --- INCOME FEATURES (8) ---
    {
        "name": "avg_6m_income",
        "category": "income",
        "source": "AMT_INCOME_TOTAL",
        "method": "direct_scale",
        "description": "Average monthly income over last 6 months",
    },
    {
        "name": "income_std_dev",
        "category": "income",
        "source": "AMT_INCOME_TOTAL",
        "method": "synthetic_volatility",
        "description": "Standard deviation of monthly income (income consistency)",
    },
    {
        "name": "income_trend_6m",
        "category": "income",
        "source": ["AMT_INCOME_TOTAL", "EXT_SOURCE_2"],
        "method": "derived_trend",
        "description": "Income growth rate over 6 months (positive = growing)",
    },
    {
        "name": "latest_month_income",
        "category": "income",
        "source": "AMT_INCOME_TOTAL",
        "method": "derived_latest",
        "description": "Most recent month's income (current earning power)",
    },
    {
        "name": "income_velocity_3m",
        "category": "income",
        "source": ["AMT_INCOME_TOTAL", "EXT_SOURCE_3"],
        "method": "derived_velocity",
        "description": "Short-term income acceleration (3-month growth rate)",
    },
    {
        "name": "platform_count",
        "category": "income",
        "source": "AMT_INCOME_TOTAL",
        "method": "synthetic_platforms",
        "description": "Number of gig platforms the worker earns from",
    },
    {
        "name": "primary_platform_earnings_pct",
        "category": "income",
        "source": "platform_count",
        "method": "derived_pct",
        "description": "Fraction of income from primary platform (concentration risk)",
    },
    {
        "name": "secondary_platform_earnings_pct",
        "category": "income",
        "source": "platform_count",
        "method": "derived_pct",
        "description": "Fraction of income from secondary platform",
    },
    # --- PLATFORM / ACCOUNT FEATURES (6) ---
    {
        "name": "avg_platform_rating",
        "category": "platform",
        "source": "EXT_SOURCE_2",
        "method": "scale_to_rating",
        "description": "Average rating across gig platforms (1.0–5.0)",
    },
    {
        "name": "days_on_primary_platform",
        "category": "platform",
        "source": "DAYS_EMPLOYED",
        "method": "abs_days",
        "description": "Days active on primary gig platform",
    },
    {
        "name": "days_on_secondary_platform",
        "category": "platform",
        "source": "DAYS_EMPLOYED",
        "method": "derived_secondary_days",
        "description": "Days active on secondary platform (0 if single-platform)",
    },
    {
        "name": "platform_account_variance",
        "category": "platform",
        "source": ["DAYS_EMPLOYED", "DAYS_REGISTRATION"],
        "method": "derived_variance",
        "description": "Variance in account ages across platforms",
    },
    {
        "name": "account_age_months",
        "category": "platform",
        "source": "DAYS_REGISTRATION",
        "method": "days_to_months",
        "description": "Months since first platform registration",
    },
    {
        "name": "total_platforms",
        "category": "platform",
        "source": "platform_count",
        "method": "copy",
        "description": "Total number of gig platforms (same as platform_count)",
    },
    # --- SPENDING FEATURES (4) ---
    {
        "name": "spending_to_income_ratio",
        "category": "spending",
        "source": ["AMT_ANNUITY", "AMT_INCOME_TOTAL"],
        "method": "ratio",
        "description": "Monthly spending as fraction of income (financial discipline)",
    },
    {
        "name": "avg_monthly_spending",
        "category": "spending",
        "source": "AMT_ANNUITY",
        "method": "scale_spending",
        "description": "Average monthly expenditure",
    },
    {
        "name": "spending_volatility",
        "category": "spending",
        "source": "AMT_ANNUITY",
        "method": "synthetic_volatility",
        "description": "Variation in monthly spending (stability indicator)",
    },
    {
        "name": "spending_trend",
        "category": "spending",
        "source": ["AMT_ANNUITY", "EXT_SOURCE_3"],
        "method": "derived_trend",
        "description": "Spending growth direction (positive = increasing spend)",
    },
    # --- CREDIT HISTORY FEATURES (3) ---
    {
        "name": "late_payments_count_6m",
        "category": "credit",
        "source": "DEF_30_CNT_SOCIAL_CIRCLE",
        "method": "direct_int",
        "description": "Number of late payments in last 6 months",
    },
    {
        "name": "missed_payments_count_6m",
        "category": "credit",
        "source": "DEF_60_CNT_SOCIAL_CIRCLE",
        "method": "direct_int",
        "description": "Number of completely missed payments in last 6 months",
    },
    {
        "name": "existing_loan_count",
        "category": "credit",
        "source": "AMT_REQ_CREDIT_BUREAU_YEAR",
        "method": "direct_int",
        "description": "Number of existing active loans / credit inquiries",
    },
    # --- DEMOGRAPHIC FEATURES (2) ---
    {
        "name": "age",
        "category": "demographic",
        "source": "DAYS_BIRTH",
        "method": "days_to_years",
        "description": "Borrower age in years",
    },
    {
        "name": "account_tenure_months",
        "category": "demographic",
        "source": "DAYS_ID_PUBLISH",
        "method": "days_to_months",
        "description": "Months since ID was last issued (proxy for financial maturity)",
    },
]

# Quick lookup: expected output feature names
EXPECTED_FEATURE_NAMES = [f["name"] for f in FEATURE_REGISTRY]
assert len(EXPECTED_FEATURE_NAMES) == 23, f"Expected 23 features, got {len(EXPECTED_FEATURE_NAMES)}"


# ===================================================================
# STEP 1 — Load Cleaned Data
# ===================================================================
def load_cleaned_data(filepath: Path) -> pd.DataFrame:
    """
    Load the output of data_prep.py (cleaned_data.csv).

    Validates that required source columns exist before proceeding.

    Args:
        filepath: Path to cleaned_data.csv

    Returns:
        DataFrame ready for feature engineering.

    Raises:
        FileNotFoundError: If cleaned_data.csv doesn't exist.
    """
    logger.info("=" * 60)
    logger.info("STEP 1 — Loading cleaned data")
    logger.info("=" * 60)

    if not filepath.exists():
        logger.error(f"Cleaned data not found: {filepath}")
        logger.error("Run `python scripts/data_prep.py` first.")
        raise FileNotFoundError(f"Cleaned data not found: {filepath}")

    start = time.time()
    df = pd.read_csv(filepath)
    elapsed = time.time() - start

    logger.info(f"  Source:  {filepath}")
    logger.info(f"  Shape:   {df.shape[0]:,} rows × {df.shape[1]} columns")
    logger.info(f"  Loaded:  {elapsed:.2f}s")

    # --- Validate required source columns ---
    # Collect all unique source columns referenced in the registry
    required_sources = set()
    for feat in FEATURE_REGISTRY:
        src = feat["source"]
        if isinstance(src, list):
            required_sources.update(src)
        elif src not in EXPECTED_FEATURE_NAMES:
            # Skip references to features we'll create (e.g., platform_count)
            required_sources.add(src)

    available = set(df.columns)
    missing = required_sources - available

    if missing:
        logger.warning(f"  ⚠ Missing source columns ({len(missing)}): {sorted(missing)}")
        logger.warning(f"    These features will use fallback generation.")
    else:
        logger.info(f"  ✓ All {len(required_sources)} source columns present")

    return df


# ===================================================================
# STEP 2 — Engineer Income Features (8)
# ===================================================================
def engineer_income_features(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Create 8 income-related features from Home Credit columns.

    Mapping Strategy:
        AMT_INCOME_TOTAL → Base income (annual → monthly conversion)
        EXT_SOURCE_2/3   → Proxy for income trajectory/velocity
        Synthetic        → Platform count, earnings distribution

    Why synthetic features?
        Home Credit doesn't track gig platforms natively. We derive
        platform_count from income brackets (higher earners tend to
        diversify) and add calibrated noise for realism.

    Args:
        df:  Cleaned DataFrame.
        rng: Seeded random number generator for reproducibility.

    Returns:
        DataFrame with 8 new income columns added.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2 — Engineering Income Features (8)")
    logger.info("=" * 60)

    n = len(df)

    # ---------------------------------------------------------------
    # 2.1 avg_6m_income
    # AMT_INCOME_TOTAL is annual in Home Credit → divide by 12.
    # Add small noise to simulate month-to-month variation.
    # ---------------------------------------------------------------
    base_monthly = df["AMT_INCOME_TOTAL"].values / 12.0
    noise = rng.normal(1.0, 0.05, size=n)  # ±5% monthly variation
    df["avg_6m_income"] = np.maximum(base_monthly * noise, 5000)  # Floor: ₹5K
    logger.info(f"  ✓ avg_6m_income         — median: ₹{df['avg_6m_income'].median():,.0f}")

    # ---------------------------------------------------------------
    # 2.2 income_std_dev
    # Simulate income volatility. Gig workers with higher income tend
    # to be more stable (multi-platform diversification effect).
    # Coefficient of variation decreases with income level.
    # ---------------------------------------------------------------
    income_rank = df["avg_6m_income"].rank(pct=True)  # 0-1 percentile
    # Lower earners → CV ~0.30, Higher earners → CV ~0.10
    cv = 0.30 - 0.20 * income_rank + rng.normal(0, 0.03, size=n)
    cv = np.clip(cv, 0.05, 0.50)
    df["income_std_dev"] = df["avg_6m_income"] * cv
    logger.info(f"  ✓ income_std_dev        — median: ₹{df['income_std_dev'].median():,.0f}")

    # ---------------------------------------------------------------
    # 2.3 income_trend_6m
    # Use EXT_SOURCE_2 as a proxy for financial trajectory.
    # EXT_SOURCE_2 is a normalized external score (0-1) that strongly
    # correlates with repayment in Home Credit — we reinterpret it as
    # income growth direction: higher EXT_SOURCE → growing income.
    # ---------------------------------------------------------------
    if "EXT_SOURCE_2" in df.columns:
        # Map [0, 1] → [-0.3, +0.8] (most gig workers have positive trends)
        ext2 = df["EXT_SOURCE_2"].values
        df["income_trend_6m"] = -0.3 + 1.1 * ext2 + rng.normal(0, 0.05, size=n)
    else:
        # Fallback: generate from income level
        df["income_trend_6m"] = rng.uniform(-0.1, 0.6, size=n)

    df["income_trend_6m"] = df["income_trend_6m"].clip(-0.5, 1.0)
    logger.info(f"  ✓ income_trend_6m       — median: {df['income_trend_6m'].median():.3f}")

    # ---------------------------------------------------------------
    # 2.4 latest_month_income
    # Most recent month = avg_6m_income adjusted by trend direction.
    # Growing income → latest > average. Declining → latest < average.
    # ---------------------------------------------------------------
    trend_factor = 1.0 + df["income_trend_6m"] * 0.15  # Trend amplifies latest
    df["latest_month_income"] = df["avg_6m_income"] * trend_factor
    df["latest_month_income"] = df["latest_month_income"].clip(lower=3000)
    logger.info(f"  ✓ latest_month_income   — median: ₹{df['latest_month_income'].median():,.0f}")

    # ---------------------------------------------------------------
    # 2.5 income_velocity_3m
    # Short-term acceleration — how fast income is changing RIGHT NOW.
    # Uses EXT_SOURCE_3 as a proxy (another external score, often
    # reflects recent financial behavior).
    # ---------------------------------------------------------------
    if "EXT_SOURCE_3" in df.columns:
        ext3 = df["EXT_SOURCE_3"].values
        df["income_velocity_3m"] = -0.2 + 0.8 * ext3 + rng.normal(0, 0.04, size=n)
    else:
        df["income_velocity_3m"] = df["income_trend_6m"] * 0.6 + rng.normal(0, 0.05, size=n)

    df["income_velocity_3m"] = df["income_velocity_3m"].clip(-0.4, 0.8)
    logger.info(f"  ✓ income_velocity_3m    — median: {df['income_velocity_3m'].median():.3f}")

    # ---------------------------------------------------------------
    # 2.6 platform_count
    # Gig workers earning more tend to diversify across platforms.
    # We model this as a discrete distribution based on income tier:
    #   Bottom 30% → mostly 1 platform
    #   Middle 40% → 1-2 platforms
    #   Top 30%    → 2-3 platforms
    # ---------------------------------------------------------------
    income_pct = df["avg_6m_income"].rank(pct=True).values

    # Probability of multi-platform increases with income
    p_multi = np.clip(income_pct * 0.7 + rng.normal(0, 0.1, size=n), 0.05, 0.95)
    base_platform = np.ones(n, dtype=int)
    base_platform += (rng.random(n) < p_multi).astype(int)  # +1 if multi-platform
    base_platform += (rng.random(n) < p_multi * 0.3).astype(int)  # +1 more for top earners
    df["platform_count"] = np.clip(base_platform, 1, 4)

    platform_dist = df["platform_count"].value_counts().sort_index().to_dict()
    logger.info(f"  ✓ platform_count        — distribution: {platform_dist}")

    # ---------------------------------------------------------------
    # 2.7 primary_platform_earnings_pct
    # What fraction of total income comes from the main platform.
    # Single-platform workers → 100%. Multi-platform → decreases.
    # ---------------------------------------------------------------
    pc = df["platform_count"].values
    primary_pct = np.where(
        pc == 1,
        1.0,
        np.where(
            pc == 2,
            rng.uniform(0.55, 0.80, size=n),
            rng.uniform(0.40, 0.65, size=n),  # 3+ platforms
        ),
    )
    df["primary_platform_earnings_pct"] = np.round(primary_pct, 3)
    logger.info(
        f"  ✓ primary_platform_earnings_pct "
        f"— median: {df['primary_platform_earnings_pct'].median():.3f}"
    )

    # ---------------------------------------------------------------
    # 2.8 secondary_platform_earnings_pct
    # Remainder split among secondary platforms.
    # Single-platform → 0. Two-platform → rest. Three+ → split.
    # ---------------------------------------------------------------
    secondary_pct = np.where(
        pc == 1,
        0.0,
        np.where(
            pc == 2,
            1.0 - primary_pct,
            (1.0 - primary_pct) * rng.uniform(0.45, 0.65, size=n),
        ),
    )
    df["secondary_platform_earnings_pct"] = np.round(secondary_pct, 3)
    logger.info(
        f"  ✓ secondary_platform_earnings_pct "
        f"— median: {df['secondary_platform_earnings_pct'].median():.3f}"
    )

    return df


# ===================================================================
# STEP 3 — Engineer Platform / Account Features (6)
# ===================================================================
def engineer_platform_features(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Create 6 platform/account features.

    Mapping Strategy:
        EXT_SOURCE_2     → Platform rating proxy (external quality score)
        DAYS_EMPLOYED    → Days on primary platform (absolute value)
        DAYS_REGISTRATION→ Account age
        Derived          → Secondary platform days, variance

    Args:
        df:  DataFrame with income features already added.
        rng: Seeded RNG.

    Returns:
        DataFrame with 6 new platform columns added.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3 — Engineering Platform Features (6)")
    logger.info("=" * 60)

    n = len(df)

    # ---------------------------------------------------------------
    # 3.1 avg_platform_rating
    # Reinterpret EXT_SOURCE_2 as a platform quality/rating proxy.
    # EXT_SOURCE_2 ranges 0-1 in Home Credit; scale to 1.0-5.0 range.
    # Most gig workers maintain 4.0+ ratings (it's required to stay active).
    # ---------------------------------------------------------------
    if "EXT_SOURCE_2" in df.columns:
        # Skew toward high ratings: most workers are 3.5-5.0
        raw_rating = 3.0 + 2.0 * df["EXT_SOURCE_2"].values
        noise = rng.normal(0, 0.15, size=n)
        df["avg_platform_rating"] = np.clip(raw_rating + noise, 1.0, 5.0)
    else:
        df["avg_platform_rating"] = rng.uniform(3.5, 5.0, size=n)

    df["avg_platform_rating"] = df["avg_platform_rating"].round(2)
    logger.info(f"  ✓ avg_platform_rating       — median: {df['avg_platform_rating'].median():.2f}")

    # ---------------------------------------------------------------
    # 3.2 days_on_primary_platform
    # DAYS_EMPLOYED in Home Credit is negative (days before application).
    # Absolute value gives employment duration → proxy for platform tenure.
    # Cap at 3650 days (10 years) since gig economy is relatively new.
    # Handle the anomaly value 365243 (unemployed marker in Home Credit).
    # ---------------------------------------------------------------
    if "DAYS_EMPLOYED" in df.columns:
        days_emp = df["DAYS_EMPLOYED"].values.copy()
        # The value 365243 is a known anomaly meaning "unemployed/not applicable"
        days_emp = np.where(days_emp == 365243, 0, days_emp)
        df["days_on_primary_platform"] = np.clip(np.abs(days_emp), 30, 3650).astype(int)
    else:
        df["days_on_primary_platform"] = rng.integers(30, 2000, size=n)

    logger.info(
        f"  ✓ days_on_primary_platform  "
        f"— median: {df['days_on_primary_platform'].median():.0f} days"
    )

    # ---------------------------------------------------------------
    # 3.3 days_on_secondary_platform
    # Only relevant for multi-platform workers. Secondary platform was
    # typically joined later, so tenure is shorter.
    # Single-platform workers → 0.
    # ---------------------------------------------------------------
    pc = df["platform_count"].values
    primary_days = df["days_on_primary_platform"].values

    secondary_days = np.where(
        pc == 1,
        0,
        # Secondary platform = 20-70% of primary platform tenure
        (primary_days * rng.uniform(0.2, 0.7, size=n)).astype(int),
    )
    df["days_on_secondary_platform"] = np.clip(secondary_days, 0, 3000).astype(int)
    logger.info(
        f"  ✓ days_on_secondary_platform"
        f"— median: {df['days_on_secondary_platform'].median():.0f} days"
    )

    # ---------------------------------------------------------------
    # 3.4 platform_account_variance
    # Variance in tenure across platforms. High variance means the worker
    # started different platforms at very different times.
    # Single-platform → variance is 0.
    # ---------------------------------------------------------------
    p_days = df["days_on_primary_platform"].values.astype(float)
    s_days = df["days_on_secondary_platform"].values.astype(float)

    # Variance of [primary_days, secondary_days] for each row
    mean_days = (p_days + s_days) / 2.0
    variance = np.where(
        pc == 1,
        0.0,
        ((p_days - mean_days) ** 2 + (s_days - mean_days) ** 2) / 2.0,
    )
    df["platform_account_variance"] = np.round(variance, 2)
    logger.info(
        f"  ✓ platform_account_variance "
        f"— median: {df['platform_account_variance'].median():,.0f}"
    )

    # ---------------------------------------------------------------
    # 3.5 account_age_months
    # DAYS_REGISTRATION = days before application when the client
    # changed their registration. Convert to months.
    # ---------------------------------------------------------------
    if "DAYS_REGISTRATION" in df.columns:
        df["account_age_months"] = (np.abs(df["DAYS_REGISTRATION"].values) / 30.44).astype(int)
    else:
        df["account_age_months"] = rng.integers(3, 120, size=n)

    df["account_age_months"] = df["account_age_months"].clip(1, 360)
    logger.info(f"  ✓ account_age_months        — median: {df['account_age_months'].median():.0f}")

    # ---------------------------------------------------------------
    # 3.6 total_platforms
    # Same as platform_count (kept as separate feature for the
    # platform feature group; the model may weight it differently
    # in this context).
    # ---------------------------------------------------------------
    df["total_platforms"] = df["platform_count"].values
    logger.info(f"  ✓ total_platforms           — copied from platform_count")

    return df


# ===================================================================
# STEP 4 — Engineer Spending Features (4)
# ===================================================================
def engineer_spending_features(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Create 4 spending-related features.

    Mapping Strategy:
        AMT_ANNUITY      → Monthly spending proxy (loan annuity ≈ fixed outflow)
        AMT_INCOME_TOTAL → Used for ratio calculation
        EXT_SOURCE_3     → Spending trend direction

    Why AMT_ANNUITY as spending proxy?
        In the Home Credit context, the annuity is the borrower's committed
        monthly payment — this is the closest column to recurring monthly
        expenditure. We scale it to represent total monthly spending.

    Args:
        df:  DataFrame with income + platform features.
        rng: Seeded RNG.

    Returns:
        DataFrame with 4 new spending columns added.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 4 — Engineering Spending Features (4)")
    logger.info("=" * 60)

    n = len(df)

    # ---------------------------------------------------------------
    # 4.1 avg_monthly_spending
    # Scale AMT_ANNUITY to represent total monthly spending.
    # Typical annuity in Home Credit is 15-50K; we scale by a
    # multiplier to represent full monthly outflow (rent + food + etc.)
    # ---------------------------------------------------------------
    if "AMT_ANNUITY" in df.columns:
        # Spending ≈ annuity × 2-4 (annuity is just one obligation)
        multiplier = rng.uniform(2.0, 4.0, size=n)
        df["avg_monthly_spending"] = df["AMT_ANNUITY"].values * multiplier
    else:
        # Fallback: spending = 40-80% of income
        ratio = rng.uniform(0.4, 0.8, size=n)
        df["avg_monthly_spending"] = df["avg_6m_income"].values * ratio

    df["avg_monthly_spending"] = df["avg_monthly_spending"].clip(lower=5000).round(0)
    logger.info(
        f"  ✓ avg_monthly_spending       "
        f"— median: ₹{df['avg_monthly_spending'].median():,.0f}"
    )

    # ---------------------------------------------------------------
    # 4.2 spending_to_income_ratio
    # Core financial discipline metric. Low ratio = disciplined saver.
    # High ratio (>0.8) = living paycheck-to-paycheck.
    # Capped at 1.5 (some people spend more than they earn via debt).
    # ---------------------------------------------------------------
    income = df["avg_6m_income"].values
    spending = df["avg_monthly_spending"].values
    df["spending_to_income_ratio"] = np.round(
        np.clip(spending / np.maximum(income, 1.0), 0.1, 1.5), 3
    )
    logger.info(
        f"  ✓ spending_to_income_ratio   "
        f"— median: {df['spending_to_income_ratio'].median():.3f}"
    )

    # ---------------------------------------------------------------
    # 4.3 spending_volatility
    # How much monthly spending fluctuates. Similar logic to
    # income_std_dev but for the spending side.
    # Lower volatility = more predictable expenses.
    # ---------------------------------------------------------------
    cv_spending = rng.uniform(0.08, 0.35, size=n)
    df["spending_volatility"] = np.round(
        df["avg_monthly_spending"].values * cv_spending, 2
    )
    logger.info(
        f"  ✓ spending_volatility        "
        f"— median: ₹{df['spending_volatility'].median():,.0f}"
    )

    # ---------------------------------------------------------------
    # 4.4 spending_trend
    # Is spending increasing or decreasing over time?
    # Positive = spending is growing (potentially risky if income isn't).
    # Uses EXT_SOURCE_3 as a weak proxy, inverted (high ext → lower
    # spending growth, since better scores correlate with discipline).
    # ---------------------------------------------------------------
    if "EXT_SOURCE_3" in df.columns:
        ext3 = df["EXT_SOURCE_3"].values
        # Invert: good external score → controlled spending growth
        df["spending_trend"] = 0.3 - 0.4 * ext3 + rng.normal(0, 0.06, size=n)
    else:
        df["spending_trend"] = rng.uniform(-0.1, 0.3, size=n)

    df["spending_trend"] = df["spending_trend"].clip(-0.3, 0.5).round(3)
    logger.info(f"  ✓ spending_trend             — median: {df['spending_trend'].median():.3f}")

    return df


# ===================================================================
# STEP 5 — Engineer Credit History Features (3)
# ===================================================================
def engineer_credit_features(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Create 3 credit history features.

    Mapping Strategy:
        DEF_30_CNT_SOCIAL_CIRCLE  → Late payments (30-day defaults in social circle)
        DEF_60_CNT_SOCIAL_CIRCLE  → Missed payments (60-day defaults)
        AMT_REQ_CREDIT_BUREAU_YEAR → Existing loan count / credit inquiries

    These are among the most direct mappings in the pipeline — Home Credit
    already tracks default counts and credit bureau activity.

    Args:
        df:  DataFrame with income + platform + spending features.
        rng: Seeded RNG.

    Returns:
        DataFrame with 3 new credit columns added.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 5 — Engineering Credit Features (3)")
    logger.info("=" * 60)

    # ---------------------------------------------------------------
    # 5.1 late_payments_count_6m
    # DEF_30_CNT_SOCIAL_CIRCLE = number of 30-day defaults observed
    # in the borrower's social circle. We use this as a proxy for
    # the borrower's own late payment tendency.
    # ---------------------------------------------------------------
    if "DEF_30_CNT_SOCIAL_CIRCLE" in df.columns:
        df["late_payments_count_6m"] = df["DEF_30_CNT_SOCIAL_CIRCLE"].clip(0, 10).astype(int)
    else:
        df["late_payments_count_6m"] = rng.choice([0, 0, 0, 0, 1, 1, 2, 3], size=len(df))

    late_dist = df["late_payments_count_6m"].value_counts().sort_index().head(5).to_dict()
    logger.info(f"  ✓ late_payments_count_6m     — distribution: {late_dist}")

    # ---------------------------------------------------------------
    # 5.2 missed_payments_count_6m
    # DEF_60_CNT_SOCIAL_CIRCLE = 60-day defaults (more severe).
    # Fewer people miss payments entirely, so counts are lower.
    # ---------------------------------------------------------------
    if "DEF_60_CNT_SOCIAL_CIRCLE" in df.columns:
        df["missed_payments_count_6m"] = df["DEF_60_CNT_SOCIAL_CIRCLE"].clip(0, 5).astype(int)
    else:
        df["missed_payments_count_6m"] = rng.choice([0, 0, 0, 0, 0, 1, 1, 2], size=len(df))

    missed_dist = df["missed_payments_count_6m"].value_counts().sort_index().head(5).to_dict()
    logger.info(f"  ✓ missed_payments_count_6m   — distribution: {missed_dist}")

    # ---------------------------------------------------------------
    # 5.3 existing_loan_count
    # AMT_REQ_CREDIT_BUREAU_YEAR = number of credit bureau inquiries
    # in the past year. High count → actively seeking credit.
    # ---------------------------------------------------------------
    if "AMT_REQ_CREDIT_BUREAU_YEAR" in df.columns:
        df["existing_loan_count"] = df["AMT_REQ_CREDIT_BUREAU_YEAR"].clip(0, 15).astype(int)
    else:
        df["existing_loan_count"] = rng.choice([0, 0, 1, 1, 2, 2, 3, 4, 5], size=len(df))

    loan_dist = df["existing_loan_count"].value_counts().sort_index().head(5).to_dict()
    logger.info(f"  ✓ existing_loan_count        — distribution: {loan_dist}")

    return df


# ===================================================================
# STEP 6 — Engineer Demographic Features (2)
# ===================================================================
def engineer_demographic_features(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Create 2 demographic features.

    Mapping Strategy:
        DAYS_BIRTH      → Age in years (negative days → positive years)
        DAYS_ID_PUBLISH → Account tenure in months (ID issuance proxy)

    These are the most straightforward mappings — just unit conversions.

    Args:
        df:  DataFrame with all prior feature groups.
        rng: Seeded RNG.

    Returns:
        DataFrame with 2 new demographic columns added.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 6 — Engineering Demographic Features (2)")
    logger.info("=" * 60)

    # ---------------------------------------------------------------
    # 6.1 age
    # DAYS_BIRTH is negative in Home Credit (days before application).
    # Convert: abs(DAYS_BIRTH) / 365.25 → years.
    # ---------------------------------------------------------------
    if "DAYS_BIRTH" in df.columns:
        df["age"] = (np.abs(df["DAYS_BIRTH"].values) / 365.25).astype(int)
    else:
        df["age"] = rng.integers(20, 60, size=len(df))

    df["age"] = df["age"].clip(18, 80)
    logger.info(
        f"  ✓ age                        "
        f"— range: {df['age'].min()}-{df['age'].max()}, "
        f"median: {df['age'].median():.0f}"
    )

    # ---------------------------------------------------------------
    # 6.2 account_tenure_months
    # DAYS_ID_PUBLISH = days since ID document was issued.
    # Longer tenure → more established in the formal financial system.
    # ---------------------------------------------------------------
    if "DAYS_ID_PUBLISH" in df.columns:
        df["account_tenure_months"] = (np.abs(df["DAYS_ID_PUBLISH"].values) / 30.44).astype(int)
    else:
        df["account_tenure_months"] = rng.integers(6, 180, size=len(df))

    df["account_tenure_months"] = df["account_tenure_months"].clip(1, 600)
    logger.info(
        f"  ✓ account_tenure_months      "
        f"— median: {df['account_tenure_months'].median():.0f} months"
    )

    return df


# ===================================================================
# STEP 7 — Normalize & Select Final 23 Features
# ===================================================================
def normalize_and_select(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    StandardScaler normalization on all 23 engineered features.

    Why normalize?
        XGBoost doesn't strictly require normalization (it's tree-based),
        but normalizing helps with:
        - SHAP value interpretation (comparable scales)
        - Feature importance visualization
        - Potential future ensemble with linear models

    The scaler is fit on the FULL dataset here. During train/test split
    in train.py, the scaler will be re-fit on training data only.
    This pass is for exploratory analysis and sanity checking.

    Args:
        df: DataFrame with all 23 engineered features + TARGET.

    Returns:
        (final_df, scaler_metadata) — normalized features + scaler stats.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 7 — Normalizing & selecting final 23 features")
    logger.info("=" * 60)

    # --- Extract the 23 features + TARGET ---
    available_features = [f for f in EXPECTED_FEATURE_NAMES if f in df.columns]
    missing_features = [f for f in EXPECTED_FEATURE_NAMES if f not in df.columns]

    if missing_features:
        logger.error(f"  ✗ Missing features: {missing_features}")
        raise KeyError(f"Missing engineered features: {missing_features}")

    logger.info(f"  Selected {len(available_features)} features + TARGET")

    # Build final DataFrame: features + target
    columns_to_keep = available_features + [TARGET_COL]
    final_df = df[columns_to_keep].copy()

    # --- Apply StandardScaler ---
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(final_df[available_features])
    final_df[available_features] = scaled_values

    logger.info(f"  ✓ StandardScaler applied")
    logger.info(f"  Final shape: {final_df.shape[0]:,} rows × {final_df.shape[1]} columns")

    # --- Build scaler metadata (for reproducibility in inference) ---
    scaler_metadata = {
        "scaler_type": "StandardScaler",
        "features": available_features,
        "means": {f: round(float(m), 6) for f, m in zip(available_features, scaler.mean_)},
        "stds": {
            f: round(float(s), 6)
            for f, s in zip(available_features, scaler.scale_)
        },
    }

    # Log a quick summary of scaled distributions
    logger.info(f"  Post-normalization stats (should be ~mean=0, std=1):")
    sample_features = available_features[:5]
    for feat in sample_features:
        mean = final_df[feat].mean()
        std = final_df[feat].std()
        logger.info(f"    {feat:<35s} mean={mean:+.4f}  std={std:.4f}")
    if len(available_features) > 5:
        logger.info(f"    ... and {len(available_features) - 5} more features")

    return final_df, scaler_metadata


# ===================================================================
# STEP 8 — Save Outputs
# ===================================================================
def save_outputs(
    df: pd.DataFrame,
    scaler_meta: dict,
    output_csv: Path,
    output_scaler: Path,
    output_mapping: Path,
) -> None:
    """
    Save the engineered feature matrix, scaler metadata, and feature mapping.

    Outputs:
        engineered_features.csv  — 23 normalized features + TARGET (307K rows)
        scaler_metadata.json     — StandardScaler means/stds for inference
        feature_mapping.json     — Full registry documenting each feature's origin

    Args:
        df:             Final feature DataFrame.
        scaler_meta:    Scaler means and standard deviations.
        output_csv:     Path for engineered_features.csv.
        output_scaler:  Path for scaler_metadata.json.
        output_mapping: Path for feature_mapping.json.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 8 — Saving outputs")
    logger.info("=" * 60)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # --- 8.1 Save feature matrix ---
    start = time.time()
    df.to_csv(output_csv, index=False)
    elapsed = time.time() - start
    size_mb = output_csv.stat().st_size / (1024 * 1024)
    logger.info(f"  ✓ {output_csv.name}")
    logger.info(f"    Shape: {df.shape[0]:,} × {df.shape[1]} | {size_mb:.1f} MB | {elapsed:.2f}s")

    # --- 8.2 Save scaler metadata ---
    scaler_meta["generated_at"] = datetime.now().isoformat()
    with open(output_scaler, "w", encoding="utf-8") as f:
        json.dump(scaler_meta, f, indent=2)
    logger.info(f"  ✓ {output_scaler.name}")

    # --- 8.3 Save feature mapping (the full registry) ---
    mapping = {
        "generated_at": datetime.now().isoformat(),
        "total_features": len(FEATURE_REGISTRY),
        "categories": {},
        "features": FEATURE_REGISTRY,
    }
    # Summarize by category
    for feat in FEATURE_REGISTRY:
        cat = feat["category"]
        mapping["categories"][cat] = mapping["categories"].get(cat, 0) + 1

    with open(output_mapping, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    logger.info(f"  ✓ {output_mapping.name}")
    logger.info(f"    Feature categories: {mapping['categories']}")


# ===================================================================
# Correlation Analysis (Bonus diagnostic)
# ===================================================================
def log_target_correlations(df: pd.DataFrame) -> None:
    """
    Log the Pearson correlation of each feature with TARGET.

    This is a quick diagnostic to verify that engineered features have
    at least some predictive signal. Features with |correlation| < 0.01
    are flagged as potentially low-value.

    Args:
        df: Final feature DataFrame (normalized).
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("DIAGNOSTIC — Feature-Target correlations")
    logger.info("=" * 60)

    features = [c for c in df.columns if c != TARGET_COL]
    correlations = df[features].corrwith(df[TARGET_COL]).sort_values()

    logger.info(f"  Top 5 negative correlations (protective against default):")
    for feat, corr in correlations.head(5).items():
        logger.info(f"    {feat:<35s} r = {corr:+.4f}")

    logger.info(f"  Top 5 positive correlations (associated with default):")
    for feat, corr in correlations.tail(5).items():
        logger.info(f"    {feat:<35s} r = {corr:+.4f}")

    # Flag weak features
    weak = correlations[correlations.abs() < 0.01]
    if len(weak) > 0:
        logger.warning(f"  ⚠ {len(weak)} features with |r| < 0.01: {list(weak.index)}")
    else:
        logger.info(f"  ✓ All features have |r| ≥ 0.01 with TARGET")


# ===================================================================
# Main Pipeline Orchestrator
# ===================================================================
def run_pipeline(input_path: Path, seed: int = DEFAULT_SEED) -> None:
    """
    Execute the full feature engineering pipeline.

    Pipeline flow:
        cleaned_data.csv (100+ cols)
          → Engineer 8 income features
          → Engineer 6 platform features
          → Engineer 4 spending features
          → Engineer 3 credit features
          → Engineer 2 demographic features
          → Normalize (StandardScaler)
          → Save 23 features + TARGET

    Args:
        input_path: Path to cleaned_data.csv (output of data_prep.py).
        seed:       Random seed for reproducibility.
    """
    pipeline_start = time.time()

    # Initialize seeded RNG for all synthetic feature generation
    rng = np.random.default_rng(seed)

    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║   FlowScore — Feature Engineering Pipeline               ║")
    logger.info("║   122 raw columns → 23 gig-specific features             ║")
    logger.info("╚" + "═" * 58 + "╝")
    logger.info("")
    logger.info(f"  Input:   {input_path}")
    logger.info(f"  Seed:    {seed}")
    logger.info(f"  Output:  {OUTPUT_FEATURES_CSV}")
    logger.info(f"  Log:     {LOG_DIR / log_filename}")
    logger.info("")

    # --- Execute pipeline steps ---
    df = load_cleaned_data(input_path)                         # Step 1
    df = engineer_income_features(df, rng)                     # Step 2
    df = engineer_platform_features(df, rng)                   # Step 3
    df = engineer_spending_features(df, rng)                   # Step 4
    df = engineer_credit_features(df, rng)                     # Step 5
    df = engineer_demographic_features(df, rng)                # Step 6
    final_df, scaler_meta = normalize_and_select(df)           # Step 7
    save_outputs(                                              # Step 8
        final_df, scaler_meta,
        OUTPUT_FEATURES_CSV, OUTPUT_SCALER_META, OUTPUT_FEATURE_MAP,
    )
    log_target_correlations(final_df)                          # Diagnostic

    # --- Pipeline summary ---
    elapsed = time.time() - pipeline_start
    logger.info("")
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║   ✓ Feature Engineering Complete                         ║")
    logger.info("╠" + "═" * 58 + "╣")
    logger.info(f"║   Total time:       {elapsed:>6.2f}s{' ' * 32}║")
    logger.info(f"║   Input columns:    {df.shape[1]:>6}{' ' * 34}║")
    logger.info(f"║   Output features:  {len(EXPECTED_FEATURE_NAMES):>6} + TARGET{' ' * 24}║")
    logger.info(f"║   Output rows:      {len(final_df):>8,}{' ' * 28}║")
    logger.info("╠" + "═" * 58 + "╣")
    logger.info("║   Next step: python model/train.py                       ║")
    logger.info("╚" + "═" * 58 + "╝")


# ===================================================================
# CLI Entry Point
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description="FlowScore — Engineer 23 gig-worker features from Home Credit data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/feature_engineering.py
  python scripts/feature_engineering.py --input data/processed/cleaned_data.csv
  python scripts/feature_engineering.py --seed 123
        """,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to cleaned CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})",
    )
    args = parser.parse_args()

    try:
        run_pipeline(input_path=args.input, seed=args.seed)
    except FileNotFoundError as e:
        logger.error(f"Input file not found: {e}")
        sys.exit(1)
    except KeyError as e:
        logger.error(f"Missing required column or feature: {e}")
        sys.exit(1)
    except MemoryError:
        logger.error("Out of memory during feature engineering.")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
