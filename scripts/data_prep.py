"""
FlowScore — Data Preparation Pipeline
=======================================

Loads the Home Credit Default Risk dataset (application_train.csv),
cleans it, and outputs analysis-ready files.

Pipeline Steps:
  1. Load raw CSV from data/raw/
  2. Drop columns with >50% missing values
  3. Impute remaining missing values (median for numeric, mode for categorical)
  4. Remove duplicate rows
  5. Analyze TARGET variable distribution (0 = repaid, 1 = defaulted)
  6. Output:
       - data/processed/cleaned_data.csv    — cleaned dataset
       - data/processed/feature_list.json   — feature metadata

Usage:
    python scripts/data_prep.py
    python scripts/data_prep.py --input data/raw/application_train.csv
    python scripts/data_prep.py --missing-threshold 0.4   # stricter: drop cols >40% missing

Author: FlowScore Team
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_INPUT = RAW_DATA_DIR / "application_train.csv"
OUTPUT_CSV = PROCESSED_DATA_DIR / "cleaned_data.csv"
OUTPUT_FEATURES = PROCESSED_DATA_DIR / "feature_list.json"

# Target column in the Home Credit dataset
TARGET_COL = "TARGET"

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Generate a timestamped log filename so runs don't overwrite each other
log_filename = f"data_prep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),                      # Console output
        logging.FileHandler(LOG_DIR / log_filename, mode="w", encoding="utf-8"),  # File output
    ],
)
logger = logging.getLogger("flowscore.data_prep")


# ===================================================================
# STEP 1 — Load Raw Data
# ===================================================================
def load_data(filepath: Path) -> pd.DataFrame:
    """
    Load the Home Credit CSV into a DataFrame.

    Args:
        filepath: Absolute path to application_train.csv

    Returns:
        Raw DataFrame (unmodified).

    Raises:
        FileNotFoundError: If the CSV does not exist.
        pd.errors.EmptyDataError: If the CSV is empty.
    """
    logger.info("=" * 60)
    logger.info("STEP 1 — Loading raw data")
    logger.info("=" * 60)

    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        logger.error(
            "Run `python scripts/download_dataset.py` first to download the dataset."
        )
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    file_size_mb = filepath.stat().st_size / (1024 * 1024)
    logger.info(f"  Source:    {filepath}")
    logger.info(f"  File size: {file_size_mb:.1f} MB")

    start = time.time()
    df = pd.read_csv(filepath)
    elapsed = time.time() - start

    logger.info(f"  Loaded in: {elapsed:.2f}s")
    logger.info(f"  Shape:     {df.shape[0]:,} rows × {df.shape[1]} columns")
    logger.info(f"  Memory:    {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    # Quick sanity check: TARGET column must exist
    if TARGET_COL not in df.columns:
        logger.error(f"TARGET column '{TARGET_COL}' not found in dataset!")
        logger.error(f"Available columns: {list(df.columns[:10])}...")
        raise KeyError(f"Missing required column: {TARGET_COL}")

    return df


# ===================================================================
# STEP 2 — Drop High-Missingness Columns
# ===================================================================
def drop_high_missing_columns(
    df: pd.DataFrame, threshold: float = 0.50
) -> pd.DataFrame:
    """
    Drop columns where the fraction of missing values exceeds `threshold`.

    The Home Credit dataset has several columns with 60-70% missing data
    (e.g., OWN_CAR_AGE, APARTMENTS_AVG, etc.). These add noise rather
    than signal, so we remove them early.

    Args:
        df:        Input DataFrame.
        threshold: Maximum allowed fraction of missing values (default: 0.50).

    Returns:
        DataFrame with high-missingness columns removed.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"STEP 2 — Dropping columns with >{threshold*100:.0f}% missing values")
    logger.info("=" * 60)

    n_rows = len(df)
    missing_fractions = df.isnull().sum() / n_rows

    # Identify columns exceeding the threshold
    cols_to_drop = missing_fractions[missing_fractions > threshold].index.tolist()

    if cols_to_drop:
        logger.info(f"  Columns exceeding {threshold*100:.0f}% missing: {len(cols_to_drop)}")

        # Log each dropped column with its missing percentage
        for col in sorted(cols_to_drop, key=lambda c: missing_fractions[c], reverse=True):
            pct = missing_fractions[col] * 100
            logger.info(f"    ✗ {col:<40s} ({pct:.1f}% missing)")

        df = df.drop(columns=cols_to_drop)
        logger.info(f"  Remaining columns: {df.shape[1]}")
    else:
        logger.info(f"  No columns exceed {threshold*100:.0f}% missing — none dropped.")

    return df


# ===================================================================
# STEP 3 — Impute Missing Values
# ===================================================================
def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill remaining missing values using appropriate strategies:
      - Numeric columns  → median imputation
      - Categorical columns (object/category dtype) → mode imputation

    Median is preferred over mean for numeric columns because the Home Credit
    dataset has highly skewed distributions (income, credit amounts, etc.).

    Args:
        df: DataFrame with high-missingness columns already removed.

    Returns:
        DataFrame with no missing values.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3 — Imputing remaining missing values")
    logger.info("=" * 60)

    total_missing_before = df.isnull().sum().sum()
    cols_with_missing = df.columns[df.isnull().any()].tolist()

    logger.info(f"  Total missing cells before: {total_missing_before:,}")
    logger.info(f"  Columns with missing data:  {len(cols_with_missing)}")

    # --- Separate columns by dtype ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # --- Numeric: Median Imputation ---
    numeric_missing = [c for c in numeric_cols if df[c].isnull().any()]
    if numeric_missing:
        logger.info(f"  Numeric columns to impute (median): {len(numeric_missing)}")

        # Compute medians once — much faster than column-by-column fillna
        medians = df[numeric_missing].median()

        for col in numeric_missing:
            n_missing = df[col].isnull().sum()
            pct = n_missing / len(df) * 100
            logger.info(
                f"    → {col:<40s} {n_missing:>7,} missing ({pct:5.1f}%) "
                f"→ filled with median={medians[col]:.4f}"
            )

        df[numeric_missing] = df[numeric_missing].fillna(medians)

    # --- Categorical: Mode Imputation ---
    categorical_missing = [c for c in categorical_cols if df[c].isnull().any()]
    if categorical_missing:
        logger.info(f"  Categorical columns to impute (mode): {len(categorical_missing)}")

        for col in categorical_missing:
            n_missing = df[col].isnull().sum()
            pct = n_missing / len(df) * 100
            mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else "UNKNOWN"
            logger.info(
                f"    → {col:<40s} {n_missing:>7,} missing ({pct:5.1f}%) "
                f'→ filled with mode="{mode_value}"'
            )
            df[col] = df[col].fillna(mode_value)

    # --- Verify: No missing values remain ---
    total_missing_after = df.isnull().sum().sum()
    if total_missing_after == 0:
        logger.info(f"  ✓ All missing values resolved. Remaining nulls: 0")
    else:
        # This shouldn't happen, but handle defensively
        logger.warning(
            f"  ⚠ {total_missing_after:,} missing values remain after imputation!"
        )
        remaining = df.columns[df.isnull().any()].tolist()
        logger.warning(f"    Columns: {remaining}")

    return df


# ===================================================================
# STEP 4 — Remove Duplicate Rows
# ===================================================================
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact duplicate rows.

    The Home Credit dataset shouldn't have true duplicates (each row is a
    unique loan application with SK_ID_CURR), but we check defensively.

    Args:
        df: Cleaned DataFrame.

    Returns:
        DataFrame with duplicates removed.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 4 — Removing duplicate rows")
    logger.info("=" * 60)

    n_before = len(df)
    df = df.drop_duplicates()
    n_after = len(df)
    n_dropped = n_before - n_after

    if n_dropped > 0:
        logger.info(f"  Removed {n_dropped:,} duplicate rows")
        logger.info(f"  Rows: {n_before:,} → {n_after:,}")
    else:
        logger.info(f"  No duplicate rows found. Total rows: {n_after:,}")

    # Also check for duplicate loan IDs (SK_ID_CURR)
    if "SK_ID_CURR" in df.columns:
        n_unique_ids = df["SK_ID_CURR"].nunique()
        if n_unique_ids < len(df):
            n_dup_ids = len(df) - n_unique_ids
            logger.warning(
                f"  ⚠ {n_dup_ids:,} duplicate SK_ID_CURR values found "
                f"(keeping all — may be multiple applications)"
            )
        else:
            logger.info(f"  ✓ All {n_unique_ids:,} SK_ID_CURR values are unique")

    return df


# ===================================================================
# STEP 5 — Analyze TARGET Variable Distribution
# ===================================================================
def analyze_target(df: pd.DataFrame) -> dict:
    """
    Analyze the TARGET variable distribution and log class imbalance stats.

    TARGET:
      0 = Loan repaid on time  (majority class, ~92%)
      1 = Loan defaulted        (minority class, ~8%)

    This extreme imbalance (11.5:1 ratio) informs our choice to use
    scale_pos_weight in XGBoost during training.

    Args:
        df: Cleaned DataFrame with TARGET column.

    Returns:
        Dictionary with distribution statistics.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 5 — Analyzing TARGET variable distribution")
    logger.info("=" * 60)

    if TARGET_COL not in df.columns:
        logger.error(f"  TARGET column '{TARGET_COL}' not found!")
        return {}

    # --- Value counts ---
    counts = df[TARGET_COL].value_counts().sort_index()
    total = len(df)

    target_stats = {
        "total_rows": total,
        "class_0_count": int(counts.get(0, 0)),
        "class_1_count": int(counts.get(1, 0)),
        "class_0_pct": round(counts.get(0, 0) / total * 100, 2),
        "class_1_pct": round(counts.get(1, 0) / total * 100, 2),
        "imbalance_ratio": round(counts.get(0, 1) / max(counts.get(1, 1), 1), 2),
    }

    logger.info(f"  Total samples: {total:,}")
    logger.info(f"  ┌──────────────────────────────────────────────┐")
    logger.info(f"  │ Class 0 (Repaid):    {target_stats['class_0_count']:>8,}  ({target_stats['class_0_pct']:5.2f}%)  │")
    logger.info(f"  │ Class 1 (Defaulted): {target_stats['class_1_count']:>8,}  ({target_stats['class_1_pct']:5.2f}%)  │")
    logger.info(f"  │ Imbalance Ratio:     {target_stats['imbalance_ratio']:>8.1f}:1              │")
    logger.info(f"  └──────────────────────────────────────────────┘")

    # --- Recommendations based on imbalance ---
    if target_stats["imbalance_ratio"] > 5:
        logger.info(
            f"  ⚠ Significant class imbalance detected ({target_stats['imbalance_ratio']:.1f}:1)"
        )
        logger.info(f"  Recommended strategies for training:")
        logger.info(f"    • XGBoost: scale_pos_weight = {target_stats['imbalance_ratio']:.0f}")
        logger.info(f"    • Scikit-learn: class_weight='balanced'")
        logger.info(f"    • Evaluation: Use AUC-ROC (not accuracy) as primary metric")

    return target_stats


# ===================================================================
# STEP 6 — Generate Feature Metadata
# ===================================================================
def generate_feature_metadata(df: pd.DataFrame, target_stats: dict) -> dict:
    """
    Build a comprehensive feature metadata dictionary.

    This JSON output serves as the single source of truth for:
      - Which features survived cleaning
      - Their data types, unique counts, and basic stats
      - The target variable distribution

    Args:
        df: Cleaned DataFrame.
        target_stats: Output from analyze_target().

    Returns:
        Feature metadata dictionary (also saved to JSON).
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 6 — Generating feature metadata")
    logger.info("=" * 60)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Build per-feature metadata
    features = []
    for col in df.columns:
        if col == TARGET_COL:
            continue  # Exclude target from feature list

        feat_info = {
            "name": col,
            "dtype": str(df[col].dtype),
            "n_unique": int(df[col].nunique()),
            "missing_pct": 0.0,  # Should be 0 after imputation
        }

        if col in numeric_cols:
            feat_info["type"] = "numeric"
            feat_info["mean"] = round(float(df[col].mean()), 4)
            feat_info["std"] = round(float(df[col].std()), 4)
            feat_info["min"] = round(float(df[col].min()), 4)
            feat_info["max"] = round(float(df[col].max()), 4)
            feat_info["median"] = round(float(df[col].median()), 4)
        else:
            feat_info["type"] = "categorical"
            # Top 5 most frequent categories
            top_values = df[col].value_counts().head(5).to_dict()
            feat_info["top_categories"] = {str(k): int(v) for k, v in top_values.items()}

        features.append(feat_info)

    # Sort: numeric first, then categorical, alphabetically within each group
    features.sort(key=lambda f: (0 if f["type"] == "numeric" else 1, f["name"]))

    metadata = {
        "generated_at": datetime.now().isoformat(),
        "pipeline_version": "1.0.0",
        "dataset": {
            "source": "home-credit-default-risk",
            "original_file": "application_train.csv",
            "rows": len(df),
            "columns": len(df.columns),
        },
        "target": target_stats,
        "feature_summary": {
            "total_features": len(features),
            "numeric_count": sum(1 for f in features if f["type"] == "numeric"),
            "categorical_count": sum(1 for f in features if f["type"] == "categorical"),
        },
        "features": features,
    }

    logger.info(f"  Total features: {len(features)}")
    logger.info(f"    Numeric:     {metadata['feature_summary']['numeric_count']}")
    logger.info(f"    Categorical: {metadata['feature_summary']['categorical_count']}")

    return metadata


# ===================================================================
# STEP 7 — Save Outputs
# ===================================================================
def save_outputs(
    df: pd.DataFrame,
    metadata: dict,
    output_csv: Path,
    output_json: Path,
) -> None:
    """
    Save the cleaned dataset and feature metadata to disk.

    Args:
        df:         Cleaned DataFrame.
        metadata:   Feature metadata dictionary.
        output_csv: Path for cleaned_data.csv.
        output_json: Path for feature_list.json.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 7 — Saving outputs")
    logger.info("=" * 60)

    # Ensure output directory exists
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # --- Save cleaned CSV ---
    start = time.time()
    df.to_csv(output_csv, index=False)
    elapsed = time.time() - start
    size_mb = output_csv.stat().st_size / (1024 * 1024)
    logger.info(f"  ✓ Saved: {output_csv}")
    logger.info(f"    Size: {size_mb:.1f} MB | Time: {elapsed:.2f}s")

    # --- Save feature metadata JSON ---
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"  ✓ Saved: {output_json}")
    logger.info(f"    Features documented: {len(metadata['features'])}")


# ===================================================================
# Main Pipeline Orchestrator
# ===================================================================
def run_pipeline(input_path: Path, missing_threshold: float = 0.50) -> None:
    """
    Execute the full data preparation pipeline end-to-end.

    Args:
        input_path:        Path to raw application_train.csv.
        missing_threshold: Drop columns with missing fraction above this value.
    """
    pipeline_start = time.time()

    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║   FlowScore — Data Preparation Pipeline                  ║")
    logger.info("║   Home Credit Default Risk Dataset                       ║")
    logger.info("╚" + "═" * 58 + "╝")
    logger.info("")
    logger.info(f"  Input:              {input_path}")
    logger.info(f"  Missing threshold:  {missing_threshold*100:.0f}%")
    logger.info(f"  Output CSV:         {OUTPUT_CSV}")
    logger.info(f"  Output features:    {OUTPUT_FEATURES}")
    logger.info(f"  Log file:           {LOG_DIR / log_filename}")
    logger.info("")

    # --- Execute pipeline steps ---
    df = load_data(input_path)                                      # Step 1
    df = drop_high_missing_columns(df, threshold=missing_threshold) # Step 2
    df = impute_missing_values(df)                                  # Step 3
    df = remove_duplicates(df)                                      # Step 4
    target_stats = analyze_target(df)                               # Step 5
    metadata = generate_feature_metadata(df, target_stats)          # Step 6
    save_outputs(df, metadata, OUTPUT_CSV, OUTPUT_FEATURES)         # Step 7

    # --- Pipeline summary ---
    elapsed = time.time() - pipeline_start
    logger.info("")
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║   ✓ Pipeline Complete                                    ║")
    logger.info("╠" + "═" * 58 + "╣")
    logger.info(f"║   Total time:      {elapsed:>6.2f}s{' ' * 33}║")
    logger.info(f"║   Output rows:     {len(df):>8,}{' ' * 28}║")
    logger.info(f"║   Output columns:  {df.shape[1]:>8}{' ' * 28}║")
    logger.info(f"║   Missing values:  {df.isnull().sum().sum():>8}{' ' * 28}║")
    logger.info("╠" + "═" * 58 + "╣")
    logger.info("║   Next step: python model/train.py                       ║")
    logger.info("╚" + "═" * 58 + "╝")


# ===================================================================
# CLI Entry Point
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description="FlowScore — Prepare Home Credit dataset for ML training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/data_prep.py
  python scripts/data_prep.py --input data/raw/application_train.csv
  python scripts/data_prep.py --missing-threshold 0.4
        """,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to raw CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--missing-threshold",
        type=float,
        default=0.50,
        help="Drop columns with missing fraction above this (default: 0.50)",
    )
    args = parser.parse_args()

    try:
        run_pipeline(
            input_path=args.input,
            missing_threshold=args.missing_threshold,
        )
    except FileNotFoundError as e:
        logger.error(f"Dataset not found: {e}")
        sys.exit(1)
    except KeyError as e:
        logger.error(f"Missing required column: {e}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        logger.error("The CSV file is empty or corrupted.")
        sys.exit(1)
    except MemoryError:
        logger.error(
            "Out of memory! The dataset is too large for available RAM. "
            "Try running on a machine with more memory or process in chunks."
        )
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error during data preparation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
