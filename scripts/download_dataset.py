"""
FlowScore — Home Credit Default Risk Dataset Downloader
========================================================

Downloads the Kaggle Home Credit Default Risk competition dataset.
Source: https://www.kaggle.com/c/home-credit-default-risk/data

Dataset overview:
  - application_train.csv : 307,511 rows × 122 columns (primary training data)
  - application_test.csv  : 48,744 rows × 121 columns  (no TARGET column)
  - We only need application_train.csv for FlowScore.

TARGET column:
  - 0 = loan repaid on time  (~92%)
  - 1 = loan defaulted        (~8%)

=====================
SETUP INSTRUCTIONS
=====================

Option A — Using the Kaggle CLI (recommended)
----------------------------------------------
1. Create a Kaggle account at https://www.kaggle.com
2. Go to Account Settings → API → "Create New Token"
3. This downloads a `kaggle.json` file containing:
       {"username": "YOUR_USERNAME", "key": "YOUR_API_KEY"}
4. Place kaggle.json in the correct location:
       Windows:  C:\\Users\\<YOU>\\.kaggle\\kaggle.json
       Linux:    ~/.kaggle/kaggle.json
       Mac:      ~/.kaggle/kaggle.json
5. Run this script:
       python scripts/download_dataset.py

Option B — Using opendatasets (interactive)
-------------------------------------------
1. pip install opendatasets
2. Run this script with --use-opendatasets flag
3. It will prompt you for username + API key interactively

Option C — Manual download (no credentials file needed)
--------------------------------------------------------
1. Go to: https://www.kaggle.com/c/home-credit-default-risk/data
2. Sign in and click "Download All" (or just download application_train.csv)
3. Unzip to: data/raw/
4. Verify: data/raw/application_train.csv exists
"""

import os
import sys
import zipfile
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------
# Project root is one level up from this script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

KAGGLE_COMPETITION = "home-credit-default-risk"
PRIMARY_FILE = "application_train.csv"
EXPECTED_ROWS = 307_511
EXPECTED_COLS = 122


def ensure_directories():
    """Create data directories if they don't exist."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[✓] Data directories ready:")
    print(f"    Raw:       {RAW_DATA_DIR}")
    print(f"    Processed: {PROCESSED_DATA_DIR}")


def check_existing():
    """Check if the dataset already exists."""
    target_path = RAW_DATA_DIR / PRIMARY_FILE
    if target_path.exists():
        size_mb = target_path.stat().st_size / (1024 * 1024)
        print(f"[✓] Dataset already exists: {target_path} ({size_mb:.1f} MB)")
        return True
    return False


# ===================================================================
# OPTION A: Kaggle CLI Download
# ===================================================================
def download_via_kaggle_cli():
    """
    Download using the official Kaggle CLI.

    Prerequisites:
        pip install kaggle
        Place kaggle.json in ~/.kaggle/kaggle.json

    Under the hood this runs:
        kaggle competitions download -c home-credit-default-risk -p data/raw/
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("[✗] kaggle package not installed. Run: pip install kaggle")
        return False

    # --- Credential Check ---
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print(f"[✗] Kaggle credentials not found at: {kaggle_json}")
        print("    → Go to https://www.kaggle.com/settings → API → Create New Token")
        print(f"    → Save the downloaded kaggle.json to: {kaggle_json}")
        return False

    print(f"[…] Authenticating with Kaggle API...")
    api = KaggleApi()
    api.authenticate()
    print(f"[✓] Authenticated successfully")

    print(f"[…] Downloading '{KAGGLE_COMPETITION}' dataset...")
    print(f"    Destination: {RAW_DATA_DIR}")
    print(f"    This may take 2-5 minutes depending on your connection...")

    api.competition_download_files(
        competition=KAGGLE_COMPETITION,
        path=str(RAW_DATA_DIR),
        quiet=False,
    )

    # --- Unzip if needed ---
    zip_path = RAW_DATA_DIR / f"{KAGGLE_COMPETITION}.zip"
    if zip_path.exists():
        print(f"[…] Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(RAW_DATA_DIR)
        zip_path.unlink()  # Remove zip after extraction
        print(f"[✓] Extracted and cleaned up zip file")

    return True


# ===================================================================
# OPTION B: opendatasets Download (interactive credentials)
# ===================================================================
def download_via_opendatasets():
    """
    Download using the opendatasets library.
    This will interactively prompt for Kaggle username + API key.

    Prerequisites:
        pip install opendatasets
    """
    try:
        import opendatasets as od
    except ImportError:
        print("[✗] opendatasets package not installed. Run: pip install opendatasets")
        return False

    dataset_url = f"https://www.kaggle.com/c/{KAGGLE_COMPETITION}"
    print(f"[…] Downloading from: {dataset_url}")
    print(f"    You will be prompted for your Kaggle username and API key.")
    print(f"    (Get your key at: https://www.kaggle.com/settings → API)\n")

    od.download(dataset_url, data_dir=str(RAW_DATA_DIR))

    return True


# ===================================================================
# Validation
# ===================================================================
def validate_dataset():
    """Validate the downloaded dataset has expected shape."""
    target_path = RAW_DATA_DIR / PRIMARY_FILE
    if not target_path.exists():
        print(f"[✗] {PRIMARY_FILE} not found in {RAW_DATA_DIR}")
        print(f"    Files present: {list(RAW_DATA_DIR.glob('*'))}")
        return False

    try:
        import pandas as pd
    except ImportError:
        print("[!] pandas not installed — skipping row/column validation")
        size_mb = target_path.stat().st_size / (1024 * 1024)
        print(f"[✓] File exists: {target_path} ({size_mb:.1f} MB)")
        return True

    print(f"[…] Validating {PRIMARY_FILE}...")
    df = pd.read_csv(target_path, nrows=5)  # Read just 5 rows for quick check
    n_cols = len(df.columns)

    # Full row count (read only the index)
    n_rows = sum(1 for _ in open(target_path, encoding="utf-8")) - 1  # minus header

    print(f"    Rows:    {n_rows:,}  (expected: {EXPECTED_ROWS:,})")
    print(f"    Columns: {n_cols}  (expected: {EXPECTED_COLS})")

    if "TARGET" not in df.columns:
        print(f"[✗] TARGET column missing! Available columns: {list(df.columns[:10])}...")
        return False

    print(f"[✓] TARGET column found ✓")

    if n_rows >= EXPECTED_ROWS * 0.95:  # Allow 5% tolerance
        print(f"[✓] Dataset validated successfully!")
        return True
    else:
        print(f"[!] Row count seems low — file may be truncated")
        return False


def print_manual_instructions():
    """Print instructions for manual download (Option C)."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║           MANUAL DOWNLOAD INSTRUCTIONS                      ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. Open in browser:                                         ║
║     https://www.kaggle.com/c/home-credit-default-risk/data  ║
║                                                              ║
║  2. Sign in with your Kaggle account                        ║
║     (create one free at kaggle.com if needed)               ║
║                                                              ║
║  3. Click "Download All" button                             ║
║     OR download just: application_train.csv                 ║
║                                                              ║
║  4. Unzip the downloaded file                               ║
║                                                              ║
║  5. Copy application_train.csv to:                          ║
║     {raw_dir}
║                                                              ║
║  6. Run this script again to validate:                      ║
║     python scripts/download_dataset.py --validate-only      ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""".format(raw_dir=RAW_DATA_DIR))


# ===================================================================
# CLI Entry Point
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Download Home Credit Default Risk dataset from Kaggle"
    )
    parser.add_argument(
        "--method",
        choices=["kaggle-cli", "opendatasets", "manual"],
        default="kaggle-cli",
        help="Download method (default: kaggle-cli)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing dataset, don't download",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if file already exists",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  FlowScore — Dataset Downloader")
    print("  Home Credit Default Risk (Kaggle)")
    print("=" * 60)
    print()

    ensure_directories()

    # Validate-only mode
    if args.validate_only:
        validate_dataset()
        return

    # Check existing
    if not args.force and check_existing():
        print("[i] Use --force to re-download")
        validate_dataset()
        return

    # Download
    success = False
    if args.method == "kaggle-cli":
        success = download_via_kaggle_cli()
    elif args.method == "opendatasets":
        success = download_via_opendatasets()
    elif args.method == "manual":
        print_manual_instructions()
        return

    if not success:
        print()
        print("[!] Automatic download failed. Falling back to manual instructions:")
        print_manual_instructions()
        return

    # Validate
    print()
    validate_dataset()

    print()
    print("=" * 60)
    print("  ✓ Dataset ready! Next step:")
    print("    python model/train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
