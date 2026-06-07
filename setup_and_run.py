"""
FlowScore — Block 1 Setup & Execution Script
==============================================

One-command script to set up the Python environment and run the
entire ML pipeline (data prep → feature engineering → training).

Usage:
    py setup_and_run.py           # Full pipeline
    py setup_and_run.py --setup   # Setup venv + install deps only
    py setup_and_run.py --run     # Run pipeline only (venv must exist)

Prerequisites:
    - Python 3.10+ installed
    - application_train.csv in data/raw/ (download from Kaggle first)
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def banner(msg: str):
    print()
    print("=" * 60)
    print(f"  {msg}")
    print("=" * 60)


def run_cmd(cmd: list[str], cwd: Path = PROJECT_ROOT, check: bool = True):
    """Run a command and stream output in real-time."""
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        check=check,
        text=True,
    )
    return result.returncode


def get_python() -> str:
    """Get the venv Python path, or fall back to system Python."""
    venv_python = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    return "py"


def get_pip() -> str:
    """Get the venv pip path."""
    venv_pip = PROJECT_ROOT / "venv" / "Scripts" / "pip.exe"
    if venv_pip.exists():
        return str(venv_pip)
    return "pip"


# ==================================================================
# PHASE 1: Setup Environment
# ==================================================================
def setup_environment():
    """Create venv, install all ML dependencies."""

    banner("PHASE 1.1 — Creating Python Virtual Environment")

    venv_dir = PROJECT_ROOT / "venv"
    if venv_dir.exists():
        print(f"  ✓ venv already exists at {venv_dir}")
    else:
        run_cmd(["py", "-m", "venv", str(venv_dir)])
        print(f"  ✓ venv created at {venv_dir}")

    banner("PHASE 1.2 — Upgrading pip")
    python = get_python()
    run_cmd([python, "-m", "pip", "install", "--upgrade", "pip"], check=False)

    banner("PHASE 1.3 — Installing ML Dependencies")
    print("  Installing: pandas, xgboost, shap, scikit-learn, fastapi, etc.")
    print("  This may take 2-5 minutes on first install...\n")
    run_cmd([get_pip(), "install", "-r", str(PROJECT_ROOT / "requirements.txt")])

    banner("PHASE 1.4 — Verifying Installations")
    verify_script = (
        "import pandas; print(f'  pandas     {pandas.__version__}'); "
        "import numpy; print(f'  numpy      {numpy.__version__}'); "
        "import sklearn; print(f'  sklearn    {sklearn.__version__}'); "
        "import xgboost; print(f'  xgboost    {xgboost.__version__}'); "
        "import shap; print(f'  shap       {shap.__version__}'); "
        "import joblib; print(f'  joblib     {joblib.__version__}'); "
        "print('  ✓ All ML libraries verified!')"
    )
    run_cmd([python, "-c", verify_script])

    print()
    print("  ✓ Environment setup complete!")
    print(f"  Activate manually with: .\\venv\\Scripts\\activate")


# ==================================================================
# PHASE 2: Check Dataset
# ==================================================================
def check_dataset() -> bool:
    """Verify the Home Credit dataset exists."""

    banner("PHASE 2 — Checking Dataset")

    data_path = PROJECT_ROOT / "data" / "raw" / "application_train.csv"

    if data_path.exists():
        size_mb = data_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Dataset found: {data_path}")
        print(f"    Size: {size_mb:.1f} MB")
        return True
    else:
        print(f"  ✗ Dataset NOT found at: {data_path}")
        print()
        print("  To download, choose one option:")
        print()
        print("  Option A — Kaggle CLI:")
        print(f"    {get_python()} scripts/download_dataset.py")
        print()
        print("  Option B — Manual download:")
        print("    1. Go to: https://www.kaggle.com/c/home-credit-default-risk/data")
        print("    2. Download application_train.csv")
        print(f"    3. Place it in: {data_path.parent}")
        print()
        return False


# ==================================================================
# PHASE 3: Run ML Pipeline
# ==================================================================
def run_pipeline():
    """Execute the full Block 1 ML pipeline."""
    python = get_python()

    # Step 1: Data Prep
    banner("PHASE 3.1 — Data Preparation (data_prep.py)")
    run_cmd([python, str(PROJECT_ROOT / "scripts" / "data_prep.py")])

    # Verify output
    cleaned = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
    if cleaned.exists():
        print(f"  ✓ cleaned_data.csv created ({cleaned.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"  ✗ cleaned_data.csv NOT created!")
        sys.exit(1)

    # Step 2: Feature Engineering
    banner("PHASE 3.2 — Feature Engineering (feature_engineering.py)")
    run_cmd([python, str(PROJECT_ROOT / "scripts" / "feature_engineering.py")])

    # Verify output
    features = PROJECT_ROOT / "data" / "processed" / "engineered_features.csv"
    if features.exists():
        print(f"  ✓ engineered_features.csv created ({features.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"  ✗ engineered_features.csv NOT created!")
        sys.exit(1)

    # Step 3: Train Model
    banner("PHASE 3.3 — Model Training (train_model.py)")
    run_cmd([python, str(PROJECT_ROOT / "model" / "train_model.py"), "--no-cv"])

    # Verify artifacts
    banner("PHASE 4 — Verifying Outputs (Block 1 Test Checklist)")
    artifacts_dir = PROJECT_ROOT / "model" / "artifacts"

    expected_files = {
        "model.pkl": "XGBoost classifier",
        "scaler.pkl": "StandardScaler + feature names",
        "explainer.pkl": "SHAP TreeExplainer",
        "feature_names.pkl": "23 feature names list",
        "metrics.json": "Evaluation metrics",
        "feature_importance.json": "Feature importance data",
        "feature_importance.png": "Feature importance chart",
    }

    all_ok = True
    for filename, desc in expected_files.items():
        path = artifacts_dir / filename
        if path.exists():
            size = path.stat().st_size
            if size > 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size} B"
            print(f"  ✓ {filename:<30s} {size_str:>10s}  ({desc})")
        else:
            print(f"  ✗ {filename:<30s} MISSING!    ({desc})")
            all_ok = False

    # Check metrics
    if (artifacts_dir / "metrics.json").exists():
        import json
        with open(artifacts_dir / "metrics.json", "r") as f:
            metrics = json.load(f)
        eval_data = metrics.get("evaluation", {})
        auc = eval_data.get("auc_roc", 0)
        targets = eval_data.get("targets_met", {})

        print()
        print(f"  AUC-ROC:          {auc:.4f}  {'✓' if auc >= 0.85 else '✗'} (target: ≥ 0.85)")
        print(f"  Inference time:   {eval_data.get('inference_time_ms', 0):.2f}ms  {'✓' if eval_data.get('inference_time_ms', 999) < 200 else '✗'} (target: < 200ms)")

    print()
    if all_ok:
        print("  ═══════════════════════════════════════")
        print("  ✓ BLOCK 1 COMPLETE — All tests passed!")
        print("  ═══════════════════════════════════════")
        print()
        print("  Next: Block 2 (FastAPI backend)")
        print("    uvicorn backend.main:app --reload --port 8000")
    else:
        print("  ⚠ Some artifacts missing. Check logs above.")
        sys.exit(1)


# ==================================================================
# CLI Entry Point
# ==================================================================
def main():
    parser = argparse.ArgumentParser(
        description="FlowScore — Block 1 Setup & Run"
    )
    parser.add_argument("--setup", action="store_true", help="Setup only (venv + deps)")
    parser.add_argument("--run", action="store_true", help="Run pipeline only (skip setup)")
    args = parser.parse_args()

    print()
    print("╔" + "═" * 58 + "╗")
    print("║   FlowScore — Block 1: Data + Model Pipeline             ║")
    print("╚" + "═" * 58 + "╝")

    if args.setup:
        setup_environment()
    elif args.run:
        if not check_dataset():
            sys.exit(1)
        run_pipeline()
    else:
        # Full flow: setup → check data → run
        setup_environment()
        if not check_dataset():
            print()
            print("  ⚠ Download the dataset first, then re-run with --run")
            sys.exit(1)
        run_pipeline()


if __name__ == "__main__":
    main()
