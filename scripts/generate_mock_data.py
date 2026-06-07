import os
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

def generate_mock_data(num_rows=5000):
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(42)
    
    data = {
        "SK_ID_CURR": np.arange(100000, 100000 + num_rows),
        "TARGET": np.random.choice([0, 1], size=num_rows, p=[0.92, 0.08]), # 8% default rate
        "AMT_INCOME_TOTAL": np.random.lognormal(mean=11.5, sigma=0.8, size=num_rows).clip(50000, 2000000),
        "EXT_SOURCE_2": np.random.uniform(0.1, 0.9, size=num_rows),
        "EXT_SOURCE_3": np.random.uniform(0.1, 0.9, size=num_rows),
        "DAYS_EMPLOYED": -np.random.randint(30, 3650, size=num_rows),
        "DAYS_REGISTRATION": -np.random.randint(100, 10000, size=num_rows),
        "AMT_ANNUITY": np.random.lognormal(mean=10.0, sigma=0.5, size=num_rows).clip(5000, 100000),
        "DEF_30_CNT_SOCIAL_CIRCLE": np.random.poisson(lam=0.2, size=num_rows),
        "DEF_60_CNT_SOCIAL_CIRCLE": np.random.poisson(lam=0.05, size=num_rows),
        "AMT_REQ_CREDIT_BUREAU_YEAR": np.random.poisson(lam=1.5, size=num_rows),
        "DAYS_BIRTH": -np.random.randint(7000, 25000, size=num_rows),
        "DAYS_ID_PUBLISH": -np.random.randint(100, 5000, size=num_rows),
    }
    
    # Introduce some nulls to test imputation
    for col in ["EXT_SOURCE_2", "EXT_SOURCE_3", "AMT_ANNUITY"]:
        mask = np.random.random(num_rows) < 0.1
        data[col] = np.where(mask, np.nan, data[col])
        
    df = pd.DataFrame(data)
    
    output_path = RAW_DATA_DIR / "application_train.csv"
    df.to_csv(output_path, index=False)
    print(f"Generated {num_rows} rows of mock data at {output_path}")

if __name__ == "__main__":
    generate_mock_data()
