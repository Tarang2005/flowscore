import os
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

def generate_mock_data(num_rows=10000):
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    
    # Target distribution (92% safe, 8% default)
    targets = np.random.choice([0, 1], size=num_rows, p=[0.92, 0.08])
    
    # Base features
    sk_id_curr = np.arange(100000, 100000 + num_rows)
    
    # Generate strongly correlated features
    amt_income_total = np.zeros(num_rows)
    amt_annuity = np.zeros(num_rows)
    ext_source_2 = np.zeros(num_rows)
    ext_source_3 = np.zeros(num_rows)
    days_employed = np.zeros(num_rows)
    days_registration = np.zeros(num_rows)
    def_30 = np.zeros(num_rows)
    def_60 = np.zeros(num_rows)
    amt_req = np.zeros(num_rows)
    days_birth = np.zeros(num_rows)
    days_id = np.zeros(num_rows)
    
    for i in range(num_rows):
        if targets[i] == 0:
            # Good borrower (like Arjun/Priya)
            amt_income_total[i] = np.random.uniform(30000, 200000)
            amt_annuity[i] = amt_income_total[i] * np.random.uniform(0.1, 0.4) # Low spending ratio
            ext_source_2[i] = np.random.uniform(0.6, 0.9) # High rating
            ext_source_3[i] = np.random.uniform(0.5, 0.9) # Good trend
            days_employed[i] = -np.random.randint(500, 3000) # Long employment
            days_registration[i] = -np.random.randint(500, 5000)
            def_30[i] = 0 # No late payments
            def_60[i] = 0
            amt_req[i] = np.random.randint(0, 2)
            days_birth[i] = -np.random.randint(10000, 20000)
            days_id[i] = -np.random.randint(2000, 5000)
        else:
            # Bad borrower (like Meera)
            amt_income_total[i] = np.random.uniform(15000, 80000)
            amt_annuity[i] = amt_income_total[i] * np.random.uniform(0.6, 1.2) # High spending ratio
            ext_source_2[i] = np.random.uniform(0.1, 0.5) # Low rating
            ext_source_3[i] = np.random.uniform(0.1, 0.4) # Bad trend
            days_employed[i] = -np.random.randint(30, 200) # Short employment
            days_registration[i] = -np.random.randint(30, 500)
            def_30[i] = np.random.randint(1, 5) # Late payments
            def_60[i] = np.random.randint(0, 3)
            amt_req[i] = np.random.randint(2, 6)
            days_birth[i] = -np.random.randint(7000, 15000)
            days_id[i] = -np.random.randint(100, 1000)

    data = {
        "SK_ID_CURR": sk_id_curr,
        "TARGET": targets,
        "AMT_INCOME_TOTAL": amt_income_total,
        "EXT_SOURCE_2": ext_source_2,
        "EXT_SOURCE_3": ext_source_3,
        "DAYS_EMPLOYED": days_employed,
        "DAYS_REGISTRATION": days_registration,
        "AMT_ANNUITY": amt_annuity,
        "DEF_30_CNT_SOCIAL_CIRCLE": def_30,
        "DEF_60_CNT_SOCIAL_CIRCLE": def_60,
        "AMT_REQ_CREDIT_BUREAU_YEAR": amt_req,
        "DAYS_BIRTH": days_birth,
        "DAYS_ID_PUBLISH": days_id,
    }
    
    df = pd.DataFrame(data)
    output_path = RAW_DATA_DIR / "application_train.csv"
    df.to_csv(output_path, index=False)
    print(f"Generated {num_rows} rows of highly correlated mock data at {output_path}")

if __name__ == "__main__":
    generate_mock_data()
