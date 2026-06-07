# FlowScore — Critical Pre-Deployment Audit
**Status: YELLOW 🟡 — Feature order verification needed before Render deployment**

Generated: June 7, 2025  
Repo: https://github.com/Tarang2005/flowscore  
User: Tarang Patra  

---

## 🚨 CRITICAL ISSUE DETECTED

**Problem:** You answered "no" to "Did you verify feature order is the same between training and inference?"

**Impact:** HIGH — This will cause model predictions to be garbage (wrong answers).

**How it breaks:**
```python
# Week 1 (training): Features in this order
['income_trend_6m', 'income_volatility', 'spending_to_income_ratio', 
 'avg_6m_income', 'latest_month_income', ...]  # 23 features

# Week 2 (inference): API featurizes in DIFFERENT order
['income_volatility', 'income_trend_6m', 'spending_to_income_ratio', ...]  
                 ^ DIFFERENT ORDER!

# Result: Model gets scrambled input → predicts garbage
# Priya's real score: 625. Predicts: 450 (completely wrong)
```

---

## IMMEDIATE ACTION REQUIRED

### Step 1: Verify Feature Order (DO THIS NOW)

In your code, find where you do this:

**In `model/feature_engineering.py` (training):**
```python
# What order do you CREATE features?
features_list = [
    'income_trend_6m',
    'income_volatility',
    'spending_to_income_ratio',
    ...
]
# SAVE THIS ORDER
with open('model/artifacts/feature_order.json', 'w') as f:
    json.dump(features_list, f)
```

**In `backend/utils.py` (inference):**
```python
# When you featurize a borrower profile, use EXACT same order
with open('model/artifacts/feature_order.json', 'r') as f:
    feature_order = json.load(f)

# Featurize in that order (not any other order)
borrower_features = [borrower.feature[name] for name in feature_order]
```

**Then verify:**
```python
# Both should print the same list
print("Training order:", features_list)
print("Inference order:", feature_order)
assert features_list == feature_order  # Should NOT fail
```

---

## AUDIT CHECKLIST — Answer ALL questions

Copy your actual code snippets below each question.

### Question 1: Feature Order Verification
**Question:** Show me the exact list of 23 features in your training code (`feature_engineering.py`)

```
[Paste the features list here]
```

**Answer:** ___________

---

### Question 2: Feature Order in API
**Question:** Show me how you featurize a borrower profile in `backend/utils.py` 

```python
# What I'm looking for:
def featurize_borrower(borrower_profile) -> np.array:
    # Is it in the SAME order as training?
    features = [
        borrower_profile.income_trend_6m,
        borrower_profile.income_volatility,
        ...
    ]
```

**Answer:** ___________

---

### Question 3: Model Artifacts
**Question:** List all `.pkl` files in your `model/artifacts/` folder

**Expected files:**
- [ ] `model.pkl` (the XGBoost model)
- [ ] `scaler.pkl` (StandardScaler)
- [ ] `explainer.pkl` (SHAP TreeExplainer)
- [ ] `feature_names.pkl` OR `feature_order.json` (list of 23 features in order)

**Your files:**
```
[Paste the list here]
```

**Answer:** ___________

---

### Question 4: SHAP Response Test
**Question:** When you called `POST /score` with Priya's data, paste the ACTUAL response JSON

**What I'm looking for:**
```json
{
  "flowscore": 625,
  "default_probability": 0.35,
  "shap_explanation": {
    "top_positive_factors": [...],
    "top_negative_factors": [...]
  },
  "coaching_tips": [...]
}
```

**Your response:**
```json
[Paste here]
```

**Answer:** ___________

---

### Question 5: Model Metrics on Real Data
**Question:** What AUC do you get when training on synthetic data vs. real Kaggle data?

- Synthetic (current): _____
- Real Kaggle (expected): _____

**Concern:** If it drops below 78%, we need to retune features.

**Answer:** ___________

---

### Question 6: Render Deployment Config
**Question:** Show me your `render.yaml` (or deployment config)

**What I'm looking for:**
```yaml
services:
  - type: web
    name: flowscore-api
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port 8000
    envVars:
      - key: PYTHONUNBUFFERED
        value: 1
```

**Your config:**
```yaml
[Paste here]
```

**Answer:** ___________

---

### Question 7: .pkl Files in Git
**Question:** Are the `.pkl` files actually committed to GitHub?

- [ ] Yes, I see them in the `model/artifacts/` folder on GitHub
- [ ] No, I forgot to commit them
- [ ] I added them to `.gitignore` (will break on Render)

**What I see on GitHub:**
- `model/artifacts/model.pkl` → [File size: ___ MB]
- `model/artifacts/scaler.pkl` → [File size: ___ MB]
- `model/artifacts/explainer.pkl` → [File size: ___ MB]

**Answer:** ___________

---

## VALIDATION RESULTS

Based on your answers above, rate each area:

| Area | Status | Notes |
|------|--------|-------|
| Feature Order | 🔴 RED | NOT VERIFIED |
| SHAP Generation | 🟡 YELLOW | Need actual response |
| Model Metrics | 🟡 YELLOW | Unknown on real data |
| Deployment Config | ❓ UNKNOWN | Need to see `render.yaml` |
| Git Artifacts | ❓ UNKNOWN | Need confirmation |

**Overall Readiness:** 🔴 RED — **DO NOT DEPLOY UNTIL FEATURE ORDER IS VERIFIED**

---

## FIX PRIORITY (In this order)

### 1. CRITICAL (Do today)
- [ ] Verify feature order matches between training and inference
- [ ] Test `POST /score` with Priya, confirm she scores ~625
- [ ] If score is wrong (not 600-650), debug feature order immediately

### 2. HIGH (Do before Render deploy)
- [ ] Confirm all 4 `.pkl` files are committed to GitHub
- [ ] Create `render.yaml` with correct config
- [ ] Test Vercel + Render deployment URLs locally (curl the live API)

### 3. MEDIUM (Do this week)
- [ ] Retrain on real Kaggle data
- [ ] Verify AUC stays ≥78%
- [ ] Update `.pkl` files on GitHub with new trained artifacts

### 4. LOW (Nice to have)
- [ ] Add error handling for missing `.pkl` files
- [ ] Add logging to track inference time
- [ ] Optimize SHAP generation if >500ms

---

## HOW TO FIX FEATURE ORDER BUG (If it exists)

**If you find they're different:**

### Option A: Use JSON to enforce order (RECOMMENDED)

```python
# In model/train_model.py (end of training)
import json

feature_order = [
    'income_trend_6m',
    'income_volatility',
    'spending_to_income_ratio',
    # ... all 23 in order
]

with open('model/artifacts/feature_order.json', 'w') as f:
    json.dump(feature_order, f)

# Also save which features were used
with open('model/artifacts/feature_metadata.json', 'w') as f:
    json.dump({
        'n_features': 23,
        'feature_order': feature_order,
        'training_auc': 0.84,
        'training_date': '2025-06-07'
    }, f)
```

```python
# In backend/utils.py (inference)
import json
import numpy as np

with open('model/artifacts/feature_order.json', 'r') as f:
    feature_order = json.load(f)

def featurize_borrower(borrower_profile: BorrowerProfile) -> np.ndarray:
    """Featurize in EXACT order that model was trained on."""
    features = []
    for feature_name in feature_order:
        value = getattr(borrower_profile.calculated_features, feature_name)
        features.append(value)
    return np.array(features).reshape(1, -1)  # Reshape for model.predict()
```

### Option B: Use a DataFrame (SAFEST)

```python
# In inference
def featurize_borrower(borrower_profile: BorrowerProfile) -> pd.DataFrame:
    """Create DataFrame with correct column order."""
    df = pd.DataFrame({
        'income_trend_6m': [borrower_profile.calculated_features.income_trend_6m],
        'income_volatility': [borrower_profile.calculated_features.income_volatility],
        # ... all 23 features
    })
    
    # Reorder columns to match training
    df = df[feature_order]
    return df
```

---

## NEXT STEPS

1. **Answer all 7 questions above** (copy your code)
2. **I'll verify** and give you GREEN/YELLOW/RED status
3. **If GREEN** → Proceed to Render deployment
4. **If RED** → Fix the feature order bug (takes 1 hour)
5. **Then deploy**

---

## DEPLOYMENT CHECKLIST (After fixes)

- [ ] Feature order verified and locked
- [ ] All `.pkl` files committed to GitHub
- [ ] `render.yaml` created with correct config
- [ ] Local testing confirms Priya scores ~625, Arjun ~750
- [ ] GitHub repo is public
- [ ] `requirements.txt` has all dependencies (fastapi, xgboost, shap, pandas, etc.)

Once ✅ all checked → You can deploy to Render/Vercel with confidence.

---

**Deadline:** Complete by EOD today. You have 24 hours before demo day.  
**Questions?** Ask before deploying. A 10-minute conversation now saves 4 hours of debugging later.

