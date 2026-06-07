# FlowScore — Project Context Document
**AI Credit Scoring for Gig Workers | College Hackathon Build**

---

## 1. PROJECT OVERVIEW

### Problem Statement
70M+ gig workers (freelancers, delivery partners, rideshare drivers) are systematically rejected for loans because traditional FICO credit scoring requires W-2 employment history, stable monthly income, and formal credit bureau reporting. These workers earn well and pay their bills on time, but the *format* of their income (irregular, multi-platform) makes them invisible to legacy underwriting systems.

- **58%** of gig workers seek emergency loans quarterly
- **73%** say their irregular income blocks them from traditional loans
- **Market opportunity**: India has 4.5M gig workers; globally 70M+

### Solution
**FlowScore** — an AI credit engine that scores gig workers based on:
1. **Alternative data**: Real income streams (Razorpay, Upwork, Fiverr, UPI transactions)
2. **Dynamic ML model**: Income velocity, volatility, trend direction, spending discipline
3. **Explainability**: SHAP-based breakdown showing exactly why the score is what it is
4. **Dual revenue**: B2B API for lenders + B2C coaching for borrowers

### Hackathon Scope (4 weeks, 20 hrs/week)
- Working ML model (XGBoost trained on real data)
- REST API endpoint that scores a borrower profile
- React dashboard showing score + SHAP breakdown + coaching tips
- Mock data pipeline (simulates Razorpay webhook ingestion)
- Live demo with 3 persona scenarios

---

## 2. TECHNICAL ARCHITECTURE

### System Architecture
```
┌─────────────────────────────────────────────────┐
│         BORROWER FACING (React Frontend)        │
│  - FlowScore gauge, income chart, coaching tips │
└────────────────────┬────────────────────────────┘
                     │ HTTP/REST
┌────────────────────▼────────────────────────────┐
│        FastAPI Backend (Python/FastAPI)         │
│  - POST /score — returns score + SHAP factors   │
│  - POST /ingest — mock Razorpay webhook         │
│  - GET /factors/:borrower_id — SHAP breakdown   │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│      ML Scoring Engine (Python/XGBoost)         │
│  - Trained XGBoost model (85%+ AUC)             │
│  - SHAP force plots + waterfall plots           │
│  - Real-time inference (<200ms)                 │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│    Feature Store / Mock Data Pipeline           │
│  - Simulated Razorpay transaction history       │
│  - Simulated UPI spending patterns              │
│  - Income velocity calculations                 │
└─────────────────────────────────────────────────┘
```

### Technology Stack
- **ML Model**: Python 3.10+ | XGBoost | SHAP
- **Backend API**: FastAPI | Pydantic | Joblib (model serialization)
- **Frontend**: React 18+ | Recharts (charting) | Tailwind CSS
- **Data Processing**: Pandas | NumPy | Scikit-learn
- **Deployment**: Vercel (frontend) | Render/Railway (API)
- **Database**: Optional (use in-memory for MVP, SQLite if needed)

---

## 3. DATA SPECIFICATIONS

### Borrower Profile Schema
```json
{
  "borrower_id": "UNIQUE_ID",
  "name": "Priya Sharma",
  "age": 28,
  "phone": "+91-XXXXXXXXXX",
  
  "income_data": {
    "platforms": [
      {
        "platform": "swiggy_partner",
        "monthly_earnings_last_6m": [35000, 38000, 42000, 45000, 48000, 52000],
        "platform_rating": 4.8,
        "platform_account_age_months": 24
      },
      {
        "platform": "upwork",
        "monthly_earnings_last_6m": [15000, 18000, 20000, 22000, 25000, 28000],
        "platform_rating": 4.9,
        "platform_account_age_months": 18
      }
    ],
    "total_monthly_income_current": 80000,
    "avg_monthly_income_6m": 65000
  },
  
  "spending_data": {
    "avg_monthly_spending": 45000,
    "spending_categories": {
      "food": 12000,
      "transport": 8000,
      "utilities": 5000,
      "personal": 20000
    },
    "late_payments_count_6m": 0,
    "missed_payments_count_6m": 0
  },
  
  "credit_profile": {
    "existing_loans": 0,
    "total_debt": 0,
    "credit_inquiries_6m": 0
  },
  
  "calculated_features": {
    "income_volatility": 0.18,      // std dev of 6m income / mean
    "income_trend": 1.48,            // (latest_3m_avg - first_3m_avg) / first_3m_avg
    "spending_to_income_ratio": 0.56,
    "platform_count": 2,
    "days_active_primary": 720,
    "income_velocity_3m": 0.32       // growth rate last 3 months
  }
}
```

### Model Input Features (23 total)
1. **Income Features** (8):
   - avg_6m_income
   - income_std_dev
   - income_trend_6m
   - latest_month_income
   - income_velocity_3m
   - platform_count
   - primary_platform_earnings_pct
   - secondary_platform_earnings_pct

2. **Platform/Account Features** (6):
   - avg_platform_rating
   - days_on_primary_platform
   - days_on_secondary_platform
   - platform_account_variance
   - account_age_months
   - total_platforms

3. **Spending Features** (4):
   - spending_to_income_ratio
   - avg_monthly_spending
   - spending_volatility
   - spending_trend

4. **Credit History Features** (3):
   - late_payments_count_6m
   - missed_payments_count_6m
   - existing_loan_count

5. **Demographic Features** (2):
   - age
   - account_tenure_months

### Training Data
- **Source**: Kaggle Home Credit Default Risk dataset (307K rows, 122 features)
- **Preprocessing**:
  1. Handle missing values (median imputation for numeric, mode for categorical)
  2. Feature engineering: Add synthetic gig-specific features
  3. Normalize numeric features (StandardScaler)
  4. One-hot encode categorical features
  5. Drop features with >50% missing
  6. Select top 23 features by feature importance

- **Target Variable**: Default (0 = repaid on time, 1 = defaulted)
- **Train/Test Split**: 70/30
- **Target Distribution**: Imbalanced (92% repaid, 8% defaulted)
  - Use class_weight='balanced' in XGBoost

### Sample Borrower Data (3 Personas for Demo)

**Persona 1: Priya (Swiggy Driver)**
```json
{
  "borrower_id": "priya_001",
  "name": "Priya Sharma",
  "age": 28,
  "income_data": {
    "platforms": [
      {"platform": "swiggy_partner", "monthly_earnings_last_6m": [35000, 38000, 42000, 45000, 48000, 52000], "platform_rating": 4.8}
    ],
    "total_monthly_income_current": 52000,
    "avg_monthly_income_6m": 43333
  },
  "spending_data": {"avg_monthly_spending": 28000, "late_payments_count_6m": 0},
  "calculated_features": {"income_volatility": 0.18, "income_trend": 0.48, "spending_to_income_ratio": 0.54}
}
```

**Persona 2: Arjun (Upwork Developer)**
```json
{
  "borrower_id": "arjun_001",
  "name": "Arjun Mehta",
  "age": 32,
  "income_data": {
    "platforms": [
      {"platform": "upwork", "monthly_earnings_last_6m": [120000, 135000, 145000, 155000, 165000, 180000], "platform_rating": 4.95},
      {"platform": "toptal", "monthly_earnings_last_6m": [25000, 28000, 32000, 35000, 38000, 42000], "platform_rating": 4.9}
    ],
    "total_monthly_income_current": 222000,
    "avg_monthly_income_6m": 160000
  },
  "spending_data": {"avg_monthly_spending": 75000, "late_payments_count_6m": 0},
  "calculated_features": {"income_volatility": 0.12, "income_trend": 0.48, "spending_to_income_ratio": 0.34}
}
```

**Persona 3: Meera (Fiverr Designer)**
```json
{
  "borrower_id": "meera_001",
  "name": "Meera Patel",
  "age": 26,
  "income_data": {
    "platforms": [
      {"platform": "fiverr", "monthly_earnings_last_6m": [18000, 15000, 22000, 25000, 28000, 35000], "platform_rating": 4.7}
    ],
    "total_monthly_income_current": 35000,
    "avg_monthly_income_6m": 23833
  },
  "spending_data": {"avg_monthly_spending": 22000, "late_payments_count_6m": 1},
  "calculated_features": {"income_volatility": 0.35, "income_trend": 0.48, "spending_to_income_ratio": 0.92}
}
```

---

## 4. ML MODEL SPECIFICATIONS

### Model Type
**Gradient Boosted Decision Tree** (XGBoost)

### Target Variable
`default` (binary classification)
- 0 = Repaid on time (negative class, ~92%)
- 1 = Defaulted (positive class, ~8%)

### Model Performance Target
- **AUC-ROC**: ≥0.85
- **Precision (recall=0.8)**: ≥0.50
- **Inference Time**: <200ms per borrower

### Hyperparameters (Baseline)
```python
xgb_params = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'scale_pos_weight': 10,  # Handle class imbalance
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'random_state': 42
}
```

### Output: FlowScore (0–850)
Normalize XGBoost probability output to a FICO-like score:
```python
def get_flowscore(xgb_pred_probability: float) -> int:
    # xgb_pred_probability is between 0 and 1
    # Convert to 300–850 scale (lower default risk = higher score)
    flowscore = 300 + (1 - xgb_pred_probability) * 550
    return int(flowscore)
```

### SHAP Explainability
- Generate SHAP force plot for every prediction
- Return top 5 features driving the score (positive and negative)
- Generate coaching tips based on SHAP factors:
  - If income_trend is negative: "Grow your income steadily — increases your score"
  - If spending_to_income_ratio is high: "Reduce discretionary spending to improve score"
  - If income_volatility is high: "Diversify across platforms for stability"

### Feature Importance (Expected, from Home Credit)
Top features typically include:
1. EXT_SOURCE_2 (~15%)
2. EXT_SOURCE_3 (~12%)
3. income_trend_6m (~8%)
4. spending_to_income_ratio (~7%)
5. income_volatility (~6%)
... (18 more)

---

## 5. API SPECIFICATIONS

### Base URL
Development: `http://localhost:8000`
Production: `https://flowscore-api.onrender.com` (or similar)

### Endpoints

#### 1. POST `/score`
**Score a borrower's creditworthiness**

Request:
```json
{
  "borrower_id": "priya_001",
  "income_data": {...},
  "spending_data": {...},
  "credit_profile": {...},
  "calculated_features": {...}
}
```

Response (200 OK):
```json
{
  "borrower_id": "priya_001",
  "flowscore": 625,
  "default_probability": 0.35,
  "confidence_interval": [610, 640],
  "risk_category": "medium",
  
  "shap_explanation": {
    "top_positive_factors": [
      {"feature": "income_trend_6m", "contribution": 45.2, "value": 0.48},
      {"feature": "platform_rating_avg", "contribution": 32.1, "value": 4.8}
    ],
    "top_negative_factors": [
      {"feature": "spending_to_income_ratio", "contribution": -28.5, "value": 0.54},
      {"feature": "income_volatility", "contribution": -12.3, "value": 0.18}
    ]
  },
  
  "coaching_tips": [
    "Your income grew 48% over 6 months — excellent trajectory! +45 points",
    "Your spending is 54% of income. Reducing discretionary spending by 10% could add +18 points",
    "Maintain your high platform rating. Every 0.1pt = +2 score points"
  ],
  
  "model_metadata": {
    "model_version": "v1.0",
    "prediction_timestamp": "2025-06-06T10:32:45Z",
    "feature_count": 23
  }
}
```

#### 2. POST `/ingest`
**Mock Razorpay/UPI transaction ingestion**

Request:
```json
{
  "borrower_id": "priya_001",
  "event": "payment.authorized",
  "transaction": {
    "id": "txn_123456",
    "amount": 45000,
    "currency": "INR",
    "timestamp": "2025-06-01T15:30:00Z",
    "platform": "swiggy_partner",
    "category": "earnings"
  }
}
```

Response (200 OK):
```json
{
  "status": "ingested",
  "borrower_id": "priya_001",
  "updated_income_current": 52000,
  "score_changed": false,
  "new_flowscore": 625
}
```

#### 3. GET `/borrower/:borrower_id`
**Fetch complete borrower profile + current score**

Response (200 OK):
```json
{
  "borrower_id": "priya_001",
  "profile": {...},
  "flowscore": 625,
  "score_history": [
    {"date": "2025-05-01", "score": 580},
    {"date": "2025-05-15", "score": 605},
    {"date": "2025-06-01", "score": 625}
  ]
}
```

#### 4. GET `/health`
**API health check**

Response (200 OK):
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "v1.0"
}
```

### Error Responses

```json
{
  "error": "Invalid borrower data",
  "detail": "Missing required fields: income_data.platforms",
  "status_code": 400
}
```

---

## 6. FRONTEND (REACT) SPECIFICATIONS

### Pages/Screens

#### 1. Dashboard (Main)
**Components**:
- **Header**: FlowScore branding, user name/ID
- **Score Gauge**: Large circular gauge (0–850 scale)
  - Green (700+), Yellow (600-699), Red (<600)
  - Shows current score + 1-month change
- **Income Trend Chart**: Line chart of 6-month income
  - X: Month | Y: Amount (₹)
- **SHAP Breakdown**: Horizontal bar chart
  - Top 5 positive factors (green) + top 3 negative factors (red)
  - Show feature name + contribution value
- **Coaching Tips**: 3 actionable tips in cards
  - Each tip shows estimated score impact (+X points)
- **Loan Pre-Approval Section**: "You qualify for pre-approved loans"
  - Mock loan offers from 3 lenders

#### 2. Lender View (Secondary Tab)
**Components**:
- **Borrower Summary**: Compact card with ID, score, risk category
- **SHAP JSON Output**: Show raw API response in structured format
- **Loan Decision**: Would approve/reject + reason

#### 3. Upload/Input Screen (Optional)
**Components**:
- JSON input form to test custom borrower profiles
- Preloaded buttons for 3 personas (Priya, Arjun, Meera)

### Design System
- **Color Palette**:
  - Primary: #185FA5 (blue)
  - Success: #3B6D11 (green for high scores)
  - Warning: #854f0b (amber for medium)
  - Danger: #A32D2D (red for low scores)
- **Typography**: System fonts, 16px base, 1.5 line height
- **Spacing**: 8px base unit (8, 16, 24, 32px gaps)
- **Buttons**: Tailwind `bg-blue-600 hover:bg-blue-700`, rounded corners

### Data Flow
1. User lands on dashboard
2. Frontend loads Priya's profile by default (or via URL param)
3. Frontend calls `GET /borrower/priya_001`
4. Backend returns full profile + current score
5. Frontend renders score gauge + charts
6. Optional: User can input a custom borrower JSON → calls `POST /score` → displays results

---

## 7. DEMO PERSONAS

### Persona 1: Priya Sharma (Swiggy Delivery Partner)
- **Income**: ₹52K/month (Swiggy only)
- **Income Trend**: +48% over 6 months (growth story)
- **Spending**: ₹28K/month (54% of income)
- **Platform Rating**: 4.8/5.0
- **Expected Score**: 625–640 (medium-good)
- **Coaching Angle**: "Growing income, but watch your spending"
- **Why it works**: Shows an underserved, informal worker with real growth trajectory

### Persona 2: Arjun Mehta (Upwork Developer)
- **Income**: ₹222K/month (Upwork + Toptal)
- **Income Trend**: +48% over 6 months (strong)
- **Spending**: ₹75K/month (34% of income)
- **Platform Ratings**: 4.95/5.0, 4.9/5.0
- **Expected Score**: 750–770 (excellent)
- **Coaching Angle**: "High earner, very stable, low risk"
- **Why it works**: Shows a high-income freelancer who would be rejected by FICO but is clearly creditworthy

### Persona 3: Meera Patel (Fiverr Designer)
- **Income**: ₹35K/month (Fiverr only)
- **Income Trend**: +48% over 6 months (growth)
- **Spending**: ₹22K/month (92% of income — high)
- **Platform Rating**: 4.7/5.0
- **Late Payments**: 1 in last 6 months
- **Expected Score**: 550–570 (below average)
- **Coaching Angle**: "Growing but needs to build emergency fund, reduce spending"
- **Why it works**: Shows a borderline case — growing income but poor spending discipline + payment history

---

## 8. HACKATHON WINNING STRATEGY

### Demo Flow (2 min)
1. **Hook (15s)**: "70 million gig workers are rejected for loans even though they earn well"
2. **Demo Priya (30s)**: Load her profile → show score 625 → highlight "income grew 48%"
3. **Demo Arjun (30s)**: Load his profile → show score 750+ → explain "traditional FICO would reject him"
4. **Explain SHAP (20s)**: Show the breakdown — judges see you know why decisions are made
5. **Show API (15s)**: Show curl request + JSON response — prove it's real
6. **Pitch (10s)**: "This is what KarmaLife built and raised $8M on. We're starting here."

### Judges' Checklist
- ✅ Real problem (70M workers, 73% rejected)
- ✅ Live working demo (not slides)
- ✅ ML is real (XGBoost + SHAP, trained model)
- ✅ Can be a startup (B2B + B2C revenue model)
- ✅ Code is clean (GitHub repo, README, architecture doc)
- ✅ Deployed live (Vercel + Render, not localhost)

### Repository Structure
```
flowscore/
├── README.md                      # Project overview + setup
├── FLOWSCORE_PROJECT_CONTEXT.md  # This file
├── data/
│   ├── home_credit_train.csv     # Training dataset (download from Kaggle)
│   └── personas.json             # Sample borrower profiles
├── model/
│   ├── train.py                  # Data loading, preprocessing, XGBoost training
│   ├── model.pkl                 # Serialized XGBoost model
│   ├── features.pkl              # Feature list + scaler
│   └── shap_explainer.pkl        # SHAP explainer object
├── backend/
│   ├── main.py                   # FastAPI app
│   ├── schemas.py                # Pydantic models for request/response
│   ├── utils.py                  # Feature engineering, SHAP generation
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx               # Main React app
│   │   ├── components/
│   │   │   ├── ScoreGauge.jsx
│   │   │   ├── IncomeChart.jsx
│   │   │   ├── SHAPBreakdown.jsx
│   │   │   └── CoachingTips.jsx
│   │   └── pages/
│   │       ├── Dashboard.jsx
│   │       └── LenderView.jsx
│   ├── package.json
│   └── tailwind.config.js
├── .env.example
└── .gitignore
```

---

## 9. TIMELINE & MILESTONES

### Week 1: Data + Model (20 hours)
- [ ] Download Home Credit dataset from Kaggle
- [ ] EDA: Understand distributions, missing values, correlations
- [ ] Feature engineering: Create 23 features from raw data
- [ ] Train XGBoost baseline model (target: 85% AUC)
- [ ] Integrate SHAP explainer
- [ ] Save model artifacts (model.pkl, scaler.pkl, explainer.pkl)
- **Deliverable**: `train.py` + trained model files + notebook with metrics

### Week 2: Backend API (20 hours)
- [ ] Set up FastAPI project structure
- [ ] Build Pydantic schemas (BorrowerProfile, ScoreResponse, etc.)
- [ ] Implement `/score` endpoint (load model → featurize → predict → SHAP → format response)
- [ ] Implement `/ingest` endpoint (mock webhook handler)
- [ ] Implement `/borrower/:id` endpoint
- [ ] Add error handling + input validation
- [ ] Create mock data fixtures for 3 personas
- [ ] Test all endpoints with Postman/curl
- **Deliverable**: Running FastAPI server on localhost:8000

### Week 3: Frontend (20 hours)
- [ ] Set up React project (Vite or CRA)
- [ ] Build ScoreGauge component (circular gauge with color coding)
- [ ] Build IncomeChart component (6-month trend line)
- [ ] Build SHAPBreakdown component (horizontal bar chart)
- [ ] Build CoachingTips component (cards with actionable tips)
- [ ] Build Dashboard page (layout all components)
- [ ] Build LenderView tab (show API response)
- [ ] Wire frontend to backend API calls
- [ ] Test with all 3 personas
- **Deliverable**: Running React app on localhost:3000

### Week 4: Polish + Deployment (20 hours)
- [ ] Create demo personas with realistic data
- [ ] Deploy backend to Render (or Railway)
- [ ] Deploy frontend to Vercel
- [ ] Update API calls to use production URLs
- [ ] Write comprehensive README (problem, architecture, setup, usage)
- [ ] Create architecture diagram (ASCII or visual)
- [ ] Record 2-min demo video
- [ ] Prepare hackathon pitch (2 min script)
- [ ] Final testing + bug fixes
- **Deliverable**: Live demo URL + GitHub repo + pitch ready

---

## 10. KEY ASSUMPTIONS & CONSTRAINTS

### Assumptions
- Kaggle Home Credit dataset is representative enough for gig-worker income patterns
- SHAP explanations will be generated in <1 sec (acceptable for MVP)
- 3 personas are sufficient to demonstrate the product concept
- No real Razorpay/UPI/AA integration needed (mock data is fine for hackathon)

### Constraints
- **No regulatory compliance** needed (this is a demo, not a real lending product)
- **No user authentication** (assume everyone accessing dashboard is authorized)
- **No database persistence** (in-memory or SQLite is fine)
- **No mobile app** (web browser only)
- **Limited to public APIs** (no proprietary Razorpay credentials)

---

## 11. SUCCESS CRITERIA (FOR JUDGING)

1. **Problem Validation** (25%): Clear statement of the gig worker credit gap with real numbers
2. **Technical Implementation** (35%):
   - XGBoost model trained and working
   - FastAPI endpoint returning credible scores
   - React dashboard displaying results
   - SHAP explanations generated correctly
3. **User Experience** (20%): Dashboard is clean, intuitive, and visually appealing
4. **Originality** (10%): Shows knowledge of existing solutions (KarmaLife, Krip) but explains why this is different
5. **Presentation** (10%): Clear 2-min pitch, confident delivery, working live demo

---

## 12. WHAT NOT TO DO

- ❌ Don't integrate real Razorpay API (needs business verification)
- ❌ Don't build a full production-grade system (focus on MVP)
- ❌ Don't spend time on mobile (web is enough)
- ❌ Don't use proprietary models (stick to open-source XGBoost)
- ❌ Don't overcomplicate the UI (simple and clean beats fancy)
- ❌ Don't skip the demo (working code > perfect docs)

---

## 13. QUICK REFERENCE: COMMANDS

### Python Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### Train Model
```bash
cd model/
python train.py
# Outputs: model.pkl, scaler.pkl, explainer.pkl, metrics.json
```

### Run Backend
```bash
cd backend/
pip install fastapi uvicorn pandas xgboost shap
uvicorn main:app --reload --port 8000
```

### Run Frontend
```bash
cd frontend/
npm install
npm run dev
# Opens on localhost:3000
```

### Deploy
```bash
# Backend (Render)
git push origin main  # Connects to Render via GitHub

# Frontend (Vercel)
vercel deploy --prod
```

---

## 14. RESOURCES & REFERENCES

**Datasets**:
- [Home Credit Default Risk (Kaggle)](https://www.kaggle.com/c/home-credit-default-risk/data)

**Research Papers**:
- FinGig-CreditNet (2025) — Hybrid deep learning for gig worker credit scoring
- Upstart empirical results — 43% approval lift

**Related Startups**:
- KarmaLife ($8M raised, India)
- Krip (Finland, acquired)
- Portify (UK, £7M Series A)

**Libraries**:
- XGBoost: https://xgboost.readthedocs.io/
- SHAP: https://shap.readthedocs.io/
- FastAPI: https://fastapi.tiangolo.com/
- React + Recharts: https://recharts.org/

---

## 15. QUESTIONS TO ANSWER DURING BUILD

1. **How do I handle missing values in the dataset?**
   - Use median imputation for numeric, mode for categorical. Drop features with >50% missing.

2. **What if my AUC is <85%?**
   - Tune hyperparameters (learning_rate, max_depth). Try feature selection. Try different class_weight.

3. **How do I generate SHAP explanations fast?**
   - Use TreeExplainer (not KernelExplainer). Cache explainer object. Generate for top features only.

4. **Should I deploy to Render or Railway?**
   - Either works. Render has free tier. Use their GitHub integration for auto-deploy.

5. **How do I make the UI look professional?**
   - Use Tailwind CSS templates. Copy from Shadcn/ui or Headless UI. Keep it minimal.

6. **Can I use a pre-trained model instead of training my own?**
   - No. Training the model + showing code is part of the impression. But you can use Home Credit baseline.

---

**Last Updated**: June 6, 2025 | **Project Lead**: [Your Name] | **Hackathon**: [Event Name]
