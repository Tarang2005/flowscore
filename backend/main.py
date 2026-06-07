"""
FlowScore — FastAPI Application
==================================

REST API for real-time credit scoring of gig workers.

Endpoints:
    POST /score              Score a borrower's creditworthiness
    POST /ingest             Mock Razorpay/UPI transaction webhook
    GET  /borrower/{id}      Fetch borrower profile + score
    GET  /demo/personas      List demo personas with expected scores
    GET  /personas           List loaded personas
    GET  /health             API health check
    GET  /                   Root redirect to docs

Startup:
    uvicorn backend.main:app --reload --port 8000

API Docs:
    http://localhost:8000/docs     (Swagger UI)
    http://localhost:8000/redoc    (ReDoc)
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

try:
    from backend.schemas import (
        BorrowerScoreRequest,
        TransactionIngestRequest,
        ScoreResponse,
        IngestResponse,
        BorrowerResponse,
        HealthResponse,
        ErrorResponse,
        ScoreHistoryEntry,
    )
    from backend.utils import (
        model_store,
        score_borrower,
        extract_features,
        probability_to_flowscore,
        get_risk_category,
        get_confidence_interval,
        generate_shap_explanation,
        generate_coaching_tips,
        FEATURE_NAMES,
    )
except ImportError:
    from schemas import (
        BorrowerScoreRequest,
        TransactionIngestRequest,
        ScoreResponse,
        IngestResponse,
        BorrowerResponse,
        HealthResponse,
        ErrorResponse,
        ScoreHistoryEntry,
    )
    from utils import (
        model_store,
        score_borrower,
        extract_features,
        probability_to_flowscore,
        get_risk_category,
        get_confidence_interval,
        generate_shap_explanation,
        generate_coaching_tips,
        FEATURE_NAMES,
    )

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("flowscore.api")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PERSONAS_PATH = PROJECT_ROOT / "data" / "personas.json"

# ---------------------------------------------------------------------------
# In-Memory Stores (MVP — no database)
# ---------------------------------------------------------------------------
# Stores borrower profiles and transaction history in memory.
# In production, this would be backed by PostgreSQL/Redis.

borrower_store: dict[str, dict] = {}      # borrower_id → profile data
transaction_log: dict[str, list] = {}     # borrower_id → list of transactions
score_history: dict[str, list[dict]] = {} # borrower_id → [{date, score}, ...]


def load_personas():
    """Load the 3 demo personas into the in-memory store at startup."""
    if PERSONAS_PATH.exists():
        try:
            with open(PERSONAS_PATH, "r", encoding="utf-8") as f:
                personas = json.load(f)

            for persona in personas:
                bid = persona["borrower_id"]
                borrower_store[bid] = persona
                transaction_log[bid] = []

                # Generate mock score history (6 data points over 5 months)
                base_scores = {
                    "priya_001": [565, 580, 595, 605, 618, 625],
                    "arjun_001": [710, 720, 730, 740, 748, 755],
                    "meera_001": [520, 530, 535, 540, 545, 555],
                }
                scores = base_scores.get(bid, [580, 590, 600, 605, 610, 620])
                today = datetime.now()
                score_history[bid] = [
                    {"date": (today - timedelta(days=150)).strftime("%Y-%m-%d"), "score": scores[0]},
                    {"date": (today - timedelta(days=120)).strftime("%Y-%m-%d"), "score": scores[1]},
                    {"date": (today - timedelta(days=90)).strftime("%Y-%m-%d"), "score": scores[2]},
                    {"date": (today - timedelta(days=60)).strftime("%Y-%m-%d"), "score": scores[3]},
                    {"date": (today - timedelta(days=30)).strftime("%Y-%m-%d"), "score": scores[4]},
                    {"date": today.strftime("%Y-%m-%d"), "score": scores[5]},
                ]

            logger.info(f"✓ Loaded {len(personas)} demo personas: {list(borrower_store.keys())}")
        except Exception as e:
            logger.warning(f"Could not load personas: {e}")
    else:
        logger.warning(f"Personas file not found at {PERSONAS_PATH}")


# ---------------------------------------------------------------------------
# Application Lifespan (startup + shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup/shutdown lifecycle.

    On startup:
        1. Load ML model artifacts (model.pkl, scaler.pkl, explainer.pkl)
        2. Load demo personas into in-memory store

    On shutdown:
        Log clean shutdown (no cleanup needed for in-memory MVP).
    """
    logger.info("╔" + "═" * 50 + "╗")
    logger.info("║   FlowScore API — Starting up                 ║")
    logger.info("╚" + "═" * 50 + "╝")

    # Load model artifacts
    artifacts_loaded = model_store.load()
    if not artifacts_loaded:
        logger.warning(
            "⚠ Model artifacts not found. API will start but /score "
            "and /borrower endpoints will return 503."
        )

    # Load demo personas
    load_personas()

    logger.info("✓ FlowScore API ready")
    logger.info(f"  Docs: http://localhost:8000/docs")

    yield  # Application runs here

    logger.info("FlowScore API shutting down...")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="FlowScore API",
    description=(
        "AI Credit Scoring Engine for Gig Workers.\n\n"
        "FlowScore analyzes income patterns, platform ratings, spending discipline, "
        "and credit history to produce a FICO-like score (300–850) with SHAP-based "
        "explanations and personalized coaching tips.\n\n"
        "**Demo personas**: Priya (Swiggy), Arjun (Upwork), Meera (Fiverr)"
    ),
    version="1.0.0",
    lifespan=lifespan,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request data"},
        404: {"model": ErrorResponse, "description": "Borrower not found"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
)

# --- CORS Middleware ---
# Allow all origins for development. In production, restrict to
# your Vercel frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global Exception Handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all for unhandled exceptions — returns structured JSON."""
    logger.exception(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc),
            "status_code": 500,
        },
    )


# ===================================================================
# ENDPOINT: GET / — Root (redirect to docs)
# ===================================================================

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to Swagger UI docs."""
    return RedirectResponse(url="/docs")


# ===================================================================
# ENDPOINT: GET /health — Health Check
# ===================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="API health check",
    description="Returns API status and whether the ML model is loaded.",
)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=model_store.is_loaded,
        version="v1.0",
    )


# ===================================================================
# ENDPOINT: POST /score — Score a Borrower
# ===================================================================

@app.post(
    "/score",
    response_model=ScoreResponse,
    tags=["Scoring"],
    summary="Score a borrower's creditworthiness",
    description=(
        "Accepts a borrower profile with income, spending, and credit data. "
        "Returns a FlowScore (300–850), SHAP-based explanation, and "
        "personalized coaching tips. Inference takes <200ms."
    ),
    responses={
        200: {
            "description": "Score computed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "borrower_id": "priya_001",
                        "flowscore": 625,
                        "default_probability": 0.35,
                        "confidence_interval": [610, 640],
                        "risk_category": "medium",
                        "shap_explanation": {
                            "top_positive_factors": [
                                {"feature": "income_trend_6m", "contribution": 45.2, "value": 0.48}
                            ],
                            "top_negative_factors": [
                                {"feature": "spending_to_income_ratio", "contribution": -28.5, "value": 0.54}
                            ],
                        },
                        "coaching_tips": [
                            "Your income grew 48% over 6 months — excellent trajectory! +45 points"
                        ],
                    }
                }
            },
        },
    },
)
async def score_borrower_endpoint(request: BorrowerScoreRequest):
    """
    Score a gig worker's creditworthiness.

    The pipeline:
        1. Extract 23 features from the borrower profile
        2. Scale with the training-fitted StandardScaler
        3. Predict P(default) with XGBoost
        4. Convert to FlowScore (300–850)
        5. Generate SHAP explanation (top 5 positive + 3 negative factors)
        6. Generate coaching tips based on SHAP values
    """
    # Check model is loaded
    if not model_store.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python model/train_model.py` first.",
        )

    try:
        result = score_borrower(request)

        # Store/update the borrower in memory
        borrower_store[request.borrower_id] = request.model_dump()

        # Update score history
        if request.borrower_id not in score_history:
            score_history[request.borrower_id] = []
        score_history[request.borrower_id].append({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "score": result["flowscore"],
        })

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid borrower data: {str(e)}")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception(f"Scoring failed for {request.borrower_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Scoring error: {str(e)}")


# ===================================================================
# ENDPOINT: POST /ingest — Mock Transaction Webhook
# ===================================================================

@app.post(
    "/ingest",
    response_model=IngestResponse,
    tags=["Data Ingestion"],
    summary="Ingest a mock Razorpay/UPI transaction",
    description=(
        "Simulates a payment gateway webhook. Accepts a transaction, "
        "updates the borrower's income data, and optionally re-scores."
    ),
)
async def ingest_transaction(request: TransactionIngestRequest):
    """
    Mock Razorpay/UPI webhook handler.

    In production, this would:
        1. Verify webhook signature (HMAC)
        2. Parse the Razorpay event payload
        3. Update the borrower's transaction history
        4. Trigger re-scoring if income changed significantly

    For MVP, we just log the transaction and update in-memory state.
    """
    bid = request.borrower_id
    txn = request.transaction

    # Store the transaction
    if bid not in transaction_log:
        transaction_log[bid] = []
    transaction_log[bid].append(txn.model_dump())

    # Calculate updated income (simple: add transaction amount to current)
    profile = borrower_store.get(bid, {})
    income_data = profile.get("income_data", {})
    current_income = income_data.get("total_monthly_income_current", 0)

    if txn.category == "earnings":
        updated_income = current_income + txn.amount
    else:
        updated_income = current_income

    # Update the stored profile
    if bid in borrower_store:
        if "income_data" not in borrower_store[bid]:
            borrower_store[bid]["income_data"] = {}
        borrower_store[bid]["income_data"]["total_monthly_income_current"] = updated_income

    # Determine if score changed (simplified: check if income delta > 10%)
    income_delta = abs(updated_income - current_income) / max(current_income, 1)
    score_changed = income_delta > 0.10

    # Get current FlowScore (from history or default)
    history = score_history.get(bid, [])
    current_score = history[-1]["score"] if history else 600

    # Attempt to recalculate score if model is loaded and income changed
    new_score = current_score
    if score_changed and model_store.is_loaded and bid in borrower_store:
        try:
            score_request = _profile_to_score_request(borrower_store[bid])
            result = score_borrower(score_request)
            new_score = result["flowscore"]

            # Append to score history
            if bid not in score_history:
                score_history[bid] = []
            score_history[bid].append({
                "date": datetime.now().strftime("%Y-%m-%d"),
                "score": new_score,
            })
            logger.info(f"Re-scored {bid} after ingestion: {current_score} → {new_score}")
        except Exception as e:
            logger.warning(f"Re-scoring failed for {bid} after ingestion: {e}")
            new_score = current_score

    logger.info(
        f"Ingested transaction: {txn.id} | borrower={bid} | "
        f"amount=₹{txn.amount:,.0f} | platform={txn.platform}"
    )

    return IngestResponse(
        status="ingested",
        borrower_id=bid,
        updated_income_current=updated_income,
        score_changed=score_changed,
        new_flowscore=new_score,
    )


# ===================================================================
# ENDPOINT: GET /borrower/{borrower_id} — Fetch Borrower Profile
# ===================================================================

@app.get(
    "/borrower/{borrower_id}",
    response_model=BorrowerResponse,
    tags=["Borrower"],
    summary="Fetch borrower profile with current score",
    description=(
        "Returns the complete borrower profile, current FlowScore, "
        "SHAP explanation, coaching tips, and score history.\n\n"
        "**Demo IDs**: `priya_001`, `arjun_001`, `meera_001`"
    ),
)
async def get_borrower(borrower_id: str):
    """
    Fetch a borrower's full profile and score.

    If the borrower has a stored profile, scores it in real-time.
    Otherwise returns 404.
    """
    if borrower_id not in borrower_store:
        raise HTTPException(
            status_code=404,
            detail=f"Borrower '{borrower_id}' not found. "
                   f"Available: {list(borrower_store.keys())}",
        )

    profile = borrower_store[borrower_id]
    history = score_history.get(borrower_id, [])

    # If model is loaded, compute a live score
    if model_store.is_loaded:
        try:
            # Build a BorrowerScoreRequest from the stored profile
            score_request = _profile_to_score_request(profile)
            result = score_borrower(score_request)

            return BorrowerResponse(
                borrower_id=borrower_id,
                profile=profile,
                flowscore=result["flowscore"],
                default_probability=result["default_probability"],
                risk_category=result["risk_category"],
                shap_explanation=result["shap_explanation"],
                coaching_tips=result["coaching_tips"],
                score_history=[ScoreHistoryEntry(**h) for h in history],
            )
        except Exception as e:
            logger.warning(f"Live scoring failed for {borrower_id}: {e}. Using cached score.")

    # Fallback: use cached score from history
    cached_score = history[-1]["score"] if history else 600

    return BorrowerResponse(
        borrower_id=borrower_id,
        profile=profile,
        flowscore=cached_score,
        default_probability=0.0,
        risk_category=get_risk_category(cached_score),
        shap_explanation={
            "top_positive_factors": [],
            "top_negative_factors": [],
        },
        coaching_tips=["Score cached — model not loaded for live explanation."],
        score_history=[ScoreHistoryEntry(**h) for h in history],
    )


# ===================================================================
# ENDPOINT: GET /personas — List Available Demo Personas
# ===================================================================

@app.get(
    "/personas",
    tags=["Borrower"],
    summary="List available demo personas",
    description="Returns the borrower IDs and names of all loaded demo personas.",
)
async def list_personas():
    """List all demo personas available for testing."""
    personas = []
    for bid, profile in borrower_store.items():
        personas.append({
            "borrower_id": bid,
            "name": profile.get("name", "Unknown"),
            "age": profile.get("age"),
            "platforms": [
                p.get("platform", "unknown")
                for p in profile.get("income_data", {}).get("platforms", [])
            ],
        })
    return {"personas": personas, "count": len(personas)}


# ===================================================================
# ENDPOINT: GET /demo/personas — Demo Personas with Expected Scores
# ===================================================================

# Expected scores from FLOWSCORE_PROJECT_CONTEXT section 7
DEMO_EXPECTED_SCORES = {
    "priya_001": {"low": 625, "high": 640, "label": "medium-good"},
    "arjun_001": {"low": 750, "high": 770, "label": "excellent"},
    "meera_001": {"low": 550, "high": 570, "label": "below average"},
}


@app.get(
    "/demo/personas",
    tags=["Demo"],
    summary="List demo personas with expected scores",
    description=(
        "Returns the 3 demo personas from the FlowScore project context, "
        "each with borrower_id, name, platform summary, and expected score range."
    ),
)
async def demo_personas():
    """
    List demo personas with expected FlowScore ranges.

    These match the personas defined in FLOWSCORE_PROJECT_CONTEXT section 7:
        - Priya Sharma (Swiggy): 625–640 (medium-good)
        - Arjun Mehta (Upwork+Toptal): 750–770 (excellent)
        - Meera Patel (Fiverr): 550–570 (below average)
    """
    results = []
    for bid, profile in borrower_store.items():
        expected = DEMO_EXPECTED_SCORES.get(bid, {"low": 600, "high": 620, "label": "unknown"})
        history = score_history.get(bid, [])
        current_score = history[-1]["score"] if history else expected["low"]

        results.append({
            "borrower_id": bid,
            "name": profile.get("name", "Unknown"),
            "age": profile.get("age"),
            "platforms": [
                p.get("platform", "unknown")
                for p in profile.get("income_data", {}).get("platforms", [])
            ],
            "expected_score": expected["low"],
            "expected_score_range": [expected["low"], expected["high"]],
            "expected_risk_label": expected["label"],
            "current_score": current_score,
            "income_current": profile.get("income_data", {}).get(
                "total_monthly_income_current", 0
            ),
            "spending_ratio": profile.get("calculated_features", {}).get(
                "spending_to_income_ratio", 0
            ),
        })
    return {"personas": results, "count": len(results)}


# ===================================================================
# Helper: Convert stored profile dict → BorrowerScoreRequest
# ===================================================================

def _profile_to_score_request(profile: dict) -> BorrowerScoreRequest:
    """
    Convert a stored persona dict into a BorrowerScoreRequest.

    The persona JSON format matches the borrower schema from the context
    doc, which is almost identical to the Pydantic request model.
    Minor field mappings are handled here.
    """
    income_data = profile.get("income_data", {})
    spending_data = profile.get("spending_data", {})
    credit_profile = profile.get("credit_profile", {})
    calc_features = profile.get("calculated_features", {})

    return BorrowerScoreRequest(
        borrower_id=profile["borrower_id"],
        name=profile.get("name"),
        age=profile.get("age"),
        income_data={
            "platforms": income_data.get("platforms", []),
            "total_monthly_income_current": income_data.get("total_monthly_income_current"),
            "avg_monthly_income_6m": income_data.get("avg_monthly_income_6m"),
        },
        spending_data={
            "avg_monthly_spending": spending_data.get("avg_monthly_spending", 0),
            "spending_categories": spending_data.get("spending_categories"),
            "late_payments_count_6m": spending_data.get("late_payments_count_6m", 0),
            "missed_payments_count_6m": spending_data.get("missed_payments_count_6m", 0),
        },
        credit_profile={
            "existing_loans": credit_profile.get("existing_loans", 0),
            "total_debt": credit_profile.get("total_debt", 0),
            "credit_inquiries_6m": credit_profile.get("credit_inquiries_6m", 0),
        },
        calculated_features={
            "income_volatility": calc_features.get("income_volatility"),
            "income_trend": calc_features.get("income_trend"),
            "spending_to_income_ratio": calc_features.get("spending_to_income_ratio"),
            "platform_count": calc_features.get("platform_count"),
            "days_active_primary": calc_features.get("days_active_primary"),
            "income_velocity_3m": calc_features.get("income_velocity_3m"),
        } if calc_features else None,
    )


# ===================================================================
# Entry Point (for direct execution)
# ===================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
