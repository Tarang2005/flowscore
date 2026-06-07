"""
FlowScore — Pydantic Request/Response Schemas
===============================================

Type-safe data models for every API endpoint. Pydantic v2 validates
all incoming JSON automatically and generates OpenAPI docs.

Schema Hierarchy:
    Request Models:
        BorrowerScoreRequest  → POST /score
        TransactionIngest     → POST /ingest

    Response Models:
        ScoreResponse         → POST /score
        IngestResponse        → POST /ingest
        BorrowerResponse      → GET  /borrower/{id}
        HealthResponse        → GET  /health
        ErrorResponse         → all error cases

    Nested Models:
        PlatformData          → individual gig platform info
        IncomeData            → aggregated income across platforms
        SpendingData          → spending patterns + payment history
        CreditProfile         → existing loans and credit inquiries
        CalculatedFeatures    → pre-computed feature values
        SHAPFactor            → single SHAP contribution
        SHAPExplanation       → top positive + negative factors
        ModelMetadata         → version, timestamp, feature count
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ===================================================================
# Nested Models — Building blocks for requests and responses
# ===================================================================

class PlatformData(BaseModel):
    """A single gig platform's earnings and metadata."""
    platform: str = Field(..., description="Platform name (e.g., 'swiggy_partner', 'upwork')")
    monthly_earnings_last_6m: list[float] = Field(
        ...,
        min_length=1,
        max_length=12,
        description="Monthly earnings for last 6 months, oldest → newest",
    )
    platform_rating: float = Field(
        ..., ge=1.0, le=5.0,
        description="Platform rating (1.0–5.0)",
    )
    platform_account_age_months: Optional[int] = Field(
        None, ge=0,
        description="Months since platform account was created",
    )


class IncomeData(BaseModel):
    """Aggregated income data across all gig platforms."""
    platforms: list[PlatformData] = Field(
        ..., min_length=1, max_length=10,
        description="List of gig platforms with earnings data",
    )
    total_monthly_income_current: Optional[float] = Field(
        None, ge=0,
        description="Current month's total income across all platforms",
    )
    avg_monthly_income_6m: Optional[float] = Field(
        None, ge=0,
        description="Average monthly income over last 6 months",
    )


class SpendingCategories(BaseModel):
    """Breakdown of spending by category."""
    food: Optional[float] = Field(0, ge=0)
    transport: Optional[float] = Field(0, ge=0)
    utilities: Optional[float] = Field(0, ge=0)
    personal: Optional[float] = Field(0, ge=0)


class SpendingData(BaseModel):
    """Monthly spending patterns and payment history."""
    avg_monthly_spending: float = Field(..., ge=0, description="Average monthly spending")
    spending_categories: Optional[SpendingCategories] = None
    late_payments_count_6m: int = Field(0, ge=0, description="Late payments in last 6 months")
    missed_payments_count_6m: int = Field(0, ge=0, description="Missed payments in last 6 months")


class CreditProfile(BaseModel):
    """Existing credit obligations."""
    existing_loans: int = Field(0, ge=0, description="Number of active loans")
    total_debt: float = Field(0, ge=0, description="Total outstanding debt")
    credit_inquiries_6m: int = Field(0, ge=0, description="Credit inquiries in last 6 months")


class CalculatedFeatures(BaseModel):
    """Pre-computed feature values (optional — server can compute these)."""
    income_volatility: Optional[float] = Field(None, ge=0, description="Income CV (std/mean)")
    income_trend: Optional[float] = Field(None, description="6-month income growth rate")
    spending_to_income_ratio: Optional[float] = Field(None, ge=0, description="Spending / Income")
    platform_count: Optional[int] = Field(None, ge=1, description="Number of active platforms")
    days_active_primary: Optional[int] = Field(None, ge=0, description="Days on primary platform")
    income_velocity_3m: Optional[float] = Field(None, description="3-month income growth rate")


# ===================================================================
# Request Models
# ===================================================================

class BorrowerScoreRequest(BaseModel):
    """
    POST /score — Request body for scoring a borrower.

    The client sends a borrower profile with income, spending, and
    credit data. The server extracts 23 features, runs the XGBoost
    model, computes SHAP explanations, and returns the FlowScore.
    """
    borrower_id: str = Field(..., min_length=1, max_length=64, description="Unique borrower ID")
    name: Optional[str] = Field(None, max_length=128, description="Borrower name")
    age: Optional[int] = Field(None, ge=18, le=100, description="Borrower age")
    income_data: IncomeData
    spending_data: SpendingData
    credit_profile: Optional[CreditProfile] = Field(default_factory=CreditProfile)
    calculated_features: Optional[CalculatedFeatures] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "borrower_id": "priya_001",
                    "name": "Priya Sharma",
                    "age": 28,
                    "income_data": {
                        "platforms": [
                            {
                                "platform": "swiggy_partner",
                                "monthly_earnings_last_6m": [35000, 38000, 42000, 45000, 48000, 52000],
                                "platform_rating": 4.8,
                                "platform_account_age_months": 24,
                            }
                        ],
                        "total_monthly_income_current": 52000,
                        "avg_monthly_income_6m": 43333,
                    },
                    "spending_data": {
                        "avg_monthly_spending": 28000,
                        "late_payments_count_6m": 0,
                        "missed_payments_count_6m": 0,
                    },
                    "credit_profile": {
                        "existing_loans": 0,
                        "total_debt": 0,
                        "credit_inquiries_6m": 1,
                    },
                }
            ]
        }
    }


class TransactionData(BaseModel):
    """A single mock transaction (Razorpay/UPI)."""
    id: str = Field(..., description="Transaction ID")
    amount: float = Field(..., gt=0, description="Transaction amount in INR")
    currency: str = Field("INR", description="Currency code")
    timestamp: datetime = Field(..., description="Transaction timestamp")
    platform: str = Field(..., description="Source platform")
    category: str = Field("earnings", description="Transaction category (earnings/expense)")


class TransactionIngestRequest(BaseModel):
    """
    POST /ingest — Mock Razorpay webhook for transaction ingestion.

    Simulates real-time income data flowing into FlowScore from
    payment gateways. In production, this would be an actual
    Razorpay/Cashfree webhook endpoint.
    """
    borrower_id: str = Field(..., min_length=1, description="Borrower ID to update")
    event: str = Field("payment.authorized", description="Webhook event type")
    transaction: TransactionData

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "borrower_id": "priya_001",
                    "event": "payment.authorized",
                    "transaction": {
                        "id": "txn_123456",
                        "amount": 45000,
                        "currency": "INR",
                        "timestamp": "2025-06-01T15:30:00Z",
                        "platform": "swiggy_partner",
                        "category": "earnings",
                    },
                }
            ]
        }
    }


# ===================================================================
# Response Models
# ===================================================================

class SHAPFactor(BaseModel):
    """A single feature's SHAP contribution to the score."""
    feature: str = Field(..., description="Feature name")
    contribution: float = Field(..., description="SHAP value (+ increases risk, - decreases risk)")
    value: float = Field(..., description="Raw feature value for this borrower")


class SHAPExplanation(BaseModel):
    """SHAP breakdown — why the score is what it is."""
    top_positive_factors: list[SHAPFactor] = Field(
        ..., description="Features that increase default risk (lower score)"
    )
    top_negative_factors: list[SHAPFactor] = Field(
        ..., description="Features that decrease default risk (higher score)"
    )


class ModelMetadata(BaseModel):
    """Model version and inference metadata."""
    model_version: str = "v1.0"
    prediction_timestamp: datetime = Field(default_factory=datetime.utcnow)
    feature_count: int = 23


class ScoreResponse(BaseModel):
    """
    POST /score — Full scoring response.

    Contains the FlowScore, default probability, risk category,
    SHAP explanation, and personalized coaching tips.
    """
    borrower_id: str
    flowscore: int = Field(..., ge=300, le=850, description="FlowScore (300–850)")
    default_probability: float = Field(..., ge=0, le=1, description="P(default)")
    confidence_interval: list[int] = Field(
        ..., min_length=2, max_length=2,
        description="[lower_bound, upper_bound] of score",
    )
    risk_category: str = Field(
        ..., description="Risk bucket: 'low', 'medium', 'high', 'very_high'"
    )
    shap_explanation: SHAPExplanation
    coaching_tips: list[str] = Field(
        ..., description="Personalized actionable tips for the borrower"
    )
    model_metadata: ModelMetadata = Field(default_factory=ModelMetadata)


class IngestResponse(BaseModel):
    """POST /ingest — Transaction ingestion confirmation."""
    status: str = Field("ingested", description="Ingestion status")
    borrower_id: str
    updated_income_current: float = Field(..., description="Updated current monthly income")
    score_changed: bool = Field(..., description="Whether the FlowScore changed")
    new_flowscore: int = Field(..., ge=300, le=850)


class ScoreHistoryEntry(BaseModel):
    """A single point in the borrower's score history."""
    date: str = Field(..., description="Date of score (YYYY-MM-DD)")
    score: int = Field(..., ge=300, le=850)


class BorrowerResponse(BaseModel):
    """GET /borrower/{borrower_id} — Full borrower profile with score."""
    borrower_id: str
    profile: dict = Field(..., description="Complete borrower profile data")
    flowscore: int = Field(..., ge=300, le=850)
    default_probability: float = Field(..., ge=0, le=1)
    risk_category: str
    shap_explanation: SHAPExplanation
    coaching_tips: list[str]
    score_history: list[ScoreHistoryEntry] = Field(
        ..., description="Historical score trajectory"
    )


class HealthResponse(BaseModel):
    """GET /health — API health check."""
    status: str = Field("healthy", description="Service status")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded")
    version: str = Field("v1.0", description="API version")


class ErrorResponse(BaseModel):
    """Standard error response format."""
    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Detailed error message")
    status_code: int = Field(..., description="HTTP status code")
