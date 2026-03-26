"""
================================================================================
 FRAUD DETECTION API, FastAPI Inference Endpoint
 ProTech Devices Asia | AIM Capstone Project | Step 8: Deployment
 Author: Mark Eliezer M. Villola
================================================================================

USAGE:
  uvicorn app:app --reload --host 0.0.0.0 --port 8000

ENDPOINTS:
  POST /predict        , Score a claim and return fraud probability + routing
  GET  /health         , Model health check
  GET  /explain/{id}   , Retrieve stored SHAP explanation by claim ID

DOCKER:
  docker build -t fraud-detector .
  docker run -p 8000:8000 fraud-detector
"""

import os
import uuid
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ──────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────────────────────────────────────

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

try:
    model     = joblib.load(os.path.join(MODELS_DIR, 'xgboost.pkl'))
    features  = joblib.load(os.path.join(MODELS_DIR, 'selected_features.pkl'))
    MODEL_LOADED = True
    print("[API] XGBoost model loaded successfully.")
except Exception as e:
    MODEL_LOADED = False
    print(f"[API] WARNING: Could not load model, {e}")
    print("[API] Running in demo mode.")

try:
    explainer = joblib.load(os.path.join(MODELS_DIR, 'shap_explainer.pkl'))
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────────
# APP INITIALISATION
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="InsurTech Fraud Detection API",
    description=(
        "AI-powered fraud scoring for device protection insurance claims. "
        "Developed by ProTech Devices Asia | AIM Capstone Project 2025."
    ),
    version="1.0.0",
    contact={
        "name": "Mark Eliezer M. Villola",
        "url": "https://www.protechdevices.com",
        "email": "mark.villola@protechdevices.com",
    },
)

# In-memory store for logged predictions (production would use a database)
prediction_log: Dict[str, dict] = {}

# ──────────────────────────────────────────────────────────────────────────────
# REQUEST / RESPONSE SCHEMAS
# ──────────────────────────────────────────────────────────────────────────────

class ClaimInput(BaseModel):
    """
    Input schema for a device insurance claim.
    Maps to the engineered feature set used by the XGBoost model.
    """
    TransactionAmt:  float = Field(..., gt=0, description="Claim amount in USD")
    TransactionDT:   int   = Field(..., gt=0, description="Days × 86400 since reference date (proxy for days since policy)")
    ProductCD:       int   = Field(default=0, description="Claim type: 0=Water, 1=Hardware, 2=Screen, 3=Theft, 4=Repair")
    card1:           int   = Field(..., description="Retail partner ID")
    card4:           int   = Field(default=0, description="Payment method: 0=credit, 1=debit, 2=prepaid")
    addr1:           float = Field(default=300.0, description="Geographic region code")
    P_emaildomain:   int   = Field(default=0, description="Email domain risk code (0=trusted, 4=high-risk)")
    dist1:           float = Field(default=50.0, description="Claim velocity proxy")
    M4:              int   = Field(default=2, description="Identity verification: 0=unverified, 2=verified")
    V126:            float = Field(default=0.8, description="Device fingerprint signal [0–1]")
    V130:            float = Field(default=0.8, description="Session behavior signal [0–1]")
    V136:            float = Field(default=0.8, description="Behavioral consistency [0–1]")
    id_02:           float = Field(default=100000.0, description="External identity risk score")
    id_06:           float = Field(default=0.5, description="Session risk indicator")

    class Config:
        json_schema_extra = {
            "example": {
                "TransactionAmt": 850.00,
                "TransactionDT":  432000,
                "ProductCD":      3,
                "card1":          4419,
                "card4":          2,
                "addr1":          350,
                "P_emaildomain":  4,
                "dist1":          5.0,
                "M4":             0,
                "V126":           0.12,
                "V130":           0.15,
                "V136":           0.18,
                "id_02":          450000,
                "id_06":          -5.2
            }
        }


class FraudPrediction(BaseModel):
    """Response schema for fraud prediction."""
    claim_id:           str
    fraud_probability:  float
    risk_tier:          str
    recommended_action: str
    top_factors:        List[str]
    model_version:      str
    timestamp:          str


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    model_version: str
    timestamp:    str


# ──────────────────────────────────────────────────────────────────────────────
# BUSINESS LOGIC HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def classify_tier(prob: float) -> tuple[str, str]:
    """Map fraud probability to operational tier and recommended action."""
    if prob < 0.20:
        return "STP_APPROVE", "Auto-approve, claim cleared for straight-through processing"
    elif prob < 0.60:
        return "STANDARD_REVIEW", "Route to standard adjuster review queue"
    elif prob < 0.85:
        return "ENHANCED_REVIEW", "Hold payment, assign to senior adjuster for enhanced review"
    else:
        return "ESCALATE", "Hold payment immediately, escalate to fraud investigation team"


def build_explanation(claim: ClaimInput, prob: float) -> List[str]:
    """
    Generate plain-language explanation factors for the fraud score.
    Uses business rules derived from SHAP analysis, fallback when SHAP explainer
    is not available or for low-latency response.
    """
    factors = []

    # Timing signal
    days_since = claim.TransactionDT / 86400
    if days_since < 14:
        factors.append(
            f"Claim filed {days_since:.0f} days after policy activation, "
            f"claims in the first 14 days are statistically 4× more likely to be fraudulent"
        )

    # Amount signal
    device_value_map = {0: 600, 1: 800, 2: 500, 3: 900, 4: 400}
    device_avg = device_value_map.get(claim.ProductCD, 600)
    ratio = claim.TransactionAmt / device_avg
    if ratio > 1.15:
        factors.append(
            f"Claim amount (${claim.TransactionAmt:.0f}) exceeds the average device "
            f"replacement value for this category by {(ratio-1)*100:.0f}%, possible over-claiming"
        )

    # Device fingerprint
    if claim.V126 < 0.3:
        factors.append(
            "Device fingerprint signal is anomalous, claim may not have been submitted "
            "from a genuine enrolled device"
        )

    # Email risk
    if claim.P_emaildomain >= 3:
        factors.append(
            "High-risk email domain detected at policy level, "
            "disposable or anonymous email providers are associated with elevated fraud risk"
        )

    # Identity verification
    if claim.M4 == 0:
        factors.append(
            "Customer identity is unverified (no matching records), "
            "unverified customers show 9.3% fraud rate vs. 1.8% for verified customers"
        )

    # Session risk
    if claim.id_06 < -3:
        factors.append(
            "Proxy/VPN usage detected during claim submission, "
            "session routing inconsistency is a fraud indicator"
        )

    # If no specific factors triggered, give a general explanation
    if not factors:
        factors.append(
            f"Fraud probability {prob:.1%} is above auto-approve threshold based on "
            "combination of transaction, behavioral, and identity signals"
        )

    return factors[:3]  # Return top 3 most relevant factors


def engineer_api_features(claim: ClaimInput) -> pd.DataFrame:
    """
    Apply feature engineering to raw claim input.
    Mirrors the feature_engineering.py logic for real-time inference.
    """
    days_since = claim.TransactionDT / 86400.0
    device_map = {0: 600, 1: 800, 2: 500, 3: 900, 4: 400}
    device_val = device_map.get(claim.ProductCD, 600)

    feat = {
        'TransactionAmt':   claim.TransactionAmt,
        'TransactionDT':    claim.TransactionDT,
        'ProductCD':        claim.ProductCD,
        'card1':            claim.card1,
        'card4':            claim.card4,
        'addr1':            claim.addr1,
        'P_emaildomain':    claim.P_emaildomain,
        'dist1':            claim.dist1,
        'M4':               claim.M4,
        'V126':             claim.V126,
        'V130':             claim.V130,
        'V136':             claim.V136,
        'id_02':            claim.id_02,
        'id_06':            claim.id_06,
        # Engineered features
        'log_claim_amt':    np.log1p(claim.TransactionAmt),
        'days_since_policy': days_since,
        'is_early_claim':   int(days_since < 14),
        'claim_velocity':   np.log1p(10),    # Default; would be computed from DB in prod
        'email_risk_score': claim.P_emaildomain / 4.0,
        'amt_to_device_ratio': claim.TransactionAmt / device_val,
        'mismatch_score':   2 - claim.M4,
        'high_risk_x_early': (claim.P_emaildomain / 4.0) * int(days_since < 14),
        'log_dist1':        np.log1p(claim.dist1),
        'amt_zscore':       (claim.TransactionAmt - device_val) / (device_val * 0.3 + 1),
    }

    return pd.DataFrame([feat])


# ──────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Check API and model health."""
    return HealthResponse(
        status="operational" if MODEL_LOADED else "demo_mode",
        model_loaded=MODEL_LOADED,
        model_version="xgboost-v1.0-capstone-2025",
        timestamp=datetime.utcnow().isoformat() + "Z"
    )


@app.post("/predict", response_model=FraudPrediction, tags=["Fraud Detection"])
def predict_fraud(claim: ClaimInput):
    """
    Score a device insurance claim for fraud probability.

    Returns:
    - **fraud_probability**: Model score [0.0–1.0]
    - **risk_tier**: Operational routing tier
    - **recommended_action**: Plain-language action for the claims team
    - **top_factors**: Top 3 fraud signals driving the score
    """
    claim_id = str(uuid.uuid4())[:12].upper()

    if MODEL_LOADED:
        try:
            X = engineer_api_features(claim)
            # Align to model's expected feature order
            for feat in features:
                if feat not in X.columns:
                    X[feat] = 0.0
            X = X.reindex(columns=features, fill_value=0.0)
            prob = float(model.predict_proba(X)[:, 1][0])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model inference error: {e}")
    else:
        # Demo mode: rule-based scoring
        prob = 0.0
        days = claim.TransactionDT / 86400
        if days < 14:
            prob += 0.35
        device_val = {0:600,1:800,2:500,3:900,4:400}.get(claim.ProductCD, 600)
        if claim.TransactionAmt / device_val > 1.2:
            prob += 0.25
        if claim.V126 < 0.3:
            prob += 0.20
        if claim.P_emaildomain >= 3:
            prob += 0.15
        if claim.M4 == 0:
            prob += 0.15
        if claim.id_06 < -3:
            prob += 0.10
        prob = min(prob, 0.99)

    tier, action = classify_tier(prob)
    factors = build_explanation(claim, prob)

    result = FraudPrediction(
        claim_id=claim_id,
        fraud_probability=round(prob, 4),
        risk_tier=tier,
        recommended_action=action,
        top_factors=factors,
        model_version="xgboost-v1.0-capstone-2025",
        timestamp=datetime.utcnow().isoformat() + "Z"
    )

    # Log prediction (in production: write to database)
    prediction_log[claim_id] = result.dict()
    prediction_log[claim_id]['raw_input'] = claim.dict()

    return result


@app.get("/explain/{claim_id}", tags=["Fraud Detection"])
def get_explanation(claim_id: str):
    """
    Retrieve stored prediction and explanation for a specific claim ID.
    """
    if claim_id not in prediction_log:
        raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found in prediction log")
    return prediction_log[claim_id]


@app.get("/", tags=["System"])
def root():
    return {
        "service": "InsurTech Fraud Detection API",
        "version": "1.0.0",
        "author": "Mark Eliezer M. Villola, ProTech Devices Asia",
        "docs": "/docs",
        "health": "/health"
    }
