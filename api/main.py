"""
main.py — FastAPI serving layer (v2)
--------------------------------------
Loads the final tuned XGBoost model (Steps 5-10) and serves fraud predictions.

Input contract: raw Time and Amount (like the original Kaggle columns), plus
V1-V28. We recompute Hour and Amount_log here EXACTLY the way training did
(see notebooks/01_..._features_imbalance.ipynb) so a client doesn't need to
know about our internal feature engineering - they just send what a real
transaction record would naturally have.

The model is a tree ensemble (XGBoost), so no scaling is applied here -
only Logistic Regression needed the StandardScaler during training.
"""

import os
import time
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# --- Step 18: Monitoring ---
# Real Prometheus metrics, not a mock - a Prometheus server (or just `curl /metrics`)
# can scrape these directly. Tracks exactly what Step 18 asked for: request volume,
# latency, error rate, and the distribution of predictions being made (useful for
# spotting drift - if predicted probabilities start clustering somewhere very
# different from training-time patterns, that's an early warning sign).
REQUEST_COUNT = Counter(
    "fraud_api_requests_total", "Total requests received", ["endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "fraud_api_request_latency_seconds", "Request latency in seconds", ["endpoint"]
)
PREDICTION_PROBABILITY = Histogram(
    "fraud_api_prediction_probability", "Distribution of predicted fraud probabilities",
    buckets=[0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
)
FLAGGED_COUNT = Counter(
    "fraud_api_flagged_total", "Total transactions flagged as fraud"
)

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Serves fraud-probability predictions for one or many transactions, "
                 "using a tuned XGBoost model (Optuna-tuned, PR-AUC ~0.83 on held-out test data).",
    version="2.0.0",
)

_model = None
_feature_columns = None
_model_name = None


class Transaction(BaseModel):
    Time: float = Field(..., description="Seconds elapsed since the first transaction in the dataset")
    V: List[float] = Field(..., min_length=28, max_length=28, description="V1..V28 PCA features, in order")
    Amount: float = Field(..., ge=0, description="Transaction amount")


class BatchRequest(BaseModel):
    transactions: List[Transaction]


def _load_artifacts():
    global _model, _feature_columns, _model_name
    try:
        _model = joblib.load(os.path.join(MODEL_DIR, "best_model.joblib"))
        _feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.joblib"))
        _model_name = joblib.load(os.path.join(MODEL_DIR, "best_model_name.joblib"))
    except FileNotFoundError:
        _model = None


@app.on_event("startup")
def startup():
    _load_artifacts()


def _build_feature_row(txn: Transaction) -> pd.DataFrame:
    """Recreates the exact feature engineering used in training (Step 3):
    Hour = (Time // 3600) % 24, Amount_log = log1p(Amount)."""
    row = {f"V{i}": v for i, v in enumerate(txn.V, start=1)}
    row["Hour"] = (txn.Time // 3600) % 24
    row["Amount_log"] = float(np.log1p(txn.Amount))
    return pd.DataFrame([row])[_feature_columns]


@app.middleware("http")
async def track_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    endpoint = request.url.path
    REQUEST_COUNT.labels(endpoint=endpoint, status=response.status_code).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
    return response


@app.get("/metrics")
def metrics():
    """Prometheus scrape endpoint. `curl /metrics` to see raw counters/histograms,
    or point a real Prometheus server at this URL to collect them over time."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
def root():
    return {
        "service": "Credit Card Fraud Detection API",
        "model": _model_name,
        "status": "ok" if _model is not None else "model not loaded",
        "endpoints": ["/health", "/predict", "/batch_predict", "/docs"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None, "model_name": _model_name}


@app.post("/predict")
def predict(txn: Transaction, threshold: float = Query(0.5, ge=0.0, le=1.0)):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded — run the training notebooks first "
                                                      "to produce models/best_model.joblib")
    try:
        X = _build_feature_row(txn)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not build features from input: {e}")

    proba = float(_model.predict_proba(X)[0, 1])
    PREDICTION_PROBABILITY.observe(proba)
    is_fraud = proba >= threshold
    if is_fraud:
        FLAGGED_COUNT.inc()

    return {
        "fraud_probability": round(proba, 6),
        "is_fraud": is_fraud,
        "threshold_used": threshold,
        "model": _model_name,
    }


@app.post("/batch_predict")
def batch_predict(batch: BatchRequest, threshold: float = Query(0.5, ge=0.0, le=1.0)):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded — run the training notebooks first "
                                                      "to produce models/best_model.joblib")
    if len(batch.transactions) == 0:
        raise HTTPException(status_code=422, detail="transactions list is empty")

    rows = [_build_feature_row(t) for t in batch.transactions]
    X = pd.concat(rows, ignore_index=True)

    probas = _model.predict_proba(X)[:, 1]
    results = [
        {
            "index": i,
            "fraud_probability": round(float(p), 6),
            "is_fraud": bool(p >= threshold),
        }
        for i, p in enumerate(probas)
    ]
    return {
        "count": len(results),
        "flagged_count": sum(r["is_fraud"] for r in results),
        "threshold_used": threshold,
        "results": results,
    }
