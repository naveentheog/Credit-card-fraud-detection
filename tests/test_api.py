"""
tests/test_api.py — smoke tests for the FastAPI service. Uses FastAPI's TestClient
so these run without needing a live uvicorn server.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_root_endpoint_responds():
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert "service" in body
    assert "endpoints" in body


def test_health_endpoint_responds():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert "model_loaded" in body


def test_predict_rejects_malformed_input():
    # Missing required fields entirely -> should be a 422, not a 500 crash
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_rejects_wrong_v_length():
    # V must have exactly 28 values - sending 5 should fail validation cleanly
    bad_payload = {"Time": 1000.0, "V": [0.1, 0.2, 0.3, 0.4, 0.5], "Amount": 10.0}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_accepts_well_formed_input():
    # If a model is loaded, this should return a valid probability.
    # If no model is loaded (e.g. fresh CI checkout with no trained model artifact),
    # it should fail cleanly with a 503, not a 500 crash - both are "correct" outcomes
    # depending on environment, so we just check it doesn't error unexpectedly.
    good_payload = {"Time": 1000.0, "V": [0.0] * 28, "Amount": 10.0}
    response = client.post("/predict", json=good_payload)
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        body = response.json()
        assert 0.0 <= body["fraud_probability"] <= 1.0
        assert isinstance(body["is_fraud"], bool)


def test_batch_predict_rejects_empty_list():
    response = client.post("/batch_predict", json={"transactions": []})
    assert response.status_code in (422, 503)
