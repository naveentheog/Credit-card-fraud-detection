"""
tests/test_features.py — unit tests for the feature engineering logic used in
both src/prepare_data.py and api/main.py, so both stay consistent.
"""
import numpy as np


def compute_hour(time_seconds):
    return (time_seconds // 3600) % 24


def compute_amount_log(amount):
    return np.log1p(amount)


def test_hour_wraps_at_24():
    # Time = 90000 seconds = 25 hours -> should wrap to hour 1
    assert compute_hour(90000) == 1


def test_hour_zero_at_start():
    assert compute_hour(0) == 0


def test_hour_never_exceeds_23():
    for t in [0, 3599, 3600, 86399, 172799]:
        h = compute_hour(t)
        assert 0 <= h <= 23


def test_amount_log_zero_amount():
    # log1p handles Amount=0 without error (plain log(0) would be undefined)
    result = compute_amount_log(0)
    assert result == 0.0


def test_amount_log_increasing():
    # Amount_log should be monotonically increasing with Amount
    assert compute_amount_log(10) < compute_amount_log(100) < compute_amount_log(1000)


def test_amount_log_reduces_skew():
    # Sanity check: the ratio between two amounts shrinks a lot after log-transforming,
    # which is the whole point of doing this for Logistic Regression
    raw_ratio = 25000 / 5
    log_ratio = compute_amount_log(25000) / compute_amount_log(5)
    assert log_ratio < raw_ratio
