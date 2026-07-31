"""
streamlit_app.py — simple demo UI for the fraud detection model
------------------------------------------------------------------
Run with: streamlit run streamlit_app.py

This is a DEMO tool, not part of the production serving path — the real
serving path is the FastAPI service in api/main.py. This just gives
someone a way to interact with the model without writing curl commands.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳")


@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(MODEL_DIR, "best_model.joblib"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.joblib"))
    model_name = joblib.load(os.path.join(MODEL_DIR, "best_model_name.joblib"))
    return model, feature_columns, model_name


st.title("💳 Credit Card Fraud Detector")

try:
    model, feature_columns, model_name = load_model()
    st.caption(f"Model loaded: **{model_name}**")
except FileNotFoundError:
    st.error("Model not found — run the training notebooks first to produce models/best_model.joblib")
    st.stop()

st.markdown("Enter a transaction's details below. `Time` and `Amount` are the two real, "
            "human-interpretable fields — `V1`-`V28` are anonymized (PCA-transformed) features "
            "from the original dataset.")

col1, col2 = st.columns(2)
with col1:
    time_val = st.number_input("Time (seconds since first transaction)", min_value=0.0, value=50000.0, step=100.0)
with col2:
    amount = st.number_input("Amount ($)", min_value=0.0, value=25.0, step=1.0)

st.markdown("**V1 - V28** (anonymized PCA features — defaults are 0, a 'perfectly typical' transaction)")

# Lay out V1-V28 in a compact grid instead of 28 stacked fields
v_values = []
n_cols = 7
rows = [st.columns(n_cols) for _ in range(4)]
for i in range(28):
    row_idx, col_idx = divmod(i, n_cols)
    with rows[row_idx][col_idx]:
        v = st.number_input(f"V{i+1}", value=0.0, step=0.1, key=f"v{i+1}", format="%.3f")
        v_values.append(v)

threshold = st.slider("Fraud decision threshold", 0.0, 1.0, 0.5, 0.01,
                       help="Lower = catch more fraud but more false alarms. Higher = fewer false "
                            "alarms but miss more fraud. See the Evaluation Metrics notes for why "
                            "this is a business decision, not a fixed default.")

if st.button("Predict", type="primary"):
    row = {f"V{i+1}": v_values[i] for i in range(28)}
    row["Hour"] = (time_val // 3600) % 24
    row["Amount_log"] = float(np.log1p(amount))
    X = pd.DataFrame([row])[feature_columns]

    proba = float(model.predict_proba(X)[0, 1])
    is_fraud = proba >= threshold

    st.markdown("---")
    if is_fraud:
        st.error(f"### 🚨 Flagged as FRAUD")
    else:
        st.success(f"### ✅ Looks legitimate")

    st.metric("Fraud Probability", f"{proba*100:.2f}%")
    st.progress(min(proba, 1.0))
    st.caption(f"Threshold used: {threshold} | Model: {model_name}")

st.markdown("---")
st.caption("This is a demo interface. The production serving path is the FastAPI service "
           "(`api/main.py`), which also exposes `/predict` and `/batch_predict` for programmatic use.")
