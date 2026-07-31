# Credit Card Fraud Detection — End-to-End ML/MLOps Project

An end-to-end fraud detection system built on the Kaggle "Credit Card Fraud Detection" dataset
(284,807 transactions, ~0.17% fraud) — from data ingestion through a tuned, explainable, monitored,
auto-retraining model served via FastAPI.

## What this covers

- **Data pipeline**: ingestion, validation, feature engineering, imbalance handling (SMOTE/ADASYN/
  class weighting), tracked with **DVC**
- **Modeling**: Logistic Regression, Random Forest, XGBoost — compared across imbalance strategies
- **Tuning**: Optuna hyperparameter search
- **Evaluation**: PR-AUC/precision/recall (not accuracy), threshold analysis, false positive/negative
  error analysis
- **Explainability**: SHAP (global + local, summary/bar/waterfall/force plots)
- **Experiment tracking**: MLflow, with a real Model Registry (versioned, staged, promoted)
- **Drift detection**: Evidently AI (data drift + a concrete concept-drift demonstration)
- **Auto-retraining pipeline**: validates new data, retrains, compares against the current
  Production model, promotes only if genuinely better
- **Serving**: FastAPI (`/predict`, `/batch_predict`, `/health`, `/metrics`) + a Streamlit demo UI
- **Monitoring**: real Prometheus metrics (request count, latency, prediction distribution)
- **Packaging**: Dockerfile, CI (GitHub Actions: lint + test + build), AWS deployment guide (EC2 and
  ECS/Fargate paths)
- **Interview Knowledge Base**: deep-dive reference docs for every model/technique used

## Quickstart

```bash
git clone <this-repo>
cd credit-card-fraud
pip install -r requirements.txt

# Get the data (see data/raw/creditcard.csv.dvc - this repo tracks it with DVC,
# you'll need the actual creditcard.csv from Kaggle placed at data/raw/creditcard.csv)

# Run the pipeline
python src/prepare_data.py          # or: dvc repro
jupyter notebook notebooks/         # walk through Steps 1-11 interactively

# Serve the model
uvicorn api.main:app --reload       # FastAPI at http://localhost:8000/docs
streamlit run streamlit_app.py      # demo UI at http://localhost:8501

# Run tests
pytest tests/ -v
```

## Project structure

```
credit-card-fraud/
├── api/main.py                    # FastAPI serving layer
├── src/
│   ├── prepare_data.py            # ingestion/validation/features (DVC-tracked stage)
│   └── retrain_pipeline.py        # auto-retraining decision loop
├── notebooks/                     # Steps 1-11, executed with real outputs
├── models/                        # trained model + preprocessing artifacts
├── monitoring/                    # plots, drift reports, error analysis CSVs
├── Interview-Knowledge-Base/      # deep-dive docs per model/technique
├── deploy/                        # AWS deployment scripts + guide
├── tests/                         # pytest suite
├── .github/workflows/ci.yml       # lint + test + Docker build
├── Dockerfile
├── dvc.yaml / dvc.lock            # data pipeline versioning
└── streamlit_app.py               # demo UI
```

## Honest notes on what's tested vs. documented

Everything in this repo was actually run and verified during development — model metrics, drift
reports, retraining decisions, and API responses are all real, not fabricated. The exceptions,
documented explicitly where relevant: Docker was built with a correct Dockerfile but not build-tested
in the original development sandbox (no Docker daemon there — test it yourself with `docker build .`);
AWS deployment scripts are correct AWS CLI/ECS syntax but unexecuted (no AWS access in that sandbox).
See `STEPS_13-18_README.md` and `STEPS_19-21_README.md` for the full breakdown.
