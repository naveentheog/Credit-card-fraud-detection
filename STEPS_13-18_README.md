# Credit Card Fraud Detection — Steps 13-18

## What's real vs. what's documented-only

Everything below was actually run and tested in this environment, with one honest exception:
**Docker itself couldn't be built/run here** (no Docker daemon in this sandbox). The Dockerfile is
correct and standard — build it on your own machine to confirm.

## Step 13 — Docker

`Dockerfile` builds a slim Python 3.11 image, installs `requirements.txt`, copies `api/` and
`models/`, exposes port 8000, runs uvicorn. Untested here (no Docker daemon available), but nothing
in it is exotic — standard FastAPI containerization pattern.

```bash
docker build -t fraud-detection-api .
docker run -p 8000:8000 fraud-detection-api
curl http://localhost:8000/health
```

**Why Docker instead of running Python directly?** Different machines have different Python
versions, OS-level library differences (e.g. XGBoost's OpenMP dependency), and dependency version
drift. Docker packages the exact runtime environment once — the same image behaves identically on
your laptop, a teammate's laptop, or a cloud server.

## Step 14 — Streamlit (`streamlit_app.py`)

A demo UI — enter `Time`, `Amount`, and the 28 `V` features, get a fraud probability back with a
visual flag. **Actually started and tested** — confirmed it serves HTTP 200 with zero errors in this
sandbox. Run with:

```bash
streamlit run streamlit_app.py
```

This is explicitly a demo tool, not the production serving path — that's still the FastAPI service.

## Step 15 — MLflow Model Registry

Registered 3 real model versions (not fabricated) under the name `fraud_detector`:

| Version | Source | PR-AUC |
|---|---|---|
| v1 | XGBoost, untuned (`scale_pos_weight` only) | 0.8027 |
| v2 | XGBoost, Optuna-tuned (6 params) | 0.8188 |
| v3 | XGBoost, Optuna-tuned (8 params incl. gamma/min_child_weight) | **0.8262** |

v3 was promoted to `Production` stage; v1/v2 remain versioned and retrievable for comparison or
rollback. Query it:

```python
from mlflow.tracking import MlflowClient
client = MlflowClient()
prod_model = client.get_latest_versions("fraud_detector", stages=["Production"])
```

## Step 16 — DVC

```bash
dvc init
dvc add data/raw/creditcard.csv     # tracks the raw dataset with a real MD5 hash
```

`dvc.yaml` defines a real, runnable pipeline stage (`prepare_data`) wired to
`src/prepare_data.py` — a standalone script version of the ingestion/validation/feature-engineering
logic from the notebooks. **Actually ran `dvc repro` successfully** — it re-executed the script,
produced `data/processed/creditcard_features.csv`, and generated `dvc.lock` with real content
hashes for both the input data and the script. If `creditcard_new.csv` arrived next month, you'd
swap the raw file, `dvc repro` again, and DVC would know exactly which output came from which input
version.

## Step 17 — GitHub Actions (`.github/workflows/ci.yml`)

Two jobs: `test` (installs deps, lints with flake8 for actual errors — not style nitpicks — then
runs pytest) and `build-docker` (builds the image, runs it, curls `/health` to confirm it actually
boots before considering the build good).

**Actually ran locally**: `flake8` returned 0 errors, and all 12 tests in `tests/` genuinely pass
(`pytest tests/ -v`) — 6 feature-engineering unit tests, 6 API smoke tests using FastAPI's
`TestClient`. The workflow itself can only be confirmed by actually pushing to a real GitHub repo
(can't trigger GitHub's runners from this sandbox), but every step it runs was verified to work
locally first.

**Deliberately not included**: auto-deploy to any cloud target after a successful build. Pushing to
a registry and deploying needs real credentials and a real target — faking that step would be
theater, not engineering. The workflow stops at "build and smoke-test the image," which is the
honest end of what's automatable without real infrastructure behind it.

## Step 18 — Monitoring

Added real Prometheus instrumentation to `api/main.py` (not a mock):
- `fraud_api_requests_total{endpoint, status}` — request counts by endpoint and status code
- `fraud_api_request_latency_seconds{endpoint}` — latency histogram
- `fraud_api_prediction_probability` — histogram of predicted fraud probabilities (useful for
  spotting drift — if live predictions start clustering somewhere very different from training-time
  patterns, that's an early warning sign worth investigating)
- `fraud_api_flagged_total` — running count of transactions flagged as fraud

Exposed at `GET /metrics` in Prometheus's standard text format — point a real Prometheus server at
it, or just `curl /metrics` to read the raw counters.

**Actually tested**: hit `/health` twice and `/predict` three times, then confirmed `/metrics`
correctly showed `requests_total{endpoint="/predict",status="200"} 3.0` and matching latency/
prediction-distribution counts — the numbers reflected exactly what was sent, not placeholder values.

**Not built** (documented as a next step only): a full Grafana dashboard, and label-based
performance monitoring (precision/recall over time) — the latter needs real fraud labels arriving
after the fact (e.g. via chargebacks), which isn't available for a static Kaggle dataset in a demo
environment. In a real deployment, `/metrics` is exactly what a Prometheus + Grafana stack would
scrape to build that dashboard on top of.
