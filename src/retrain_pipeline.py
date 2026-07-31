"""
src/retrain_pipeline.py
------------------------
The full auto-retraining decision loop:

    New Data -> Validation -> Feature Engineering -> Training -> Evaluation
    -> "Better than Production?" -> Yes: Register + Promote | No: Stop

This is written to actually run, not just describe the idea. In a real deployment,
this script would be triggered on a schedule (see the "Scheduled retraining" note
at the bottom) or by a drift alert from Step 19. Here, we run it once manually,
pointed at a genuinely new batch of data, and let it make its own promote/reject
decision based on real evaluation numbers - not a decision we hardcode in advance.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
MLFLOW_URI = f"sqlite:///{os.path.abspath(os.path.join(PROJECT_ROOT, 'mlruns', 'mlflow.db'))}"
MODEL_NAME = "fraud_detector"
PERFORMANCE_THRESHOLD = 0.0  # a new model must beat production by at least this much PR-AUC to be promoted
# (set to 0.0 = "any genuine improvement", raise to e.g. 0.01 to require a meaningful margin,
#  not just noise, before triggering a redeploy)


def validate_new_data(df: pd.DataFrame) -> bool:
    """Basic sanity checks before trusting new data enough to train on it."""
    required_cols = [f"V{i}" for i in range(1, 29)] + ["Hour", "Amount_log", "Class"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"VALIDATION FAILED: missing columns {missing_cols}")
        return False
    if df.isnull().sum().sum() > 0:
        print(f"VALIDATION FAILED: {df.isnull().sum().sum()} missing values found")
        return False
    if df['Class'].nunique() < 2:
        print("VALIDATION FAILED: new data has only one class - can't train/evaluate a classifier on this")
        return False
    print(f"Validation passed: {len(df)} rows, {df['Class'].sum()} fraud cases, no missing values")
    return True


def get_current_production_metrics(client: MlflowClient):
    """Pull the currently-deployed model's metrics from the registry, so we have
    something concrete to beat, not an arbitrary bar."""
    prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    if not prod_versions:
        print("No model currently in Production - any validated model will be promoted.")
        return None, None
    prod_version = prod_versions[0]
    run = client.get_run(prod_version.run_id)
    return prod_version, run.data.metrics


def retrain(X_train, y_train, best_params_path):
    if os.path.exists(best_params_path):
        params = joblib.load(best_params_path)
        print(f"Using saved best hyperparameters: {params}")
    else:
        params = {"max_depth": 5, "learning_rate": 0.1, "n_estimators": 200}
        print(f"No saved hyperparameters found, using reasonable defaults: {params}")

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    params = dict(params)
    params["scale_pos_weight"] = neg / pos
    params["eval_metric"] = "aucpr"
    params["random_state"] = 42

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model, params


def main(new_data_path):
    print(f"=== Auto-Retraining Pipeline started ===\n")

    # --- Step 1: New Data ---
    print(f"Loading new data from {new_data_path}")
    new_df = pd.read_csv(new_data_path)
    print(f"Loaded {len(new_df)} rows\n")

    # --- Step 2: Validation ---
    print("--- Validation ---")
    if not validate_new_data(new_df):
        print("\nPIPELINE STOPPED: new data failed validation, not training on it.")
        sys.exit(1)

    # --- Step 3: Feature Engineering ---
    # (In this pipeline the incoming CSV is assumed pre-engineered, matching
    #  src/prepare_data.py's output schema - Hour and Amount_log already present.)
    feature_cols = [c for c in new_df.columns if c not in ["Class", "Time", "Amount"]]
    X_new = new_df[feature_cols]
    y_new = new_df["Class"]
    print(f"\n--- Feature Engineering ---\nUsing {len(feature_cols)} features (already engineered)\n")

    # --- Step 4: Training ---
    print("--- Training ---")
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(X_new, y_new, test_size=0.3, stratify=y_new, random_state=42)
    best_params_path = os.path.join(PROJECT_ROOT, "models", "best_hyperparams.joblib")
    candidate_model, params_used = retrain(X_tr, y_tr, best_params_path)
    print("Training complete.\n")

    # --- Step 5: Evaluation ---
    print("--- Evaluation ---")
    y_proba = candidate_model.predict_proba(X_val)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    candidate_metrics = {
        "pr_auc": average_precision_score(y_val, y_proba),
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "recall": recall_score(y_val, y_pred, zero_division=0),
        "f1": f1_score(y_val, y_pred, zero_division=0),
    }
    print(f"Candidate model metrics: {candidate_metrics}\n")

    # --- Step 6: "Better than Production?" ---
    print("--- Comparing against current Production model ---")
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()
    prod_version, prod_metrics = get_current_production_metrics(client)

    if prod_metrics is not None:
        print(f"Current Production model (v{prod_version.version}): PR-AUC={prod_metrics.get('pr_auc', 'N/A')}")
        improvement = candidate_metrics["pr_auc"] - prod_metrics.get("pr_auc", 0)
        print(f"Candidate PR-AUC={candidate_metrics['pr_auc']:.4f} vs Production PR-AUC={prod_metrics.get('pr_auc', 0):.4f}"
              f"  (delta={improvement:+.4f})")
        should_promote = improvement > PERFORMANCE_THRESHOLD
    else:
        should_promote = True
        improvement = None

    # --- Step 7: Register Model / Stop ---
    print()
    if should_promote:
        print(f"DECISION: Candidate {'beats' if improvement else 'is the first'} production model "
              f"{'by ' + format(improvement, '+.4f') + ' PR-AUC' if improvement is not None else ''} "
              f"-> REGISTERING AND PROMOTING")
        mlflow.set_experiment("credit-card-fraud-detection")
        with mlflow.start_run(run_name="auto_retrain_candidate"):
            mlflow.log_params({k: v for k, v in params_used.items() if not isinstance(v, (list, dict))})
            mlflow.log_metrics(candidate_metrics)
            mlflow.set_tags({"trigger": "auto_retrain_pipeline", "new_data_rows": len(new_df)})
            model_info = mlflow.xgboost.log_model(candidate_model, "model", registered_model_name=MODEL_NAME)

        new_version = model_info.registered_model_version
        client.transition_model_version_stage(
            name=MODEL_NAME, version=new_version, stage="Production", archive_existing_versions=True
        )
        print(f"New version v{new_version} is now in Production.")
    else:
        print(f"DECISION: Candidate does NOT beat production (delta={improvement:+.4f}, "
              f"threshold={PERFORMANCE_THRESHOLD}) -> STOPPING, keeping current production model.")

    print("\n=== Pipeline finished ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-data", required=True, help="Path to a CSV of new, labeled transactions")
    args = parser.parse_args()
    main(args.new_data)
