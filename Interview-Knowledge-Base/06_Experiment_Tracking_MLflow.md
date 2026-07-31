# Experiment Tracking with MLflow

## Overview

Experiment tracking is the practice of systematically recording every model training run — its
hyperparameters, resulting metrics, and the model artifact itself — so that results are comparable,
reproducible, and auditable, instead of living in scattered notebook cells or someone's memory.

**Beginner level:** Instead of manually writing down "XGBoost with max_depth=5 got 94% precision" in a
notes file, a tool logs it automatically every time you train, alongside the exact settings used.

**Intermediate level:** MLflow provides four main components: Tracking (logging params/metrics/
artifacts per run), Projects (packaging code for reproducible runs), Models (a standard format for
packaging models for deployment), and Model Registry (versioning and stage-managing models — staging,
production, archived).

**Advanced level:** In a real MLOps pipeline, MLflow's tracking store and model registry become the
system of record that connects experimentation (Step 5-7 of this project) to deployment (the FastAPI
serving layer) — the API loads whatever model is currently marked "Production" in the registry, and
promoting a new model from "Staging" to "Production" becomes a deliberate, auditable action rather than
manually swapping a file on disk.

## Theory

**Why not just use notebooks and print statements?** Notebooks are stateful and easy to run out of
order, making results hard to trust or reproduce later. Print statements disappear once the kernel
restarts. Neither gives you a structured, queryable history of every experiment across days or weeks of
work, which becomes essential once you're comparing dozens of hyperparameter combinations (as in Step 6
of this project) rather than 2-3 manual attempts.

**Runs and Experiments:** An "Experiment" in MLflow is a logical grouping (e.g. "credit-card-fraud-
detection"). Each individual training execution within that experiment is a "Run," which stores:
- **Parameters** — hyperparameters and other training config (immutable once logged)
- **Metrics** — numeric results, which CAN be logged multiple times per run (e.g., loss per epoch/
  boosting round), enabling training curves
- **Artifacts** — files: the serialized model, plots (confusion matrix, PR curve), even the training
  data snapshot if desired
- **Tags** — free-form metadata for filtering/organizing runs (e.g., `model_type=xgboost`,
  `resampling=smote`)

**Tracking backend:** MLflow needs somewhere to persist this — a local file store (deprecated in recent
versions in favor of SQL), a SQLite database (simple, file-based, good for solo/small projects), or a
full database (Postgres/MySQL) for team/production use with a separate artifact store (e.g. S3) for the
actual model files, which can get large.

**Model Registry:** A separate layer on top of tracked runs that lets you formally "register" a model
under a name and version, then transition it through stages (`None → Staging → Production → Archived`).
This decouples "which run produced the best metrics" from "which model is actually serving traffic
right now" — the latter is a deliberate, trackable decision, not implicit.

**MLflow Model format (`mlflow.pyfunc`):** A standard, framework-agnostic way to save a model so it can
be loaded and served consistently regardless of whether it was originally scikit-learn, XGBoost, or a
custom PyTorch model — `mlflow.sklearn.log_model()`, `mlflow.xgboost.log_model()`, etc. all produce
models loadable via the same generic `mlflow.pyfunc.load_model()` interface, which simplifies the
serving layer since it doesn't need model-type-specific loading logic.

**Strengths:**
- Turns ad-hoc experimentation into a structured, comparable, auditable history
- Framework-agnostic — works the same way across scikit-learn, XGBoost, PyTorch, etc.
- The UI (`mlflow ui`) gives an immediate visual comparison across runs without extra tooling
- Model Registry cleanly separates "best run by metrics" from "what's actually deployed," supporting
  safer rollouts and rollbacks

**Weaknesses:**
- Adds setup/infrastructure overhead compared to just training in a notebook — not worth it for truly
  trivial one-off experiments
- The default file/SQLite backend doesn't scale well to large teams working concurrently; needs a real
  database + remote artifact store for production use
- Doesn't automatically solve data versioning (that's what DVC or similar tools handle) — MLflow tracks
  *model* artifacts and metrics, not necessarily the exact dataset version each run used, unless you
  explicitly log that yourself

**Common misconceptions:**
- "MLflow trains models for you" — no, it only tracks/manages runs and models; you still write the
  training code.
- "Logging a model to MLflow automatically deploys it" — no, logging and registering are separate from
  actually serving; deployment (e.g. via the FastAPI layer in this project) is a separate step that
  loads a registered/tracked model.
- "The Model Registry and Tracking are the same thing" — Tracking records every run's history;
  Registry is specifically about versioning and stage-managing a curated subset of models chosen to be
  candidates for deployment.

## Interview Questions

**Beginner**
1. What problem does experiment tracking solve that plain notebooks don't?
2. What are the main things MLflow logs per run?
3. What's the difference between a Run and an Experiment in MLflow?

**Intermediate**
4. What's the difference between MLflow Tracking and the MLflow Model Registry?
5. Why would you log multiple values for the same metric within a single run (e.g., across boosting
   rounds)?
6. What tracking backend options does MLflow support, and when would you choose SQLite vs. a full
   database?
7. How does `mlflow.pyfunc` help decouple the serving layer from the specific ML framework used?

**Advanced**
8. How would you structure MLflow experiments and tags to compare 3 models × 3 imbalance strategies ×
   50 Optuna trials cleanly, without the UI becoming unmanageable?
9. Design a promotion workflow: a new model is trained and looks better on offline metrics than the
   current production model. Walk through the steps, using MLflow's registry, to safely get it into
   production.
10. What are the limitations of MLflow for a large team with concurrent experimentation, and how would
    you address them?

**Practical / Scenario-based (Fraud-specific)**
11. You ran 50 Optuna trials for XGBoost overnight. How would you use MLflow to quickly identify the
    best trial and understand what made it better than the others?
12. Six months from now, a new engineer needs to know exactly which model is in production and why it
    was chosen over alternatives. How does your MLflow setup answer that without asking you directly?
13. A newly trained model has better PR-AUC but worse precision at your chosen production threshold than
    the current model. How would you handle this using the registry (would you promote it)?

**Coding**
14. Write code that logs an XGBoost training run to MLflow, including parameters, PR-AUC, and the model
    artifact.
15. Write code to query MLflow programmatically (not via UI) to retrieve the run with the best PR-AUC
    across an experiment.

**Follow-ups an interviewer might throw at you**
- "Why not just save models as pickle files with descriptive filenames instead of using MLflow?"
  (Answer: works for a handful of models, but doesn't scale — no structured metric comparison, no
  registry/stage management, no standard loading interface across frameworks, and no audit trail of
  *why* a model was chosen, just that it exists.)
- "How does MLflow relate to the rest of your MLOps stack — Docker, FastAPI, CI/CD?" (Answer: MLflow
  sits between experimentation and deployment — the FastAPI service loads whatever model MLflow's
  registry marks as "Production"; CI/CD can include a step that checks the registry for a newly
  promoted model and triggers a redeploy; Docker packages the serving code, while MLflow manages which
  model artifact that code actually loads.)

## Answers

**Q4 — Difference between Tracking and Model Registry?**
Tracking is the complete, append-only history of every training run you've ever executed within an
experiment — every hyperparameter combination you tried, including failed or mediocre ones, all
searchable and comparable. The Model Registry is a curated, smaller set — models you've explicitly
decided are worth naming and versioning as real candidates for deployment, with an associated lifecycle
(Staging → Production → Archived). In practice: you might have 50 tracked runs from an Optuna sweep, but
only register the single best one (or a small handful of finalists) into the Model Registry once you've
decided it's a legitimate deployment candidate. Tracking answers "what did I try and what happened";
Registry answers "what's actually a candidate for, or currently is, in production."

**Q9 — Design a promotion workflow for a better-performing new model.**
A defensible workflow: (1) new model finishes training, gets logged as a Run with full metrics via
Tracking; (2) if it beats the current production model's metrics on the same held-out test set, register
it in the Model Registry under the existing model name, creating a new version, initially in "Staging"
stage; (3) run it in shadow mode if possible — score live production traffic with it without actually
acting on its predictions, comparing its outputs against the current production model's real-world
behavior for some period, since offline test-set metrics don't always translate perfectly to live
traffic; (4) if shadow performance confirms the improvement, transition the new version to "Production"
in the registry (and correspondingly move the previous production version to "Archived," keeping it
retrievable for rollback); (5) the FastAPI serving layer, on its next restart or via a hot-reload
mechanism, picks up the new "Production"-tagged model. This whole sequence is deliberately staged rather
than "the training script finishes, immediately overwrite the file the API is loading," because that
skips both the human review of the metric comparison and any live-traffic validation.

**Q13 — Better PR-AUC but worse precision at the chosen threshold — promote or not?**
Don't promote automatically based on PR-AUC alone — PR-AUC summarizes performance across *all*
thresholds, but the production system operates at one specific, business-chosen threshold. It's
entirely possible for a model to have better overall PR-AUC (better performance in aggregate across the
curve) while being worse specifically at your current operating threshold, if its precision-recall curve
crosses the other model's curve near that particular threshold. The correct check: explicitly compare
precision and recall of both models AT the actual production threshold (or re-derive the optimal
threshold for the new model, since a better model doesn't necessarily share the same optimal threshold
as the old one), and make the promotion decision based on that operating-point comparison plus the
business's actual cost tradeoff — not the aggregate PR-AUC number alone. This is a genuinely senior-
level distinction: aggregate metrics and operating-point metrics can disagree, and interviewers use this
scenario specifically to check whether you conflate the two.

## Project Connection

MLflow is the experiment tracking backbone for Steps 5-7 of this project. Concretely:

- Every training run — for all 3 models (Logistic Regression, Random Forest, XGBoost) crossed with
  every imbalance strategy tested (SMOTE, class weighting, undersampling) — is logged as a separate Run
  within a single `credit-card-fraud-detection` Experiment, tagged with `model_type` and `resampling`
  for easy filtering in the UI.
- Every one of the ~50 Optuna trials from Step 6's hyperparameter search is *also* logged as an
  individual nested Run under the XGBoost tuning parent run, so the full search history (not just the
  final winner) is inspectable later — directly answering "how would you show your tuning process was
  thorough, not just report a final number."
- Metrics logged per run: precision, recall, F1, PR-AUC, ROC-AUC (with the caveat noted in the
  Evaluation Metrics file about which one to trust), plus training time.
- Artifacts logged per run: the serialized model itself, the confusion matrix plot, and the precision-
  recall curve plot, so a run's full picture is reviewable without re-running anything.
- The single best run (by PR-AUC, cross-checked against precision/recall at the actual production
  threshold, per the reasoning in Q13 above) gets registered in the Model Registry, and the FastAPI
  service (built earlier in this project) loads the "Production"-staged model from there rather than a
  hardcoded file path — connecting experimentation directly to serving.

## Code Examples

```python
import mlflow
import mlflow.xgboost
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score

mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
mlflow.set_experiment("credit-card-fraud-detection")

with mlflow.start_run(run_name="xgboost_scale_pos_weight"):
    mlflow.set_tags({"model_type": "xgboost", "resampling": "scale_pos_weight"})

    params = {"max_depth": 5, "learning_rate": 0.05, "n_estimators": 400,
              "scale_pos_weight": scale_pos_weight}
    mlflow.log_params(params)

    model = XGBClassifier(**params, eval_metric="aucpr", random_state=42)
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    mlflow.log_metrics({
        "pr_auc": average_precision_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    })

    mlflow.xgboost.log_model(model, "model", registered_model_name="fraud_detector")

# Querying programmatically for the best run across the whole experiment
from mlflow.tracking import MlflowClient

client = MlflowClient()
experiment = client.get_experiment_by_name("credit-card-fraud-detection")
best_run = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.pr_auc DESC"],
    max_results=1
)[0]
print("Best run ID:", best_run.info.run_id)
print("Best PR-AUC:", best_run.data.metrics["pr_auc"])
print("Params:", best_run.data.params)

# Promoting a registered model version to Production
client.transition_model_version_stage(
    name="fraud_detector",
    version=3,
    stage="Production",
    archive_existing_versions=True
)
```

## Visual Explanation

```
 Experiment: "credit-card-fraud-detection"
  │
  ├── Run 1: LogReg + class_weight     (params, metrics, model.pkl logged)
  ├── Run 2: LogReg + SMOTE
  ├── Run 3: RandomForest + class_weight
  ├── Run 4: RandomForest + SMOTE
  ├── Run 5: XGBoost + scale_pos_weight
  ├── Run 6-55: XGBoost Optuna trials (nested under a parent tuning run)
  └── Run 56: XGBoost + best Optuna params  ──►  registered as
                                                  "fraud_detector" v3
                                                        │
                                          Model Registry │
                                          ┌──────────────┴──────────────┐
                                          │  v1: Archived                │
                                          │  v2: Archived                │
                                          │  v3: Production  ◄── loaded  │
                                          │       by FastAPI /predict    │
                                          └───────────────────────────────┘
```

## Common Mistakes

- **Only logging the final metric, not the full training curve** — makes it hard to diagnose
  overfitting (e.g., train vs. validation PR-AUC diverging across boosting rounds) after the fact.
- **Not tagging runs consistently** (`model_type`, `resampling` strategy) — makes filtering/comparing
  dozens of runs in the UI painful later, even though each individual run was logged correctly.
- **Registering every run instead of only genuine candidates** — clutters the Model Registry and makes
  it unclear which versions were ever seriously considered for production versus just experimental
  noise.
- **Promoting a model to Production purely by comparing aggregate metrics** (e.g. PR-AUC) without
  checking performance at the actual operating threshold — see the Q13 answer above for why this can be
  wrong.
- **Treating MLflow as a substitute for data versioning** — it tracks model artifacts and metrics well,
  but doesn't inherently guarantee you know the exact dataset version behind a given run unless you
  explicitly log a data hash/version tag yourself (this is precisely the gap tools like DVC fill).

## Advanced Discussion

- **Nested runs for hyperparameter sweeps:** MLflow supports parent/child run nesting, so an entire
  Optuna study can be logged as one parent run with each of its 50 trials as child runs — this keeps
  the top-level experiment view clean (one row per "real" experiment) while preserving full drill-down
  detail into the sweep.
- **Reproducibility beyond metrics:** For genuine reproducibility, a mature MLflow setup also logs the
  exact code version (git commit hash), the data version/hash, and the environment (`conda.yaml` /
  `requirements.txt` snapshot) alongside each run — MLflow supports all of this via
  `mlflow.log_param("git_commit", ...)` and its Projects feature, and a senior interviewer may probe
  whether you know reproducibility requires more than just logging hyperparameters and metrics.
- **MLflow in a CI/CD-driven retraining pipeline:** In an automated retraining setup, a CI/CD pipeline
  can be configured to only promote a newly retrained model to Production automatically if it beats the
  current production model's metrics by a defined margin on a fixed, held-out evaluation set — otherwise
  it stays in Staging pending manual review. This connects MLflow's registry stages directly to a
  concrete automated/human-gated decision policy, rather than leaving "should we deploy this" as an
  undefined manual step every time.

## Revision Notes

- MLflow Tracking logs params, metrics, and artifacts per training Run, grouped into Experiments
- Model Registry separately versions and stage-manages models chosen as real deployment candidates
  (Staging → Production → Archived)
- `mlflow.pyfunc` gives a framework-agnostic loading interface, decoupling serving code from the
  original training framework
- Nested runs keep large hyperparameter sweeps (e.g. 50 Optuna trials) organized under one parent run
- Promotion to Production should check performance at the actual operating threshold, not just
  aggregate metrics like PR-AUC
- In this project: every model/resampling combination and every Optuna trial is logged; the best
  candidate is registered and promoted, and the FastAPI service loads whatever is currently marked
  Production

## References

- MLflow official documentation — particularly the Tracking and Model Registry guides
- Zaharia, M. et al. (2018). "Accelerating the Machine Learning Lifecycle with MLflow." *IEEE Data
  Engineering Bulletin* — the original MLflow paper/overview from Databricks
- *Designing Machine Learning Systems* (Chip Huyen) — Ch. 6 covers experiment tracking and versioning
  in the broader MLOps lifecycle this fits into
- *Introducing MLOps* (O'Reilly, Treveil et al.) — for the organizational/process context around model
  promotion workflows
