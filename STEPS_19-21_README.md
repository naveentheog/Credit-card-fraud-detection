# Steps 19-21 — Drift Detection, Auto-Retraining, AWS Deployment

## Step 19 — Drift Detection (Evidently AI)

Ran two real comparisons using Evidently's `DataDriftPreset`:

1. **Train vs. real test data** (`monitoring/drift_report_train_vs_test.html`) — **0 columns
   drifted**, as expected, since both come from the same underlying distribution.
2. **Train vs. a deliberately drifted synthetic batch** (`monitoring/drift_report_simulated_drift.html`)
   — I shifted exactly 3 features (`V14 += noise~N(2.5,0.5)`, `V4 *= 1.8`, `Amount_log += 0.8`) to
   simulate fraud patterns genuinely changing. Evidently correctly flagged **exactly those 3
   features** as drifted (out of 30), with real Wasserstein distance scores:
   ```
   V14          2.6292  <-- DRIFTED
   V4           0.5907  <-- DRIFTED
   Amount_log   0.4759  <-- DRIFTED
   (everything else stayed under 0.02)
   ```

**Concept drift, made concrete** — same model, same 0.5 threshold, run on both datasets:
```
Real test data:     PR-AUC=0.8262  Precision=0.9146  Recall=0.7895
Drifted data:        PR-AUC=0.7995  Precision=0.8642  Recall=0.7368
```
This is the actual mechanism connecting the four terms from the step: **feature drift** (the V14/V4/
Amount_log shift) caused **concept drift** (the same model performs worse), which is exactly why
data drift monitoring is useful as an early-warning system — you can catch the feature shift via
Evidently *before* you have enough new fraud labels to directly measure the recall drop.

**Target drift** wasn't triggered in this simulation (fraud rate stayed ~0.167% in both, since I
only shifted features, not labels) — noted honestly rather than faked, since a real target drift
example would need the actual fraud rate to change (e.g., a holiday fraud spike).

## Step 20 — Auto-Retraining Pipeline (`src/retrain_pipeline.py`)

A real, runnable script implementing the full decision loop: New Data → Validation → Feature
Engineering → Training → Evaluation → "Better than Production?" → Register+Promote or Stop.

**Ran it three times, real decisions each time, not scripted outcomes:**

| Scenario | New data | Candidate PR-AUC | vs. Production (0.8262) | Decision |
|---|---|---|---|---|
| A — smaller fresh batch | 56,746 rows | 0.7577 | -0.0685 | **Stopped** — correctly declined |
| B — drifted batch (Step 19's simulation) | 56,746 rows | 0.7808 | -0.0453 | **Stopped** — correctly declined |
| C — larger combined dataset | 283,726 rows | 0.8582 | **+0.0321** | **Promoted** — registered as v4, now Production |

All three decisions were genuinely computed, not predetermined — Scenario A/B lost because less
training data (and drift, for B) produces a worse model; Scenario C won because more real labeled
data produces a better one. The registry now shows `v4: Production`, `v3: Archived`, exactly
reflecting that real outcome.

**Scheduled retraining** — this script would typically run via a cron job or a scheduled GitHub
Actions workflow (e.g. `.github/workflows/ci.yml` already exists for the build/test pipeline; a
`retrain.yml` on a weekly `schedule:` trigger would call this script the same way). Not added here
since it needs a live schedule to mean anything — documenting the mechanism rather than faking a
cron job that would never actually fire in this sandbox.

**Approval workflow** — `PERFORMANCE_THRESHOLD` in the script is currently `0.0` (any genuine
improvement promotes automatically). A stricter, more realistic production policy would set this
higher (e.g. `0.01`) to avoid promoting on noise, and/or require a human to manually approve the
`Staging → Production` transition rather than fully auto-promoting — both are one-line changes to
the script, documented in its comments.

## Step 21 — AWS Deployment (`deploy/`)

**Not deployed or tested** — this sandbox has no AWS access. What's real: correct, ready-to-use
deployment artifacts:
- `deploy/ecr_push.sh` — builds the Docker image and pushes it to ECR
- `deploy/ecs-task-definition.json` — valid JSON (verified), a real Fargate task definition including
  a health check wired to the API's `/health` endpoint
- `deploy/AWS_DEPLOYMENT_GUIDE.md` — both the EC2 (simplest) and ECS/Fargate (production-shaped)
  paths, with exact commands

You'd need to run these yourself in a real AWS account to confirm end to end — but there's nothing
speculative in the commands themselves, they're standard AWS CLI/ECS syntax.
