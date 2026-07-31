# Hyperparameter Tuning

## Overview

Hyperparameters are the settings you choose *before* training (e.g. `max_depth`, `learning_rate`,
`n_estimators`) as opposed to parameters the model *learns* from data (e.g. tree splits, coefficient
weights). Hyperparameter tuning is the process of systematically searching for the combination of
these settings that produces the best-generalizing model.

**Beginner level:** Instead of guessing values, try a bunch of combinations and keep the one that
performs best on validation data.

**Intermediate level:** Different search strategies (Grid Search, Random Search, Bayesian Optimization)
trade off exhaustiveness, computational cost, and how efficiently they find good regions of the search
space, especially as the number of hyperparameters grows.

**Advanced level:** Tuning is itself an optimization problem over a black-box, expensive-to-evaluate
function (model performance as a function of hyperparameters). Bayesian optimization methods like
Optuna's default sampler (TPE — Tree-structured Parzen Estimator) build a probabilistic model of which
regions of hyperparameter space are promising, based on past trials, rather than searching blindly or
exhaustively — making them far more sample-efficient for expensive models like XGBoost on large
datasets.

## Theory

**Grid Search:** Define a fixed set of values per hyperparameter, try every combination. Exhaustive but
scales exponentially with the number of hyperparameters ("curse of dimensionality") — tuning 4
hyperparameters with 5 values each is `5^4 = 625` combinations.

**Random Search:** Sample random combinations from defined distributions instead of a fixed grid.
Counter-intuitively often outperforms Grid Search for the same compute budget, because it explores more
distinct values per individual hyperparameter (Bergstra & Bengio, 2012 showed this formally) — if only
2-3 hyperparameters actually matter much, random search is much more likely to hit good values for
those specific ones than a grid constrained to a few fixed levels.

**Bayesian Optimization (what Optuna primarily uses — TPE):** Builds two probability distributions over
hyperparameter values — one for trials that performed well, one for trials that performed poorly — and
uses the ratio between them to decide which region to sample next. Each new trial is informed by all
previous trials, focusing search effort on promising regions rather than sampling uniformly at random.
This matters a lot when each trial (training XGBoost on 200K+ rows) is expensive — you want every trial
to count.

**Cross-validation during tuning:** Hyperparameters should never be tuned against the final test set —
that would leak test set information into model selection. Standard practice: k-fold cross-validation
on the training set (or a dedicated validation set) to evaluate each hyperparameter combination, keeping
the test set completely untouched until final evaluation.

**Early stopping as a tuning tool:** For boosting models specifically, early stopping (monitoring
validation loss/metric during training itself, stopping when it stops improving) is a cheap way to tune
the *effective* number of boosting rounds without a separate search over `n_estimators` — it's found
automatically per hyperparameter combination.

**Strengths of Bayesian/Optuna approach:**
- Sample-efficient — fewer trials needed to reach a good result compared to grid/random search
- Supports pruning (stopping unpromising trials early, before they finish training) which saves massive
  compute on expensive models
- Handles mixed parameter types (continuous, integer, categorical) naturally

**Weaknesses:**
- More complex to set up and reason about than grid/random search
- Sequential by nature (each trial depends on previous ones) which limits parallelization compared to
  grid/random search, though Optuna does support some parallel strategies
- Can get stuck in a locally good region if the search space or trial budget is poorly configured

**Common misconceptions:**
- "More hyperparameter tuning always improves test performance" — false beyond a point; over-tuning on
  a validation set can itself overfit to that validation set's specific noise, especially with a small
  validation set or too many trials.
- "Grid Search is more thorough so it's always better" — not necessarily, for a fixed compute budget,
  Random Search often finds comparable or better results by covering more distinct hyperparameter
  values.
- "Bayesian optimization guarantees the global optimum" — no, it's still a heuristic search over a
  non-convex space; it's more efficient, not guaranteed-optimal.

## Interview Questions

**Beginner**
1. What's the difference between a parameter and a hyperparameter?
2. What is Grid Search, and what's its main drawback?
3. What is Random Search, and why can it outperform Grid Search?
4. Why shouldn't you tune hyperparameters using your test set?

**Intermediate**
5. Explain how cross-validation is used during hyperparameter tuning.
6. What is early stopping, and how does it interact with hyperparameter tuning for boosting models?
7. What's "pruning" in the context of Optuna, and why does it save compute?
8. How would you choose which hyperparameters are worth tuning for XGBoost, given limited time/compute?

**Advanced / Mathematical**
9. Explain, at a conceptual level, how Bayesian optimization / TPE decides which hyperparameter values
   to try next.
10. Why is Bayesian optimization more sample-efficient than Random Search for expensive-to-train
    models?
11. How would you detect if your hyperparameter tuning process is itself overfitting to the validation
    set?

**Practical / Scenario-based (Fraud-specific)**
12. You have limited compute and need to tune XGBoost on 200K+ transactions. How would you structure
    the search to be efficient (search space, number of trials, use of pruning/early stopping)?
13. Which XGBoost hyperparameters would you prioritize tuning first for a severely imbalanced fraud
    dataset, and why?
14. Your tuned model has excellent cross-validation PR-AUC but noticeably worse performance on the
    held-out test set. What could explain this gap?

**Coding**
15. Write an Optuna study that tunes XGBoost's `max_depth`, `learning_rate`, and `n_estimators`,
    optimizing for PR-AUC via cross-validation.
16. Write code that uses Optuna's pruning callback to stop unpromising trials early.

**Follow-ups an interviewer might throw at you**
- "Why not just use the defaults? XGBoost's defaults are pretty good." (Answer: defaults are reasonable
  general-purpose starting points, but for a highly imbalanced, high-stakes problem like fraud, the
  tuning cost is worth it to specifically optimize PR-AUC/recall-at-fixed-precision rather than the
  generic defaults tuned for balanced datasets.)
- "How many trials is 'enough' for Optuna?" (Answer: no fixed number — depends on search space size and
  budget; a common practical approach is to run until the best score plateaus over a rolling window of
  trials, or until compute/time budget is exhausted, using pruning to make each trial cheaper.)

## Answers

**Q3 — Why can Random Search outperform Grid Search?**
If you have, say, 4 hyperparameters but really only 2 of them meaningfully affect performance, Grid
Search wastes most of its trials varying the unimportant 2 while only trying a handful of distinct
values for the important 2 (because the grid is fixed and shared across all dimensions). Random Search,
by sampling each hyperparameter independently from its own distribution, ends up trying many more
distinct values for every hyperparameter — including the important ones — for the same total number of
trials. Bergstra & Bengio (2012) demonstrated this empirically and it's become a standard argument for
preferring random over grid search when compute is limited.

**Q9 — How does Bayesian optimization / TPE decide what to try next?**
TPE splits past trials into two groups based on a performance threshold: "good" trials (top some
percentage) and "bad" trials (the rest). It then fits two probability density estimates — `l(x)` for
the good group and `g(x)` for the bad group — over the hyperparameter values that led to each. The next
hyperparameter combination to try is chosen to maximize the ratio `l(x)/g(x)` — roughly, "values that
were common among good trials but uncommon among bad trials." This is why it's more sample-efficient
than random search: after even a modest number of trials, it has a real (if imperfect) model of which
regions of the space are promising, and concentrates search effort there instead of spending trials
uniformly across the whole space.

**Q14 — Great CV PR-AUC, worse test PR-AUC — what explains the gap?**
A few candidate explanations, roughly in order of how commonly they're the actual cause: (1) data
leakage during cross-validation — e.g., if SMOTE was applied *before* the CV split rather than inside
each fold, synthetic points derived from what should be held-out data leak into training folds,
inflating CV scores unrealistically; (2) the tuning process itself overfitting to the validation
folds — with enough Optuna trials, you can find a hyperparameter combination that happens to fit the
specific noise in your CV folds, which is why final evaluation must be on a completely untouched test
set; (3) genuine distribution shift between the CV folds and test set, if the split wasn't properly
stratified or if there's temporal structure in the data that a random split ignores. The fix for (1) is
always resampling inside each CV fold's training portion only, never before the split.

## Project Connection

Hyperparameter tuning is applied specifically to XGBoost in this project (Random Forest and Logistic
Regression are left closer to sensible defaults / lightly tuned, since XGBoost is both the primary
model and the most sensitive to hyperparameter choices). Concretely:

- An Optuna study searches over `max_depth`, `learning_rate`, `n_estimators` (via early stopping rather
  than direct search), `subsample`, `colsample_bytree`, and `reg_lambda`.
- The objective function being optimized is **PR-AUC via stratified k-fold cross-validation on the
  training set**, not accuracy — consistent with the evaluation philosophy used throughout this project.
- SMOTE, when used, is applied *inside* each cross-validation fold (fit only on that fold's training
  portion), not before the CV split — directly avoiding the leakage bug described in the Answers section
  above, which is a mistake worth being able to say out loud that you specifically avoided.
- The best hyperparameter combination and its CV score are logged to MLflow (Step 8) alongside every
  individual trial, so the tuning process itself is auditable, not just the final chosen configuration.

## Code Examples

```python
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
        'scale_pos_weight': scale_pos_weight,   # computed once from training data, not tuned
        'eval_metric': 'aucpr',
        'random_state': 42,
    }

    model = XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='average_precision', n_jobs=-1)
    return scores.mean()

study = optuna.create_study(
    direction='maximize',
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
)
study.optimize(objective, n_trials=50, timeout=3600)  # 50 trials or 1 hour, whichever first

print("Best PR-AUC:", study.best_value)
print("Best params:", study.best_params)
```

## Visual Explanation

```
Grid Search (3x3 grid = 9 trials, fixed points):
  learning_rate
   0.3 │  x     x     x
   0.1 │  x     x     x
  0.01 │  x     x     x
       └──────────────────
          3     6     9   max_depth

Random Search (9 trials, sampled freely):
  learning_rate
   0.3 │      x  x
   0.1 │   x        x  x
  0.01 │ x    x        x
       └──────────────────
          3     6     9   max_depth
  (covers more distinct values along EACH axis individually)

Bayesian/TPE (9 trials, later trials cluster near earlier good results):
  learning_rate
   0.3 │
   0.1 │        x x
       │       x x x     <- search concentrates here after
  0.01 │  x  x            early trials showed this region is promising
       └──────────────────
          3     6     9   max_depth
```

## Common Mistakes

- **Tuning on the test set** — even "just checking" test performance during tuning and adjusting the
  search space in response is a leakage, whether or not the test set is directly used in the objective
  function.
- **Applying SMOTE before cross-validation splitting** — leaks synthetic-derived signal across folds,
  inflating CV scores in a way that won't hold up on genuinely new data.
- **Optimizing for accuracy or plain ROC-AUC** on a severely imbalanced dataset — the tuning process
  will happily find hyperparameters that look great on the wrong metric.
- **Too few CV folds or too small a validation set** relative to the number of Optuna trials — with
  enough trials, you can "get lucky" and overfit to a small validation set's specific noise, similar to
  how a model can overfit to training data.
- **Not using pruning** for expensive models — wastes enormous compute finishing trials that were
  clearly unpromising after just a few boosting rounds.

## Advanced Discussion

- **Nested cross-validation:** For a fully rigorous estimate of "how good is my hyperparameter tuning
  process, not just my final model," nested CV runs an outer CV loop for performance estimation and an
  inner CV loop for hyperparameter selection within each outer fold. Rarely done in industry due to
  compute cost, but a senior interviewer may ask if you know why it exists and why it's often skipped
  practically.
- **Multi-objective tuning:** In a real fraud system, you might not want to optimize PR-AUC alone —
  inference latency also matters (a model that takes 200ms per prediction may be unacceptable at
  transaction time). Optuna supports multi-objective optimization (e.g., maximize PR-AUC while
  minimizing model size/latency), a good "how would you extend this for production constraints" answer.
- **Warm-starting tuning after retraining:** In an automatic retraining pipeline (as sketched in this
  project's broader architecture), you don't need to re-run hyperparameter search from scratch every
  time new data arrives — Optuna studies can be persisted and continued, using prior trials to warm-
  start the search on the updated dataset, saving significant compute over time.

## Revision Notes

- Hyperparameters are set before training; parameters are learned from data
- Grid Search: exhaustive, expensive, wastes trials on unimportant dimensions
- Random Search: often better than Grid Search for the same budget, still not sample-efficient
- Bayesian Optimization (TPE, used by Optuna): models which regions are promising based on past trials,
  most sample-efficient of the three
- Always tune against cross-validation/validation performance, NEVER the test set
- Resampling (SMOTE) must happen inside each CV fold, not before splitting
- In this project: Optuna tunes XGBoost specifically, optimizing PR-AUC via stratified CV, logged to
  MLflow

## References

- Bergstra, J. & Bengio, Y. (2012). "Random Search for Hyper-Parameter Optimization." *JMLR* — the paper
  formally justifying random over grid search
- Bergstra, J. et al. (2011). "Algorithms for Hyper-Parameter Optimization" — the original TPE paper
- Optuna official documentation — particularly the sections on samplers and pruners
- scikit-learn documentation: `GridSearchCV`, `RandomizedSearchCV`, and the User Guide on cross-
  validation
