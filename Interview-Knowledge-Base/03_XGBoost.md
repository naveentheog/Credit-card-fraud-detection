# XGBoost (Extreme Gradient Boosting)

## Overview

XGBoost is a gradient-boosted decision tree algorithm, and one of the most widely used models for
structured/tabular data in industry — including fraud detection, credit risk, ranking, and recommendation
systems.

**Beginner level:** Instead of building one big tree, or many independent trees like Random Forest,
XGBoost builds trees **sequentially**, where each new tree tries to correct the mistakes of all the
trees built before it.

**Intermediate level:** It's an implementation of gradient boosting that adds regularization (both L1
and L2 on leaf weights), a more efficient split-finding algorithm, and built-in handling for missing
values and class imbalance (`scale_pos_weight`), making it faster and generally more accurate than
plain gradient boosting implementations.

**Advanced level:** XGBoost optimizes a regularized objective using a **second-order Taylor
approximation** (using both gradient and Hessian of the loss), not just the first-order gradient that
basic gradient boosting uses. This gives it more precise information about how to update each tree,
which is part of why it converges faster and often generalizes better than naive boosting.

## Theory

**Boosting, generally:** Start with a simple model (often just predicting the mean/log-odds). Compute
the residual errors. Train a new tree to predict those residuals. Add that tree's predictions (scaled
by a learning rate) to the running total. Repeat for many rounds. Each tree is weak on its own but the
sum becomes strong.

**XGBoost's objective function:**
```
Obj = Σ Loss(y_i, ŷ_i) + Σ Ω(tree_k)
```
where `Ω(tree_k) = γT + (1/2)λ‖w‖²` — a regularization term penalizing the number of leaves `T` and the
magnitude of leaf weights `w`. This is what makes XGBoost less prone to overfitting than plain gradient
boosting — the regularization is baked directly into the objective the trees are built to minimize, not
bolted on afterward.

**Second-order approximation:** At each boosting round, XGBoost approximates the loss function around
the current prediction using both the first derivative (gradient, `g`) and second derivative (Hessian,
`h`) of the loss. This lets it compute an optimal leaf weight in closed form for a candidate split:
```
w* = -G / (H + λ)
```
where `G` and `H` are the sums of gradients and Hessians of examples in that leaf. This closed-form
optimal weight, combined with a "gain" formula for evaluating candidate splits, is what makes tree
construction both fast and precise.

**Handling imbalance:** `scale_pos_weight` parameter multiplies the gradient contribution of positive
(minority/fraud) class examples, effectively telling the algorithm to care more about getting fraud
right during each boosting round, without needing SMOTE.

**Key hyperparameters:**
- `n_estimators` — number of boosting rounds (trees)
- `max_depth` — depth of each tree (shallower trees, typically 3-8, are standard in boosting, unlike
  Random Forest which can use deeper trees)
- `learning_rate` (`eta`) — shrinks each tree's contribution; lower values need more trees but
  generalize better
- `subsample` — fraction of rows sampled per tree (adds randomness, like bagging, to reduce overfitting)
- `colsample_bytree` — fraction of features sampled per tree
- `scale_pos_weight` — class imbalance handling
- `reg_alpha` (L1) / `reg_lambda` (L2) — regularization on leaf weights

**Strengths:**
- Usually the best-performing model on tabular/structured data among "classical" ML methods
- Handles non-linear relationships and feature interactions natively
- Built-in regularization reduces overfitting risk compared to plain boosting
- Handles missing values natively (learns a default split direction)
- Fast, parallelized tree construction; efficient enough for very large datasets
- Provides multiple types of feature importance (gain, cover, frequency) plus native SHAP support

**Weaknesses:**
- More hyperparameters to tune than Random Forest; more sensitive to bad hyperparameter choices
  (can overfit badly with too many rounds and too high a learning rate)
- Sequential tree-building is inherently harder to parallelize across trees than bagging (though
  within-tree construction is parallelized)
- Less interpretable than Logistic Regression by default (though SHAP largely closes this gap)
- Can still overfit on very small or very noisy datasets if not regularized properly

**Common misconceptions:**
- "XGBoost always beats Random Forest" — usually true on tabular data with proper tuning, but not
  guaranteed; on very small or very noisy datasets, Random Forest's simplicity can generalize better.
- "More boosting rounds is always better" — false; too many rounds without early stopping leads to
  overfitting, since each new tree keeps fitting to the (increasingly small) remaining residuals,
  including noise.
- "XGBoost feature importance ('gain') is unbiased" — like Random Forest's default importance, gain-
  based importance in XGBoost has known biases and is best supplemented with SHAP for
  reliable, per-prediction explanations.

## Interview Questions

**Beginner**
1. What is gradient boosting, in plain terms?
2. How is XGBoost different from a single Decision Tree?
3. How is XGBoost different from Random Forest at a high level?
4. What does `n_estimators` control?

**Intermediate**
5. What does `learning_rate` do, and why not just set it to 1.0?
6. What is `scale_pos_weight`, and how would you set it for this fraud dataset?
7. Explain `subsample` and `colsample_bytree` — why would boosting need randomness like bagging does?
8. What's the difference between `gain`, `cover`, and `weight` feature importance in XGBoost?
9. Why does XGBoost typically use shallower trees (`max_depth` 3-8) compared to Random Forest?

**Advanced / Mathematical**
10. Explain the second-order Taylor approximation XGBoost uses, and why it's an improvement over
    first-order gradient boosting.
11. Derive (conceptually) the optimal leaf weight formula `w* = -G/(H+λ)`.
12. How does XGBoost's regularization term `Ω(tree) = γT + (1/2)λ‖w‖²` prevent overfitting compared to
    unregularized boosting?
13. Explain how XGBoost evaluates whether a split is "worth it" using the gain formula.

**Practical / Scenario-based (Fraud-specific)**
14. You trained XGBoost with 1000 rounds and no early stopping — training PR-AUC is near 1.0, but test
    PR-AUC is much lower. Diagnose and fix.
15. Would you use `scale_pos_weight`, SMOTE, or both with XGBoost? Justify the choice specifically for
    XGBoost (not just imbalance in general).
16. How would you tune XGBoost's hyperparameters efficiently, given training on 200K+ transactions is
    not instant?
17. Fraud patterns shift over time (concept drift). How would this affect a deployed XGBoost model,
    and what would you monitor?

**Coding**
18. Write code to train an XGBoost classifier with early stopping on a validation set.
19. Write code to extract SHAP values for a single prediction and interpret them.

**Follow-ups an interviewer might throw at you**
- "Why does XGBoost need a validation set and early stopping, but Random Forest doesn't as urgently?"
  (Answer: because boosting keeps improving training fit round after round — without a stopping
  criterion tied to validation performance, it will keep fitting until it overfits; Random Forest's
  trees are independent, so more trees mainly reduce variance, not increase overfitting risk the same
  way.)
- "If XGBoost is more prone to overfitting, why is it still usually preferred over Random Forest?"
  (Answer: with proper regularization and early stopping, it typically achieves both lower bias AND
  competitive variance, netting a better bias-variance tradeoff overall — the overfitting risk is
  manageable with standard practices, and the accuracy ceiling is usually higher.)

## Answers

**Q5 — What does `learning_rate` do, and why not just set it to 1.0?**
`learning_rate` (often called `eta`) scales down the contribution of each new tree before adding it to
the running prediction: `new_prediction = old_prediction + learning_rate * new_tree_prediction`. Setting
it to 1.0 means each tree fully "trusts" its own correction, which tends to overfit quickly and can
overshoot the optimal solution, similar to using too large a step size in gradient descent. A smaller
learning rate (e.g. 0.01-0.1) forces the model to take smaller, more conservative steps, needing more
boosting rounds to converge but generally reaching a better-generalizing solution — this is the classic
learning_rate/n_estimators tradeoff, and it's why they're almost always tuned together, not
independently.

**Q10 — Explain the second-order Taylor approximation and why it's an improvement.**
Basic gradient boosting only uses the first derivative (gradient) of the loss function to decide how to
fit each new tree — essentially just the direction of steepest descent. XGBoost additionally uses the
second derivative (Hessian), which captures the *curvature* of the loss function, giving it information
about how confident to be in that gradient direction — similar to how Newton's method converges faster
than plain gradient descent by using curvature information. Practically, this lets XGBoost compute a
mathematically optimal leaf weight for each candidate split in closed form, rather than approximating
it iteratively, which contributes to both faster convergence and often better final performance.

**Q14 — Training PR-AUC near 1.0, test PR-AUC much lower — diagnose and fix.**
This is overfitting from too many boosting rounds without regularization or early stopping. Each
additional round fits the residuals more and more precisely, including noise specific to the training
set, especially since the training set was SMOTE-resampled (synthetic points can be over-fit to as
well). Fix: (1) hold out a validation set and use early stopping, monitoring validation PR-AUC, stopping
when it stops improving; (2) reduce `max_depth` and increase `reg_lambda`/`reg_alpha`; (3) lower the
learning rate and correspondingly reduce max rounds; (4) double check whether SMOTE was applied before
or after the train/validation split — if applied before, that's an additional, more severe leakage bug
causing this exact symptom.

## Project Connection

XGBoost is the primary model in this project — the one most likely to become the actual deployed
scorer, and the one specifically named in Amex, Oracle, and Point72 interviews from the placement
report. Concretely, in this project:

- Trained on the same feature set as Logistic Regression and Random Forest for a fair three-way
  comparison, using `scale_pos_weight` computed from the training set's actual class ratio as the
  primary imbalance strategy, compared against the SMOTE-resampled version to see whether resampling
  adds anything on top of the built-in weighting.
- Tuned with Optuna (Step 6) rather than a manual grid, given how sensitive XGBoost is to
  `max_depth`/`learning_rate`/`n_estimators` interactions.
- Evaluated primarily on PR-AUC (Step 7), not accuracy or even plain ROC-AUC, given the severe class
  imbalance.
- Its SHAP values (built in natively, via `model.get_booster()` compatibility with the `shap` library)
  become the main explainability layer for individual fraud flags — directly answering "how would you
  explain a specific flagged transaction to a fraud analyst or a customer" style questions.

## Code Examples

```python
from xgboost import XGBClassifier

# scale_pos_weight = (# negative examples) / (# positive examples)
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
scale_pos_weight = neg / pos

model = XGBClassifier(
    n_estimators=1000,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    reg_lambda=1.0,
    eval_metric='aucpr',   # PR-AUC, not accuracy or plain AUC
    early_stopping_rounds=30,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print("Best iteration:", model.best_iteration)

y_proba = model.predict_proba(X_test)[:, 1]

# SHAP values for explainability
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test.iloc[:100])
shap.summary_plot(shap_values, X_test.iloc[:100])
```

## Visual Explanation

```
Round 1:  Base prediction (e.g. log-odds of overall fraud rate)
             │
             ▼
          Residuals (actual - predicted)
             │
             ▼
Round 2:  Tree 1 fits the residuals ──► scaled by learning_rate ──► added to prediction
             │
             ▼
          New residuals (smaller, hopefully)
             │
             ▼
Round 3:  Tree 2 fits THESE residuals ──► scaled ──► added
             │
             ▼
            ... repeat for n_estimators rounds ...
             │
             ▼
     Final prediction = base + Σ (learning_rate × tree_k's output)

Unlike Random Forest (parallel, independent trees averaged),
boosting is SEQUENTIAL - each tree depends on all previous trees' mistakes.
```

## Common Mistakes

- **No early stopping / validation monitoring** — the single most common XGBoost overfitting mistake,
  especially dangerous with high `n_estimators` and low `learning_rate` combos.
- **Applying SMOTE when `scale_pos_weight` would suffice** — doing both isn't necessarily wrong, but
  candidates often do it without justifying why, or without testing whether `scale_pos_weight` alone
  was already sufficient (it's cheaper — no synthetic data risk).
- **Tuning hyperparameters independently instead of jointly** — `learning_rate` and `n_estimators` are
  tightly coupled; tuning one while holding the other fixed at an arbitrary value gives misleading
  results. Optuna/joint search avoids this.
- **Using accuracy or plain ROC-AUC as the tuning objective** — ROC-AUC can look deceptively good under
  severe imbalance because the True Negative Rate dominates; PR-AUC is the more honest metric here (see
  the Evaluation Metrics knowledge base file for the full explanation).
- **Ignoring feature importance bias** — gain-based importance can be inflated for high-cardinality
  continuous features; cross-check with SHAP before making claims about "the most important feature."

## Advanced Discussion

- **XGBoost vs. LightGBM vs. CatBoost:** LightGBM uses leaf-wise (rather than level-wise) tree growth
  and histogram-based split finding, making it faster on very large datasets, sometimes at a slight cost
  to overfitting risk on smaller datasets. CatBoost handles categorical features natively via ordered
  target statistics, and uses symmetric trees. A senior interviewer may ask you to justify sticking
  with XGBoost specifically — for a fully-numeric, already-PCA'd dataset like this one, the categorical-
  handling advantages of CatBoost aren't relevant, and dataset size (~285K rows) isn't large enough to
  strongly favor LightGBM's speed advantages, making XGBoost a defensible default choice.
- **DART (Dropout meets boosting):** XGBoost supports a "DART" booster that randomly drops trees during
  training, similar to dropout in neural networks, as an additional regularization technique — a good
  "what else do you know about XGBoost internals" answer.
- **Monotonic constraints:** XGBoost supports forcing certain features to have a monotonically
  increasing/decreasing relationship with the prediction (e.g., "higher Amount_log should never
  decrease fraud probability, all else equal") — relevant in regulated industries like finance where
  model behavior needs to be explainable and defensible to auditors, a genuinely senior-level point.

## Revision Notes

- Gradient boosting builds trees sequentially, each correcting prior trees' residual errors
- XGBoost adds regularization (L1/L2 on leaf weights, `γ` per-leaf penalty) directly into the objective
- Uses second-order (gradient + Hessian) information for more precise, faster-converging tree building
- `scale_pos_weight` handles class imbalance without needing resampling
- Requires early stopping / careful `learning_rate` + `n_estimators` tuning to avoid overfitting
- Evaluate with PR-AUC under severe imbalance, not accuracy or plain ROC-AUC
- In this project: the primary model, tuned with Optuna, explained with SHAP

## References

- Chen, T. & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." *KDD 2016* — the original
  paper, worth skimming for the exact objective function and split-finding algorithm
- XGBoost official documentation — especially the "Introduction to Boosted Trees" page
- Friedman, J. (2001). "Greedy Function Approximation: A Gradient Boosting Machine." — the foundational
  gradient boosting paper XGBoost builds on
- *Hands-On Machine Learning* (Géron) — Ch. 7, Gradient Boosting section
- Lundberg, S. & Lee, S-I. (2017). "A Unified Approach to Interpreting Model Predictions" (SHAP paper)
