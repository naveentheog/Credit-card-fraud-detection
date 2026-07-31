# Random Forest

## Overview

Random Forest is an ensemble learning method that builds many decision trees and combines their
predictions (majority vote for classification, average for regression). It belongs to the family of
**bagging** (Bootstrap Aggregating) algorithms.

**Beginner level:** Instead of trusting one decision tree (which tends to overfit), build hundreds of
slightly-different trees and let them vote. The "wisdom of the crowd" smooths out individual trees'
mistakes.

**Intermediate level:** Each tree is trained on a bootstrap sample (random sample with replacement) of
the training data, AND at each split, only a random subset of features is considered. This double
randomness (row sampling + feature sampling) is what decorrelates the trees from each other, which is
essential — averaging correlated trees doesn't reduce variance much, averaging decorrelated trees does.

**Advanced level:** Random Forest is fundamentally a **variance-reduction** technique — individual deep
decision trees have low bias but high variance (they overfit training data easily). Bagging averages
away that variance while keeping bias roughly the same as a single tree, because averaging many
unbiased-but-noisy estimators reduces noise without introducing new bias (assuming trees are not
perfectly correlated).

## Theory

**Bootstrap sampling:** For N training examples, each tree is trained on a sample of N examples drawn
*with replacement* from the training set. On average, each bootstrap sample contains about 63.2% of
the unique original examples (the rest are duplicates); the remaining ~36.8% not selected for a given
tree are called **out-of-bag (OOB)** samples, and can be used as a free validation set for that tree.

**Feature randomness:** At each split in each tree, only a random subset of features (commonly
`sqrt(n_features)` for classification) is considered as split candidates, not all features. This
prevents a few strong features from dominating every tree's early splits, which would otherwise make
all trees look similar.

**Aggregation:** For classification, the forest's prediction is the majority vote across trees (or the
average predicted probability, which `sklearn` uses by default via `predict_proba`).

**Why variance reduction works — intuition:**
If you average `T` independent, identically distributed random variables each with variance `σ²`, the
variance of the average is `σ²/T`. Trees aren't fully independent (they're trained on overlapping,
correlated data), so the real benefit is smaller than a naive `1/T` reduction, but it's still
substantial — this is exactly why the row+feature randomness matters: more independence between trees
means closer to the ideal `σ²/T` reduction.

**Strengths:**
- Handles non-linear relationships and feature interactions automatically (unlike Logistic Regression)
- Robust to outliers and doesn't require feature scaling
- Provides feature importance out of the box
- Less prone to overfitting than a single decision tree, and less sensitive to hyperparameter tuning
  than boosting methods (a "reasonably good" Random Forest is easy to get; a "reasonably good" XGBoost
  needs more tuning)
- Handles missing values and mixed data types reasonably well in many implementations

**Weaknesses:**
- Less interpretable than Logistic Regression (though feature importances help)
- Can still overfit on very noisy data if trees are too deep and there are too few for the noise level
- Generally underperforms gradient boosting (XGBoost/LightGBM) on structured/tabular data when
  boosting is properly tuned, because bagging doesn't correct previous trees' mistakes the way boosting
  does
- Slower to predict than a single tree or linear model (has to query every tree)
- Can struggle with extreme class imbalance similarly to other classifiers unless class_weight or
  resampling is used

**Common misconceptions:**
- "More trees always means better accuracy" — false beyond a point; more trees reduce variance but
  performance plateaus, while training/prediction time keeps growing.
- "Random Forest can't overfit" — false; it's more resistant than a single tree, but still can, especially
  with very deep trees on small/noisy datasets.
- "Feature importance from Random Forest is unbiased" — not quite; default (Gini/impurity-based)
  importances are biased toward high-cardinality features. Permutation importance is more reliable, a
  legitimate senior-level point to raise.

## Interview Questions

**Beginner**
1. What is a Random Forest, in one sentence?
2. What is bagging, and how does Random Forest use it?
3. Why does Random Forest use multiple trees instead of one?
4. What's the difference between Random Forest and a single Decision Tree?

**Intermediate**
5. What is bootstrap sampling, and why is it used?
6. What role does feature randomness play at each split, and why is it necessary in addition to row
   sampling?
7. What is an Out-of-Bag (OOB) score, and how is it useful?
8. How does Random Forest compute feature importance, and what's a limitation of that method?
9. Does Random Forest need feature scaling? Why or why not?
10. How does Random Forest handle missing values?

**Advanced / Mathematical**
11. Explain why averaging decorrelated estimators reduces variance more than averaging correlated ones.
12. Explain the bias-variance tradeoff in the context of a single Decision Tree vs. a Random Forest.
13. Why doesn't increasing the number of trees cause Random Forest to overfit, unlike increasing depth?
14. Compare Gini impurity vs. entropy as splitting criteria — do they usually give very different trees?

**Practical / Scenario-based (Fraud-specific)**
15. Your Random Forest gets great training accuracy but mediocre test recall on fraud. What do you check
    first?
16. How would you use Random Forest's feature importances to justify which features matter most to a
    fraud investigation team?
17. Would you use class_weight, SMOTE, or both with Random Forest for this fraud dataset? Justify your
    choice.
18. Compare how Random Forest and XGBoost would behave differently if a new, previously unseen type of
    fraud pattern appeared in production data.

**Coding**
19. Write code to train a Random Forest with class weighting and extract the top 10 feature
    importances.
20. Write code to compute permutation importance instead of default impurity-based importance, and
    explain why you'd prefer it.

**Follow-ups an interviewer might throw at you**
- "If Random Forest and XGBoost both use decision trees, why does XGBoost usually win on tabular data?"
  (Answer: boosting sequentially corrects errors of prior trees — a targeted, iterative process — while
  bagging just averages independently-built trees; boosting typically achieves lower bias for the same
  variance budget, if regularized properly to avoid overfitting.)
- "Would you trust feature importance from Random Forest to remove features?" (Answer: cautiously —
  default impurity-based importance is biased toward continuous/high-cardinality features; prefer
  permutation importance or SHAP for a trustworthy ranking before dropping anything.)

## Answers

**Q5 — What is bootstrap sampling, and why is it used?**
Bootstrap sampling means drawing a sample of size N from your N training examples *with replacement*,
so some examples appear multiple times and some don't appear at all in a given tree's training set. It's
used to give each tree in the forest a slightly different view of the data — this diversity is what
allows averaging across trees to actually reduce variance. Without it, every tree would see the exact
same data and learn nearly identical splits (assuming no other randomness), making the "ensemble" no
better than one tree.

**Q11 — Why does averaging decorrelated estimators reduce variance more than correlated ones?**
For `T` estimators each with variance `σ²` and pairwise correlation `ρ`, the variance of their average is:
`ρσ² + (1-ρ)σ²/T`. As `T → ∞`, this approaches `ρσ²` — NOT zero. So if trees are highly correlated
(`ρ` close to 1), adding more trees barely helps beyond a point, because the `ρσ²` term dominates. This
is exactly why Random Forest deliberately decorrelates trees via bootstrap sampling AND random feature
subsets at each split — lowering `ρ` is what makes the `1/T` variance reduction actually kick in. This
formula is a favorite "prove you understand the math" interview question at more quantitative shops.

**Q15 — Great training accuracy but mediocre test recall on fraud — what do you check first?**
First, check whether this is genuine overfitting (trees too deep, too few trees, or insufficient
randomness) versus a class imbalance masking issue — verify by looking at training vs. test
precision/recall/PR-AUC specifically for the fraud class, not overall accuracy. If train recall on
fraud is high but test recall is low, that's classic overfitting — the model memorized specific fraud
examples' noise rather than general patterns; the fix is `max_depth` limiting, increasing
`min_samples_leaf`, or adding more trees with regularization. If BOTH train and test recall on fraud
are low despite high accuracy, the issue is more likely that class imbalance is being ignored entirely
and the model defaults toward the majority class — check `class_weight` settings first in that case.

## Project Connection

In this project, Random Forest sits between Logistic Regression and XGBoost — more expressive than the
linear baseline (captures non-linear interactions among the PCA'd V1-V28 features automatically,
without manual feature engineering), while being noticeably more forgiving to tune than XGBoost (fewer
hyperparameters are truly critical to get right).

Concretely:
- Trained with `class_weight='balanced'` and compared against SMOTE-resampled training data, mirroring
  the comparison done for Logistic Regression, to see whether resampling or weighting works better for
  a tree ensemble specifically (they don't always behave the same way as linear models here).
- Its feature importances are compared against XGBoost's and against SHAP values later, partly as a
  sanity check ("do multiple methods agree on which V-features matter most for fraud?") and partly
  because "which features matter most" is a direct answer to the kind of business question
  Amex/EY-Parthenon ask about explaining a model's decisions.
- Serves as the "did tree ensembles help over linear models, and did boosting help further over
  bagging" two-step comparison story, which is a clean narrative for an interview walkthrough.

## Code Examples

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# No scaling needed - tree-based models split on raw feature values
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=5,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42,
    oob_score=True   # free validation estimate from out-of-bag samples
)
model.fit(X_train, y_train)

print("OOB score:", model.oob_score_)

y_proba = model.predict_proba(X_test)[:, 1]

# Default (impurity-based) feature importance - fast but biased toward
# high-cardinality/continuous features
import pandas as pd
default_importance = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print(default_importance.head(10))

# Permutation importance - slower, but more trustworthy: measures actual
# performance drop when a feature's values are shuffled
perm_result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
perm_importance = pd.Series(perm_result.importances_mean, index=X_train.columns).sort_values(ascending=False)
print(perm_importance.head(10))
```

## Visual Explanation

```
        Training Data
             │
   ┌─────────┼─────────┬─────────┐
   ▼         ▼         ▼         ▼
 Bootstrap Bootstrap Bootstrap Bootstrap
 Sample 1  Sample 2  Sample 3  Sample N     (rows sampled w/ replacement)
   │         │         │         │
   ▼         ▼         ▼         ▼
 Tree 1    Tree 2    Tree 3    Tree N       (each split only considers a
 (random   (random   (random   (random       random subset of features)
 features) features) features) features)
   │         │         │         │
   └─────────┴────┬────┴─────────┘
                   ▼
            Majority Vote /
          Average Probability
                   │
                   ▼
            Final Prediction
```

## Common Mistakes

- **Assuming more trees always helps** — performance plateaus; wastes compute/latency budget past a
  point without meaningfully improving predictions.
- **Not setting `max_depth` or `min_samples_leaf`** — fully-grown trees on noisy data (like anonymized
  fraud features) can overfit individual noisy points.
- **Trusting default feature importance blindly** for business decisions — it's biased toward
  continuous/high-cardinality features (V1-V28 are all continuous, so this bias applies directly here);
  permutation importance or SHAP is the more defensible choice for reporting to stakeholders.
- **Forgetting `class_weight='balanced'`** (or equivalent resampling) — Random Forest defaults toward
  majority class predictions just like any classifier under severe imbalance.
- **Confusing OOB score with a proper held-out test set** — OOB is convenient and free, but it's not a
  substitute for a genuinely separate test set, especially if there's any temporal structure to
  transactions that a random bootstrap wouldn't respect.

## Advanced Discussion

- **Random Forest vs. Extremely Randomized Trees (Extra-Trees):** Extra-Trees add even more randomness
  by choosing split thresholds randomly rather than optimizing them, trading a bit more bias for less
  variance and faster training — worth mentioning if an interviewer probes "what Random Forest
  variants do you know."
- **Why bagging doesn't reduce bias:** Since every tree is grown on data drawn from the same
  distribution as the original training set, each tree has roughly the same bias as a single
  fully-grown tree; averaging only helps with variance, not bias. This is precisely why Random Forest
  doesn't typically beat well-tuned boosting on bias-limited problems — boosting explicitly targets
  reducing bias by fitting residuals.
- **Correlation-variance tradeoff as a design lever:** The `max_features` parameter directly controls
  the ρ (correlation) term in the earlier variance formula — lowering `max_features` decorrelates trees
  more (helps variance) but can increase each tree's individual bias since it's choosing splits from a
  smaller candidate pool. A senior interviewer may push you to articulate this tradeoff explicitly
  rather than just citing it as a "hyperparameter to tune."

## Revision Notes

- Random Forest = bagging (bootstrap row sampling) + random feature subsets per split, over many trees
- Reduces variance, doesn't really reduce bias — a variance-reduction technique
- No feature scaling needed; handles non-linearity and interactions automatically
- OOB score gives a free validation estimate
- Default feature importance is biased toward high-cardinality features — use permutation importance
  or SHAP for trustworthy rankings
- Generally a strong "reasonably good with little tuning" model; usually beaten by well-tuned boosting
  on tabular data
- In this project: the "did non-linearity help over Logistic Regression" checkpoint before XGBoost

## References

- *The Elements of Statistical Learning* — Ch. 15, Random Forests (Breiman's original variance formula
  is derived here)
- Breiman, L. (2001). "Random Forests." *Machine Learning*, 45(1), 5-32 — the original paper
- scikit-learn documentation: `sklearn.ensemble.RandomForestClassifier`, and the User Guide section on
  permutation importance vs. impurity-based importance
- *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (Géron) — Ch. 7, Ensemble
  Learning and Random Forests
