# Evaluation Metrics for Imbalanced Classification

## Overview

Evaluation metrics quantify how good a model actually is — but the right metric depends entirely on
the problem. For severely imbalanced problems like fraud detection (0.17% positive class), the
"obvious" metric (accuracy) is actively misleading, and choosing the right alternative is one of the
most-tested concepts in ML interviews for this domain.

**Beginner level:** Accuracy = fraction of correct predictions. Sounds reasonable, but breaks down
when one class vastly outnumbers the other.

**Intermediate level:** Precision, Recall, and F1 give class-specific views of performance. ROC-AUC and
PR-AUC summarize performance across all possible classification thresholds rather than just one.

**Advanced level:** Under severe imbalance, ROC-AUC can look deceptively high because it's dominated by
the True Negative Rate on the (huge) majority class; PR-AUC is far more sensitive to how the model
handles the minority class, making it the more honest metric for this problem specifically. Threshold
selection itself becomes a business decision tied to the relative cost of false positives vs. false
negatives, not a fixed default.

## Theory

**Confusion Matrix — the foundation everything else is built from:**
```
                  Predicted Legit    Predicted Fraud
Actual Legit      True Negative (TN)  False Positive (FP)
Actual Fraud      False Negative (FN) True Positive (TP)
```

**Accuracy:** `(TP + TN) / (TP + TN + FP + FN)`
With 0.17% fraud, predicting "not fraud" for every single transaction gives 99.83% accuracy while
catching zero fraud. This is why accuracy is essentially never reported as a headline metric for fraud
problems.

**Precision:** `TP / (TP + FP)` — "Of everything I flagged as fraud, what fraction was actually fraud?"
High precision means few false alarms (few legitimate customers wrongly blocked).

**Recall (Sensitivity, True Positive Rate):** `TP / (TP + FN)` — "Of all the actual fraud, what
fraction did I catch?" High recall means little fraud slips through undetected.

**The precision-recall tradeoff:** These two move in opposite directions as you change the
classification threshold. Lowering the threshold (flagging more transactions as fraud) increases recall
(catch more real fraud) but decreases precision (more false alarms), and vice versa. There is no
threshold that maximizes both simultaneously in general — the "right" threshold is a business decision
based on the relative cost of a false positive (annoyed/blocked legitimate customer) versus a false
negative (undetected fraud loss).

**F1 Score:** `2 * (Precision * Recall) / (Precision + Recall)` — the harmonic mean of precision and
recall, giving a single number that penalizes extreme imbalance between the two (unlike a simple
average). Useful when you want one number balancing both concerns roughly equally, but it obscures
*which* of precision or recall is being sacrificed — a real limitation to name in interviews.

**ROC Curve / ROC-AUC:** Plots True Positive Rate (Recall) against False Positive Rate
(`FP / (FP + TN)`) across all thresholds. AUC (Area Under Curve) summarizes this into one number (0.5 =
random, 1.0 = perfect). **Problem under severe imbalance:** the False Positive Rate denominator
(`FP + TN`) is dominated by the huge number of true negatives, so even a large absolute number of false
positives barely moves the FPR, making ROC-AUC look artificially good.

**Precision-Recall Curve / PR-AUC (Average Precision):** Plots Precision against Recall across all
thresholds. Because Precision's denominator (`TP + FP`) does NOT include the huge pool of true
negatives, PR-AUC is much more sensitive to how well the model handles the rare positive class — making
it the standard, more honest metric for severely imbalanced problems like fraud.

**Why ROC-AUC vs PR-AUC diverge under imbalance — concretely:** Imagine 100,000 legit and 100 fraud
transactions. A model with 100 false positives barely moves FPR (`100/100,000 = 0.001`), so ROC-AUC
stays high. But those same 100 false positives could mean precision drops from 90% to 50% if true
positives are also around 100 — a huge, business-relevant difference that PR-AUC reflects and ROC-AUC
largely hides.

**Strengths of PR-AUC/Precision/Recall framing:**
- Directly reflects performance on the class that actually matters (fraud)
- Threshold-independent (PR-AUC) or explicitly threshold-dependent (Precision/Recall at a chosen
  threshold) — both useful depending on whether you're comparing models or setting a production
  operating point

**Weaknesses:**
- PR-AUC has no fixed "baseline" interpretation like ROC-AUC's 0.5 — a random classifier's PR-AUC
  equals the positive class prevalence (0.17% here), which is easy to forget and can make a mediocre
  PR-AUC number look more impressive than it is if not compared to that baseline
- F1 obscures which of precision/recall is being traded away
- None of these metrics account for the actual dollar cost of false positives vs false negatives —
  that requires a cost-sensitive framing on top

**Common misconceptions:**
- "High accuracy means a good model" — false and dangerous under imbalance, the single most common
  interview trap on this topic.
- "ROC-AUC is always a safe general-purpose metric" — not under severe class imbalance; needs to be
  paired with or replaced by PR-AUC.
- "F1 score is always the right single-number summary" — not if precision and recall have genuinely
  different business costs; a weighted Fβ score or a fixed-recall/fixed-precision operating point is
  often more appropriate.

## Interview Questions

**Beginner**
1. Define precision and recall in your own words.
2. Why is accuracy a poor metric for this fraud dataset?
3. What is a confusion matrix?
4. What does F1 score measure, and how is it different from a simple average of precision and recall?

**Intermediate**
5. Explain the precision-recall tradeoff and why changing the classification threshold affects both.
6. What is ROC-AUC, and what does an AUC of 0.5 vs 1.0 mean?
7. What is PR-AUC / Average Precision?
8. Why can ROC-AUC be misleading on a highly imbalanced dataset like this one?

**Advanced / Mathematical**
9. Derive why the random-classifier baseline for PR-AUC equals the positive class prevalence, while for
   ROC-AUC it's always 0.5 regardless of class balance.
10. Explain False Positive Rate vs (1 - Precision) — why do they behave so differently under imbalance?
11. How would you choose between optimizing for F1, PR-AUC, or a fixed-recall operating point, given a
    specific business context?

**Practical / Scenario-based (Fraud-specific)**
12. Your model has 99.9% accuracy and 0.5 recall on fraud. Is this a good model? How do you explain the
    discrepancy to a non-technical stakeholder?
13. The business tells you: "we must catch at least 90% of fraud." How do you translate that into a
    concrete modeling/thresholding decision?
14. Compare two models: Model A has precision 0.9, recall 0.6. Model B has precision 0.6, recall 0.9.
    Which would you recommend for a fraud system, and what additional information would you want before
    deciding?
15. How would you incorporate the actual dollar cost of a false positive (customer friction) vs a false
    negative (fraud loss) into your threshold selection?

**Coding**
16. Write code to plot a precision-recall curve and compute PR-AUC for a trained model.
17. Write code to find the threshold that achieves at least 90% recall while maximizing precision.

**Follow-ups an interviewer might throw at you**
- "If PR-AUC is better here, why do people still report ROC-AUC at all?" (Answer: ROC-AUC is still
  useful for comparing models when class balance might vary or for genuinely balanced problems, and it
  has a fixed, easily-interpretable 0.5 baseline; it's not that ROC-AUC is "wrong," it's that it's the
  wrong primary metric specifically under severe imbalance.)
- "Why not just always maximize recall, since missing fraud is worse than a false alarm?" (Answer:
  taken to the extreme, flagging every single transaction as fraud gives 100% recall but is
  operationally useless — the real answer requires a cost-sensitive threshold reflecting the actual
  relative cost of FP vs FN, not maximizing one metric in isolation.)

## Answers

**Q9 — Why does the random-classifier PR-AUC baseline equal the positive class prevalence?**
A random classifier that assigns scores uniformly at random will, on average, have precision equal to
the overall fraction of positives in the dataset — regardless of the recall level you're evaluating at,
since randomly selecting any subset of transactions preserves the same underlying prevalence. So the
PR curve for a random model is roughly a flat horizontal line at `y = prevalence` (0.0017 here), and
its area under that curve is approximately the prevalence itself. This is why a PR-AUC of, say, 0.3
sounds unremarkable in the abstract but is actually roughly **170x better than random** for a 0.17%
prevalence problem — always report PR-AUC alongside this baseline for it to be interpretable to someone
unfamiliar with the metric. ROC-AUC's baseline, by contrast, is always exactly 0.5 regardless of class
balance, because it's measuring rank-ordering ability (probability a random positive is scored higher
than a random negative), which is a well-defined 50/50 proposition under random guessing no matter how
skewed the classes are.

**Q12 — 99.9% accuracy, 0.5 recall — good model? How to explain to a stakeholder?**
No, this is very likely a poor fraud model despite the impressive-looking accuracy — it's catching only
half of all fraud, meaning half of actual fraud losses are going completely undetected. Explaining to a
non-technical stakeholder: "If we had 1,000 fraudulent transactions last month, this model would have
caught 500 of them and missed 500 — the 99.9% accuracy number is misleading because it's mostly telling
you we correctly identified normal transactions as normal, which is the easy part; almost any model
gets that right given how rare fraud is. The number that actually matters for fraud losses is recall,
and right now it's 50%." This is exactly the kind of translation-to-business-language skill the
placement report flags as being tested directly (Accenture, Amex, EY-Parthenon all probe for this).

**Q14 — Model A (P=0.9, R=0.6) vs Model B (P=0.6, R=0.9) — which for fraud?**
There's no universally correct answer without more context — this is intentionally a "what questions do
you ask before deciding" question, not a "know the right answer" question. Model B catches far more
actual fraud (90% vs 60%) but generates more false alarms (lower precision), meaning more legitimate
customers get flagged/blocked/inconvenienced. Model A is more conservative — fewer false alarms but
misses more fraud. The right choice depends on: the actual dollar cost of a missed fraud case (chargebacks,
liability) vs the cost of a false positive (customer service load, customer churn risk from being
wrongly blocked), whether flagged transactions get an automatic hard block or a soft review step (which
changes the true cost of a false positive substantially), and regulatory/liability requirements specific
to the business. A strong interview answer explicitly asks for this context rather than picking one
model confidently without it.

## Project Connection

Evaluation in this project is built around PR-AUC and precision/recall at multiple thresholds, not
accuracy, for every model (Logistic Regression, Random Forest, XGBoost) and every imbalance-handling
strategy (SMOTE, class weighting, undersampling) compared in Step 5. Concretely:

- `evaluate.py`'s `sweep_thresholds()` function reports precision/recall/F1 at several thresholds
  (0.1 through 0.9) rather than defaulting to a single 0.5 cutoff, since 0.5 is not a meaningful
  business-chosen threshold on its own.
- `best_threshold_for_recall_floor()` directly encodes the "business says we must catch at least 90% of
  fraud" scenario from the interview questions above — translating a stated business requirement into a
  concrete threshold and reporting the precision cost of meeting it.
- PR-AUC (not ROC-AUC) is the primary metric used both for model comparison and as the Optuna tuning
  objective, since ROC-AUC would give a misleadingly rosy picture at 0.17% fraud prevalence.
- Every metric, for every model/strategy combination, is logged to MLflow (Step 8) so the final model
  choice has a complete, comparable evidence trail rather than resting on a single number.

## Code Examples

```python
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt

y_proba = model.predict_proba(X_test)[:, 1]

# PR-AUC (Average Precision) - the primary metric for this project
pr_auc = average_precision_score(y_test, y_proba)
baseline = y_test.mean()  # random-classifier baseline for PR-AUC
print(f"PR-AUC: {pr_auc:.4f}  (random baseline: {baseline:.4f}, "
      f"{pr_auc/baseline:.1f}x better than random)")

# For comparison - ROC-AUC, with the caveat about imbalance noted
roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC: {roc_auc:.4f}  (use PR-AUC as primary given severe imbalance)")

# Full precision-recall curve
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
plt.plot(recalls, precisions)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve (PR-AUC = {pr_auc:.3f})')
plt.axhline(y=baseline, color='red', linestyle='--', label=f'Random baseline ({baseline:.4f})')
plt.legend()
plt.show()

# Find threshold meeting a business-stated recall floor
def best_threshold_for_recall_floor(y_true, y_proba, min_recall=0.9):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    valid = [(p, r, t) for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds) if r >= min_recall]
    if not valid:
        return None
    best = max(valid, key=lambda x: x[0])  # maximize precision among those meeting the recall floor
    return {'threshold': best[2], 'precision': best[0], 'recall': best[1]}

result = best_threshold_for_recall_floor(y_test, y_proba, min_recall=0.9)
print(f"To catch >=90% of fraud: threshold={result['threshold']:.3f}, "
      f"precision={result['precision']:.3f}, recall={result['recall']:.3f}")
```

## Visual Explanation

```
Confusion Matrix                    Precision-Recall Tradeoff
                                     as threshold changes:
              Predicted
           Legit    Fraud
Actual  ┌────────┬────────┐         High threshold (e.g. 0.9):
 Legit  │  TN    │  FP    │         few flagged -> high precision,
        ├────────┼────────┤                        low recall
 Fraud  │  FN    │  TP    │
        └────────┴────────┘         Low threshold (e.g. 0.1):
                                     many flagged -> low precision,
Precision = TP/(TP+FP)                              high recall
Recall    = TP/(TP+FN)

ROC-AUC vs PR-AUC under imbalance:

ROC space (FPR barely moves          PR space (Precision drops
  because TN pool is huge):            sharply - no huge TN pool
                                        to dilute the FP impact):
  TPR                                  Precision
  1.0 |        ____----                1.0 |\
      |    ___/                            |  \___
      |  _/            (looks great)       |      \____   (shows the
  0.0 |_/________________ FPR         0.0 |___________\_ Recall  real cost)
      0.0              1.0                 0.0            1.0
```

## Common Mistakes

- **Leading with accuracy** as the headline metric on an imbalanced dataset — an immediate red flag to
  any interviewer familiar with fraud/imbalanced classification problems.
- **Using ROC-AUC as the sole metric** without acknowledging its limitation under severe imbalance.
- **Defaulting to threshold 0.5** without justification — under 0.17% prevalence, 0.5 is essentially an
  arbitrary, usually-too-conservative choice with no inherent business meaning.
- **Reporting F1 without stating whether precision or recall matters more** for the specific business
  context — F1 assumes they're equally important, which is rarely true in fraud (missing fraud is
  usually costlier than a false alarm, but not infinitely so).
- **Not reporting the random-baseline comparison for PR-AUC** — a PR-AUC of 0.4 sounds mediocre in
  isolation but is actually excellent at 0.17% prevalence; omitting the baseline makes the number
  uninterpretable to someone unfamiliar with the dataset.

## Advanced Discussion

- **Cost-sensitive evaluation:** Beyond precision/recall, assigning actual dollar costs to FP (customer
  friction cost) and FN (average fraud loss amount) lets you compute **expected cost** at each threshold
  and choose the threshold that minimizes total expected cost, rather than optimizing an abstract
  metric. This is the single strongest "business-minded" answer available on this topic, and ties
  directly to Amex-style case questions from the placement report ("how do you avoid loss while
  launching a credit card").
- **Calibration vs. discrimination:** PR-AUC and ROC-AUC measure how well a model *ranks* positives
  above negatives (discrimination), but say nothing about whether the predicted probabilities are
  *calibrated* (i.e., among predictions of 0.7, do roughly 70% actually turn out to be fraud?).
  Calibration matters if the business wants to reason about expected fraud losses in dollar terms using
  the raw probability, not just the resulting binary flag — a genuinely senior distinction to raise.
- **Metric stability under temporal drift:** A model's PR-AUC measured on a random train/test split can
  look very different from PR-AUC measured on a strictly time-based split (train on earlier
  transactions, test on later ones) — the latter is a more honest estimate of real-world deployment
  performance, since fraud patterns evolve over time and a random split can leak "future" information
  into training in ways a time-based split wouldn't.

## Revision Notes

- Accuracy is misleading under severe imbalance — 99.8%+ accuracy is trivially achievable by predicting
  the majority class
- Precision = "of flagged, how many were real fraud"; Recall = "of real fraud, how many did we catch"
- Precision and recall trade off against each other as the classification threshold changes
- ROC-AUC can look artificially high under imbalance because FPR is diluted by a huge true-negative pool
- PR-AUC is the more honest primary metric for this problem; compare against the random baseline
  (= positive class prevalence) for it to be interpretable
- Threshold selection is a business decision (cost of FP vs FN), not a fixed default like 0.5
- In this project: PR-AUC is the primary metric and Optuna tuning objective; thresholds are chosen
  against business-stated recall floors, not left at 0.5

## References

- Davis, J. & Goadrich, M. (2006). "The Relationship Between Precision-Recall and ROC Curves." *ICML* —
  the canonical paper explaining exactly why PR curves are more informative under class imbalance
- Saito, T. & Rehmsmeier, M. (2015). "The Precision-Recall Plot Is More Informative than the ROC Plot
  When Evaluating Binary Classifiers on Imbalanced Datasets." *PLOS ONE*
- scikit-learn documentation: `sklearn.metrics` — precision_recall_curve, average_precision_score,
  roc_auc_score
- *Imbalanced Learning: Foundations, Algorithms, and Applications* (He & Ma) — for the broader
  imbalanced-classification context these metrics sit within
