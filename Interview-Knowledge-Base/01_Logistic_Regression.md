# Logistic Regression

## Overview

Logistic Regression is a supervised learning algorithm used for classification, despite the word
"regression" in its name. Given input features, it predicts the **probability** that an observation
belongs to a class (e.g., fraud vs. not fraud).

**Beginner level:** Think of it as drawing a line (or in higher dimensions, a hyperplane) that
separates two classes as well as possible, then squashing the distance from that line into a
probability between 0 and 1.

**Intermediate level:** It's a generalized linear model where the linear combination of features is
passed through the sigmoid (logistic) function to produce a probability, and the model is trained by
maximizing the likelihood of the observed labels.

**Advanced level:** It's the simplest member of the exponential family of GLMs, forms the building
block for understanding neural network output layers (a single-layer NN with sigmoid activation IS
logistic regression), and its decision boundary is always linear in feature space — which is both its
biggest strength (interpretability, speed) and its biggest limitation (can't capture non-linear
interactions without manual feature engineering or transformation).

## Theory

**The model:**

```
z = w0 + w1*x1 + w2*x2 + ... + wn*xn
p = sigmoid(z) = 1 / (1 + e^(-z))
```

`p` is the predicted probability of the positive class (fraud = 1).

**Why sigmoid?** It maps any real number (−∞ to +∞) to a value between 0 and 1, giving a valid
probability. It's also differentiable everywhere, which is required for gradient-based optimization.

**Loss function — Binary Cross-Entropy (Log Loss):**

```
Loss = -[y * log(p) + (1-y) * log(1-p)]
```

This penalizes confident wrong predictions heavily. If the true label is fraud (y=1) and the model
predicts p=0.01 (very confident it's NOT fraud), `-log(0.01)` is a huge loss. This asymmetric, heavy
penalty for confident mistakes is why log loss (not squared error) is used for classification.

**Optimization:** Gradient Descent (or variants like L-BFGS, which `sklearn`'s default solver uses)
iteratively adjusts weights to minimize the loss. There's no closed-form solution like ordinary linear
regression has, because of the non-linear sigmoid.

**Assumptions:**
- Linear relationship between features and the **log-odds** of the outcome (not the outcome itself)
- Little to no multicollinearity between features
- Observations are independent
- Large sample size preferred (asymptotic properties of MLE)

**Strengths:**
- Fast to train, fast to predict — matters for real-time fraud scoring
- Highly interpretable — each weight tells you the change in log-odds per unit change in a feature
- Outputs well-calibrated probabilities (with regularization, can be adjusted)
- Works well as a baseline and rarely overfits badly with regularization

**Weaknesses:**
- Assumes linear decision boundary — can't capture feature interactions unless you manually add them
- Sensitive to outliers (each point contributes to the log-likelihood)
- Sensitive to feature scale — needs standardization
- Struggles with severe class imbalance without adjustment (defaults to predicting majority class)

**Common misconceptions:**
- "It's a regression model for continuous output" — no, output is a probability for classification.
- "The coefficients are the change in probability" — false; they are the change in **log-odds**, which
  is not linearly related to probability. A common interview trap.
- "Logistic Regression can't handle non-linear problems at all" — it can, if you manually engineer
  polynomial/interaction features first; it just can't discover them automatically like a tree model.

## Interview Questions

**Beginner**
1. What is Logistic Regression used for?
2. Why is it called "regression" if it's used for classification?
3. What does the sigmoid function do?
4. What is the output range of Logistic Regression?
5. What loss function does it use, and why not mean squared error?

**Intermediate**
6. Derive the sigmoid function from odds and log-odds.
7. What do the coefficients represent?
8. Why do we need feature scaling for Logistic Regression but not for tree-based models?
9. How does L1 (Lasso) vs L2 (Ridge) regularization affect Logistic Regression?
10. How would you handle multicollinearity in your features before fitting the model?
11. What's the difference between Logistic Regression's decision boundary and a Decision Tree's?

**Advanced / Mathematical**
12. Derive the gradient of the log-loss with respect to the weights.
13. Explain Maximum Likelihood Estimation in the context of Logistic Regression.
14. Why is log loss convex for Logistic Regression, and why does that matter for optimization?
15. How does Logistic Regression relate to a single-neuron neural network?

**Practical / Scenario-based (Fraud-specific)**
16. Your fraud dataset is 99.8% legitimate transactions. How does that affect a default Logistic
    Regression model, and how would you fix it?
17. If a coefficient for "Amount_log" is 0.8, how do you explain what that means to a non-technical
    stakeholder?
18. Your model gets 99.8% accuracy but catches 0% of fraud. What happened, and how do you diagnose it?
19. How would you use Logistic Regression's probability output to build a decision at a specific
    business-chosen threshold rather than the default 0.5?

**Coding**
20. Write code to train a Logistic Regression model with class weighting to handle imbalance.
21. Write code to extract and interpret the top 5 most influential features by coefficient magnitude.

**Follow-ups an interviewer might throw at you**
- "You said Logistic Regression can't model non-linear boundaries — how would you make it able to?"
  (Answer: manual polynomial/interaction terms, or use it after a non-linear feature extractor.)
- "Why would you ever use Logistic Regression over XGBoost if XGBoost usually performs better?"
  (Answer: interpretability requirements, regulatory/compliance needs in finance, latency constraints,
  smaller data regimes where XGBoost might overfit.)

## Answers

**Q3 — What does the sigmoid function do?**
The sigmoid function `1/(1+e^-z)` takes any real-valued number (the raw linear combination of weighted
features, called the "logit" or "z") and compresses it into the range (0, 1). As z → +∞, sigmoid(z) →
1; as z → −∞, sigmoid(z) → 0; at z=0, sigmoid(z)=0.5. This lets us interpret the output as a
probability. It's chosen (rather than, say, a simple clipped linear function) because it's smooth and
differentiable everywhere, which gradient-based optimizers require, and because it naturally arises
from the log-odds formulation of binary outcomes.

**Q7 — What do the coefficients represent?**
Each coefficient `w_i` represents the change in the **log-odds** of the positive class for a one-unit
increase in feature `x_i`, holding all other features constant. Concretely: `odds = p/(1-p)`, and
`log(odds) = w0 + w1*x1 + ... `. So `e^(w_i)` gives you the **odds ratio** — how much the odds multiply
by for a one-unit increase in that feature. This is NOT the same as "how much the probability changes,"
which is a very common interview trap — probability change depends on where you currently are on the
sigmoid curve (near p=0.5 a unit change in log-odds moves probability a lot; near p=0.99 it barely
moves it at all).

**Q16 — 99.8% accuracy, 0% fraud caught, how do you diagnose and fix it?**
This is the class imbalance failure mode. Diagnosis: check the confusion matrix, not just accuracy —
you'll see the model predicts the majority class (not fraud) for almost everything, meaning recall on
the fraud class is near zero. Fix: (1) use `class_weight='balanced'` in sklearn so misclassifying fraud
is penalized proportionally to the imbalance ratio, (2) or resample the training data with SMOTE/
undersampling, (3) and critically — stop looking at accuracy; evaluate with precision, recall, F1, and
PR-AUC instead, since accuracy is meaningless at 0.17% fraud prevalence.

## Project Connection

In the fraud detection project, Logistic Regression serves as the **baseline model** — the floor that
XGBoost and Random Forest need to beat to justify their added complexity. Concretely:
- Trained on `Amount_log`, `Hour`, and the scaled `V1-V28` features, with `class_weight='balanced'`
  to handle the 0.17% fraud rate without needing SMOTE (though we also compare it with SMOTE).
- Its coefficients give an explainable answer to "how does the model decide" — useful for the exact
  kind of business-facing question EY-Parthenon, Amex, and Accenture ask ("explain your model to a
  non-technical stakeholder").
- Its speed makes it a candidate for a real-time pre-filter, even if XGBoost is the final scorer,
  since it's cheap enough to run on every transaction before a heavier model is invoked.

## Code Examples

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Scaling is required - Logistic Regression is sensitive to feature scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# class_weight='balanced' handles imbalance without needing SMOTE
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train_scaled, y_train)

y_proba = model.predict_proba(X_test_scaled)[:, 1]

# Interpreting coefficients
import pandas as pd
coef_df = pd.DataFrame({
    'feature': X_train.columns,
    'coefficient': model.coef_[0],
    'odds_ratio': np.exp(model.coef_[0])
}).sort_values('coefficient', key=abs, ascending=False)
print(coef_df.head(10))
```

## Visual Explanation

```
   Feature space                     Sigmoid squashing
                                      1.0 |            _____
   x2                                     |          /
   |    o o o                        p    |         /
   |   o o  o    x x                      |        /
   |  o  o  \    x  x                 0.5 |-------/--------
   |    o    \   x   x                    |      /
   |          \    x x                    |     /
   |___________\_______ x1            0.0 |____/____________
              linear                       -inf  0    z    +inf
              decision
              boundary
```
The line on the left is `z = w0 + w1*x1 + w2*x2 = 0`. Everything is a linear combination; the sigmoid
on the right just re-expresses distance-from-that-line as a probability.

## Common Mistakes

- **Forgetting to scale features.** Since coefficients are directly tied to feature magnitude,
  unscaled features (e.g., `Amount` ranging 0–25,000 vs `V14` ranging roughly −20 to 20) will distort
  which features appear "important."
- **Interpreting coefficients as probability changes**, not log-odds changes — leads to wrong
  explanations in interviews and in stakeholder presentations.
- **Using default threshold 0.5 blindly** under severe imbalance — with 0.17% fraud, the model may
  never predict a probability above 0.5 for anything, even correctly-identified fraud, if not
  weighted/resampled properly.
- **Reporting accuracy as the headline metric** on imbalanced data — always a red flag to an
  interviewer if you lead with accuracy on a fraud problem.
- **Not checking for multicollinearity** among features — inflates coefficient variance and makes
  individual coefficients hard to trust, even if overall prediction quality is fine.

## Advanced Discussion

- **Regularization deep dive:** L1 (Lasso) drives some coefficients exactly to zero, effectively doing
  feature selection — useful if you suspect several of the V1-V28 components are uninformative. L2
  (Ridge) shrinks all coefficients smoothly, useful when you believe most features contribute a little.
  Elastic Net blends both. A senior interviewer may ask you to justify your regularization choice given
  the specific structure of PCA-derived features (which are already decorrelated by construction — so
  multicollinearity concerns are lower than with raw features).
- **Calibration:** Logistic Regression, when properly regularized, tends to output well-calibrated
  probabilities (predicted probability roughly matches observed frequency), unlike tree ensembles which
  often need explicit calibration (Platt scaling / isotonic regression) after training. This is a
  legitimate reason to keep a calibrated LogReg around even if XGBoost wins on raw metrics — calibrated
  probabilities matter if the business wants to reason about expected fraud losses in dollar terms.
- **Relationship to Naive Bayes:** Both are linear classifiers in log-odds space, but Naive Bayes is
  generative (models P(X|Y)) while Logistic Regression is discriminative (models P(Y|X) directly) —
  a classic senior-level conceptual question.

## Revision Notes

- Logistic Regression predicts probability via `sigmoid(linear combination of features)`
- Trained via Maximum Likelihood / minimizing log loss, not MSE
- Coefficients = change in log-odds, NOT probability, per unit feature increase
- Requires feature scaling; assumes linear boundary in log-odds space
- Fast, interpretable, good baseline; struggles with non-linear patterns and severe imbalance without
  adjustment (`class_weight='balanced'` or resampling)
- In this project: the interpretable baseline and a candidate real-time pre-filter

## References

- *An Introduction to Statistical Learning* (James, Witten, Hastie, Tibshirani) — Ch. 4, Classification
- *The Elements of Statistical Learning* (Hastie, Tibshirani, Friedman) — Ch. 4, for the mathematical
  depth on GLMs
- scikit-learn documentation: `sklearn.linear_model.LogisticRegression`
- Andrew Ng's Machine Learning course (Coursera) — Week 3, Classification and Logistic Regression
