# 56 — Evaluating models honestly

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/applied-ds/56-model-evaluation/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/applied-ds/model-evaluation?utm_source=github&utm_medium=labs)

*~45 minutes · 19 tests · Intermediate*

When building machine learning models, high accuracy is the ultimate fool's gold. If you are predicting credit card fraud and only 0.1% of transactions are fraudulent, a brain-dead model that marks every transaction as legitimate is 99.9% accurate. It is also completely useless.

To build models that actually solve business problems, you must move beyond accuracy. You need to understand the exact trade-offs between false positives and false negatives, optimize decision thresholds for specific business costs, and validate your models using techniques that prevent data leakage and represent the real-world distribution of your data.

In this module, you will build the core machinery of model evaluation from scratch. By implementing confusion matrices, precision, recall, F1, ROC AUC, threshold tuning, and stratified cross-validation, you will gain a deep, mathematical intuition for how these metrics behave and how to avoid the pitfalls that lead to over-optimistic evaluations.

## What you'll build

Six functions in `exercises.py`, in this order. Keep this table open as you go — each section below tells you which row it's building:

| Function | What it teaches | Shows up in real code as |
|---|---|---|
| `confusion_counts` | TP, FP, FN, TN extraction | `sklearn.metrics.confusion_matrix` |
| `precision_recall_f1` | Precision, Recall, and F1 calculations | `sklearn.metrics.precision_recall_fscore_support` |
| `roc_auc` | ROC Area Under Curve via rank-sum formula | `sklearn.metrics.roc_auc_score` |
| `best_threshold` | Threshold optimization for a target metric | Custom post-processing threshold tuning |
| `cross_validate_scores` | Stratified K-Fold cross-validation | `sklearn.model_selection.cross_val_score` |
| `compare_to_baseline` | Lift comparison against a majority-class dummy | `sklearn.dummy.DummyClassifier` comparison |

---

## 1. Confusion Matrix Counts

```python
>>> import numpy as np
>>> y_true = np.array([1, 0, 1, 1, 0])
>>> y_pred = np.array([1, 1, 0, 1, 0])
>>> # True Positives (actual 1, predicted 1)
>>> tp = np.sum((y_true == 1) & (y_pred == 1))
>>> tp
2
```

Every evaluation of a binary classifier starts with the raw counts of correct and incorrect predictions. These are grouped into four categories:
- **True Positives (TP)**: Actual positive, predicted positive.
- **False Positives (FP)**: Actual negative, predicted positive (Type I error).
- **False Negatives (FN)**: Actual positive, predicted negative (Type II error).
- **True Negatives (TN)**: Actual negative, predicted negative.

**The Pitfall:** The most common mistake is swapping the definitions of False Positives and False Negatives, or getting the array indices inverted. A False Positive means your model falsely cried wolf (predicted 1, but it was 0). A False Negative means your model missed a real signal (predicted 0, but it was 1).

**In `exercises.py`:** Implement `confusion_counts` to return these four counts as a dictionary.

**Test it:** `python3 -m unittest test_exercises.TestConfusionCounts -v`

---

## 2. Precision, Recall, and F1

```python
>>> tp, fp, fn = 2, 1, 1
>>> precision = tp / (tp + fp)
>>> recall = tp / (tp + fn)
>>> f1 = 2 * (precision * recall) / (precision + recall)
>>> print(f"P: {precision:.2f}, R: {recall:.2f}, F1: {f1:.2f}")
P: 0.67, R: 0.67, F1: 0.67
```

Once you have the raw counts, you can calculate ratios that describe your model's behavior:
- **Precision** answers: *"Of all the positive predictions, how many were actually positive?"* High precision is critical when false alarms are expensive (e.g., spam filters).
- **Recall** (Sensitivity) answers: *"Of all the actual positives, how many did we find?"* High recall is critical when missing a positive is catastrophic (e.g., medical diagnoses).
- **F1 Score** is the harmonic mean of precision and recall, balancing both into a single metric.

**The Pitfall:** Division by zero! If your model predicts zero positives, `tp + fp` is 0, causing a `ZeroDivisionError` for precision. If there are no actual positives in the dataset, `tp + fn` is 0, causing a `ZeroDivisionError` for recall. Your code must handle these edge cases gracefully by returning `0.0` instead of crashing.

**In `exercises.py`:** Implement `precision_recall_f1` to return a tuple of these three metrics.

**Test it:** `python3 -m unittest test_exercises.TestPrecisionRecallF1 -v`

---

## 3. ROC AUC via Rank Sum

```python
>>> import pandas as pd
>>> y_true = [0, 0, 1, 1]
>>> y_proba = [0.1, 0.4, 0.35, 0.8]
>>> ranks = pd.Series(y_proba).rank(method='average').values
>>> ranks
array([1., 3., 2., 4.])
```

The Area Under the Receiver Operating Characteristic Curve (ROC AUC) measures how well a model ranks predictions. It represents the probability that a randomly chosen positive sample will receive a higher predicted probability than a randomly chosen negative sample.

While you can compute this by plotting the ROC curve and calculating the area under the curve using trapezoids, there is an exact, elegant mathematical shortcut using the Wilcoxon-Mann-Whitney rank-sum formula:

$$\text{AUC} = \frac{R_p - \frac{n_p(n_p + 1)}{2}}{n_p \times n_n}$$

Where:
- $R_p$ is the sum of the ranks of the positive samples.
- $n_p$ is the number of positive samples.
- $n_n$ is the number of negative samples.
- Ranks are 1-based (the smallest probability gets rank 1, the largest gets rank $N$).

**The Pitfall:** Ties in predicted probabilities. If two samples have the exact same predicted probability, they must share the average of the ranks they would have occupied (fractional ranking). Using standard sorting without handling ties correctly will lead to unstable and incorrect AUC scores. Pandas' `.rank(method='average')` handles this automatically.

**In `exercises.py`:** Implement `roc_auc` using the rank-sum formula.

**Test it:** `python3 -m unittest test_exercises.TestRocAuc -v`

---

## 4. Optimizing the Decision Threshold

```python
>>> y_proba = np.array([0.1, 0.4, 0.6, 0.9])
>>> threshold = 0.5
>>> y_pred = (y_proba >= threshold).astype(int)
>>> y_pred
array([0, 0, 1, 1])
```

Classifiers output probabilities, not binary labels. By default, scikit-learn uses a threshold of `0.5` to convert these probabilities into classes. However, `0.5` is rarely the optimal threshold for real-world business goals.

If you want to maximize the F1 score, or if your business demands a minimum recall of 90%, you must search for the decision threshold that optimizes your target metric. You do this by evaluating your metric across a range of candidate thresholds.

**The Pitfall:** Searching over an arbitrary grid (like `np.linspace(0, 1, 100)`) is inefficient and can miss the exact transition points. The only thresholds that can actually change your model's predictions are the unique values present in your predicted probabilities (`y_proba`). You should only search over these unique values.

**In `exercises.py`:** Implement `best_threshold` to find the threshold that maximizes a chosen metric.

**Test it:** `python3 -m unittest test_exercises.TestBestThreshold -v`

---

## 5. Stratified Cross-Validation

```python
>>> from sklearn.model_selection import StratifiedKFold
>>> skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
>>> # StratifiedKFold ensures each fold has the same class ratio as the whole dataset
```

If you split your dataset into train and test sets randomly, you might get lucky (or unlucky). Your test set might contain an unusually high or low number of positive cases, leading to an inaccurate evaluation.

Cross-validation solves this by splitting the data into $K$ folds, training on $K-1$ folds, and testing on the remaining fold. For imbalanced datasets, we must use **Stratified K-Fold**, which guarantees that each fold has the exact same proportion of positive and negative classes as the entire dataset.

**The Pitfall:** Data leakage. If you perform any preprocessing (like scaling features or oversampling) on the entire dataset *before* splitting, information from the validation folds leaks into the training folds, inflating your cross-validated performance.

**In `exercises.py`:** Implement `cross_validate_scores` to perform stratified cross-validation manually and return the F1 scores.

**Test it:** `python3 -m unittest test_exercises.TestCrossValidateScores -v`

---

## 6. Comparing to a Baseline

```python
>>> # A dummy baseline that always predicts the majority class
>>> majority_class = 0
>>> y_pred_baseline = np.zeros(len(y_true))
```

A model with an F1 score of 0.85 sounds impressive. But what if 85% of your dataset belongs to the positive class? A dummy model that always predicts `1` would also get an F1 score of 0.92 without learning a single pattern!

To prove your model is actually useful, you must compare its performance against a baseline. The most common baseline for classification is a model that always predicts the majority class of the training data.

**The Pitfall:** Calculating the baseline's performance on the entire dataset instead of fold-by-fold. To make a fair comparison, the baseline must be evaluated using the exact same cross-validation splits as your model, determining the majority class using only the training fold for each split.

**In `exercises.py`:** Implement `compare_to_baseline` to calculate the lift of your model over a majority-class baseline.

**Test it:** `python3 -m unittest test_exercises.TestCompareToBaseline -v`

---

## The exercises

| Function | What it teaches |
|---|---|
| `confusion_counts` | Extracting TP, FP, FN, TN using boolean masks |
| `precision_recall_f1` | Safe metric calculation with zero-division handling |
| `roc_auc` | Direct ROC AUC calculation via Mann-Whitney U rank-sum |
| `best_threshold` | Finding the optimal probability threshold for a metric |
| `cross_validate_scores` | Implementing Stratified K-Fold cross-validation |
| `compare_to_baseline` | Calculating model lift over a majority-class baseline |

## Check your understanding

1. **Why is accuracy a poor metric for highly imbalanced datasets?**
   <details><summary>Answer</summary>Because a model can achieve extremely high accuracy simply by always predicting the majority class, completely failing to detect the rare class of interest.</details>

2. **What is the difference between Precision and Recall?**
   <details><summary>Answer</summary>Precision measures the quality of your positive predictions (how many predicted positives are actually positive). Recall measures the quantity of positive cases found (how many actual positives were successfully detected).</details>

3. **Why does the rank-sum formula for ROC AUC work?**
   <details><summary>Answer</summary>Because ROC AUC is mathematically equivalent to the probability that a randomly selected positive sample is ranked higher than a randomly selected negative sample. The Mann-Whitney U statistic calculates exactly this pairwise ranking probability.</details>

4. **Why should you search for the best threshold only among the unique values of `y_proba`?**
   <details><summary>Answer</summary>Because predictions only change when the threshold crosses an actual predicted probability. Any threshold value between two adjacent predicted probabilities will yield the exact same predictions and metric scores.</details>

5. **Why is Stratified K-Fold preferred over standard K-Fold for imbalanced datasets?**
   <details><summary>Answer</summary>Standard K-Fold splits data randomly, which can lead to folds with very few or even zero positive samples. Stratified K-Fold guarantees that every fold maintains the same class ratio as the overall dataset, ensuring stable and honest evaluations.</details>

## Interview questions this module answers

- *How do you handle division by zero when calculating Precision and Recall?*
- *Can you explain the mathematical relationship between ROC AUC and the Mann-Whitney U test?*
- *How would you optimize a model's decision threshold for a custom business metric?*
- *What is data leakage in cross-validation, and how do you prevent it?*
- *How do you establish a baseline to prove your classification model has actually learned something?*
