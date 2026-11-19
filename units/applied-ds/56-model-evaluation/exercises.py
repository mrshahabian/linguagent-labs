"""56 - Evaluating models honestly

Implement each function so the tests in test_exercises.py pass. Read
README.md first; every exercise below is covered there.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """Calculate True Positives, False Positives, False Negatives, and True Negatives.

    Args:
        y_true: Array of actual binary labels (0 or 1).
        y_pred: Array of predicted binary labels (0 or 1).

    Returns:
        A dictionary with keys "tp", "fp", "fn", "tn" mapped to their respective counts.
        If arrays are empty, return all counts as 0.
    """
    raise NotImplementedError("your code here")


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Calculate precision, recall, and F1 score.

    If a denominator is zero, the corresponding metric should be 0.0.
    If both precision and recall are 0.0, F1 should be 0.0.

    Args:
        y_true: Array of actual binary labels (0 or 1).
        y_pred: Array of predicted binary labels (0 or 1).

    Returns:
        A tuple of (precision, recall, f1) as floats.
    """
    raise NotImplementedError("your code here")


def roc_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Calculate the Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    using the Wilcoxon-Mann-Whitney rank-sum formula.

    Formula:
        AUC = (R_p - (n_p * (n_p + 1)) / 2) / (n_p * n_n)
        where R_p is the sum of ranks of the positive class samples,
        n_p is the number of positive samples, and n_n is the number of negative samples.

    Ranks must be 1-based. Use fractional ranking (average method) for ties.
    If n_p or n_n is 0, return 0.0.

    Args:
        y_true: Array of actual binary labels (0 or 1).
        y_proba: Array of predicted probabilities for the positive class.

    Returns:
        The ROC AUC score as a float.
    """
    raise NotImplementedError("your code here")


def best_threshold(y_true: np.ndarray, y_proba: np.ndarray, metric: str = "f1") -> float:
    """Find the threshold in y_proba that maximizes the specified metric ('precision', 'recall', or 'f1').

    For each unique probability in y_proba, treat it as a threshold:
        y_pred = (y_proba >= threshold).astype(int)
    Calculate the metric using precision_recall_f1.

    If there is a tie for the maximum metric value, return the lowest threshold.
    If y_proba is empty, return 0.5.

    Args:
        y_true: Array of actual binary labels (0 or 1).
        y_proba: Array of predicted probabilities.
        metric: The metric to maximize ('precision', 'recall', or 'f1').

    Returns:
        The optimal threshold as a float.
    """
    raise NotImplementedError("your code here")


def cross_validate_scores(X: pd.DataFrame, y: np.ndarray, model, n_splits: int = 5) -> list[float]:
    """Perform Stratified K-Fold cross-validation and return the F1 score for each fold.

    Use sklearn.model_selection.StratifiedKFold with shuffle=True and random_state=42.
    For each fold, fit the model on the training fold, predict on the test fold,
    and compute the F1 score using precision_recall_f1.

    Args:
        X: Feature DataFrame or array.
        y: Target binary labels array.
        model: A scikit-learn compatible classifier (has fit and predict methods).
        n_splits: Number of folds.

    Returns:
        A list of F1 scores (floats), one for each fold.
    """
    raise NotImplementedError("your code here")


def compare_to_baseline(X: pd.DataFrame, y: np.ndarray, model, n_splits: int = 5) -> dict[str, float]:
    """Compare the model's cross-validated F1 score against a majority-class baseline.

    The baseline model always predicts the majority class of the training fold's target (y_train).
    In case of a tie in class counts in y_train, default to predicting class 0.

    Calculate the mean F1 score across all folds for both the model and the baseline.

    Args:
        X: Feature DataFrame or array.
        y: Target binary labels array.
        model: A scikit-learn compatible classifier.
        n_splits: Number of folds.

    Returns:
        A dictionary with keys:
            "model_f1": Mean F1 score of the model.
            "baseline_f1": Mean F1 score of the baseline.
            "lift": The difference (model_f1 - baseline_f1).
    """
    raise NotImplementedError("your code here")
