"""56 - Evaluating models honestly

Reference implementations for the exercises.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    counts = confusion_counts(y_true, y_pred)
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
        
    return (float(precision), float(recall), float(f1))


def roc_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n_p = int(np.sum(y_true == 1))
    n_n = int(np.sum(y_true == 0))
    if n_p == 0 or n_n == 0:
        return 0.0
    
    ranks = pd.Series(y_proba).rank(method='average').values
    R_p = np.sum(ranks[y_true == 1])
    auc = (R_p - (n_p * (n_p + 1)) / 2) / (n_p * n_n)
    return float(auc)


def best_threshold(y_true: np.ndarray, y_proba: np.ndarray, metric: str = "f1") -> float:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    if len(y_proba) == 0:
        return 0.5
    
    unique_thresholds = np.unique(y_proba)
    best_val = -1.0
    best_thresh = 0.5
    metric_idx = {"precision": 0, "recall": 1, "f1": 2}[metric]
    
    for thresh in unique_thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        scores = precision_recall_f1(y_true, y_pred)
        score = scores[metric_idx]
        
        if score > best_val:
            best_val = score
            best_thresh = thresh
        elif score == best_val and thresh < best_thresh:
            best_thresh = thresh
            
    return float(best_thresh)


def cross_validate_scores(X: pd.DataFrame, y: np.ndarray, model, n_splits: int = 5) -> list[float]:
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, test_idx in skf.split(X_arr, y_arr):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]
        
        fold_model = clone(model)
        fold_model.fit(X_train, y_train)
        y_pred = fold_model.predict(X_test)
        _, _, f1 = precision_recall_f1(y_test, y_pred)
        scores.append(f1)
        
    return scores


def compare_to_baseline(X: pd.DataFrame, y: np.ndarray, model, n_splits: int = 5) -> dict[str, float]:
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    model_scores = []
    baseline_scores = []
    
    for train_idx, test_idx in skf.split(X_arr, y_arr):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]
        
        # Model
        fold_model = clone(model)
        fold_model.fit(X_train, y_train)
        y_pred = fold_model.predict(X_test)
        _, _, f1_m = precision_recall_f1(y_test, y_pred)
        model_scores.append(f1_m)
        
        # Baseline
        classes, counts = np.unique(y_train, return_counts=True)
        if len(classes) == 1:
            majority_class = classes[0]
        elif counts[0] == counts[1]:
            majority_class = 0
        else:
            majority_class = classes[np.argmax(counts)]
            
        y_pred_base = np.full_like(y_test, fill_value=majority_class)
        _, _, f1_b = precision_recall_f1(y_test, y_pred_base)
        baseline_scores.append(f1_b)
        
    mean_model = float(np.mean(model_scores))
    mean_base = float(np.mean(baseline_scores))
    
    return {
        "model_f1": mean_model,
        "baseline_f1": mean_base,
        "lift": mean_model - mean_base
    }
