"""Tests for 56 - Evaluating models honestly."""

import importlib
import os
import unittest
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestConfusionCounts(unittest.TestCase):
    def test_perfect_predictions(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])
        res = m.confusion_counts(y_true, y_pred)
        self.assertEqual(res, {"tp": 2, "fp": 0, "fn": 0, "tn": 2})

    def test_all_wrong(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1])
        res = m.confusion_counts(y_true, y_pred)
        self.assertEqual(res, {"tp": 0, "fp": 2, "fn": 2, "tn": 0})

    def test_imbalanced_counts(self):
        y_true = np.array([1, 1, 1, 0, 0, 0, 0])
        y_pred = np.array([1, 0, 1, 1, 0, 0, 0])
        res = m.confusion_counts(y_true, y_pred)
        self.assertEqual(res, {"tp": 2, "fp": 1, "fn": 1, "tn": 3})

    def test_empty_arrays(self):
        res = m.confusion_counts(np.array([]), np.array([]))
        self.assertEqual(res, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})


class TestPrecisionRecallF1(unittest.TestCase):
    def test_standard_case(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 1, 0])
        p, r, f1 = m.precision_recall_f1(y_true, y_pred)
        self.assertAlmostEqual(p, 0.5)
        self.assertAlmostEqual(r, 0.5)
        self.assertAlmostEqual(f1, 0.5)

    def test_division_by_zero_precision(self):
        y_true = np.array([1, 0, 1])
        y_pred = np.array([0, 0, 0])
        p, r, f1 = m.precision_recall_f1(y_true, y_pred)
        self.assertEqual(p, 0.0)
        self.assertEqual(r, 0.0)
        self.assertEqual(f1, 0.0)

    def test_division_by_zero_recall(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 0, 1])
        p, r, f1 = m.precision_recall_f1(y_true, y_pred)
        self.assertEqual(p, 0.0)
        self.assertEqual(r, 0.0)
        self.assertEqual(f1, 0.0)

    def test_perfect_scores(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        p, r, f1 = m.precision_recall_f1(y_true, y_pred)
        self.assertEqual(p, 1.0)
        self.assertEqual(r, 1.0)
        self.assertEqual(f1, 1.0)


class TestRocAuc(unittest.TestCase):
    def test_perfect_auc(self):
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(m.roc_auc(y_true, y_proba), 1.0)

    def test_worst_auc(self):
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.9, 0.8, 0.2, 0.1])
        self.assertAlmostEqual(m.roc_auc(y_true, y_proba), 0.0)

    def test_with_ties(self):
        y_true = np.array([0, 1])
        y_proba = np.array([0.5, 0.5])
        self.assertAlmostEqual(m.roc_auc(y_true, y_proba), 0.5)

    def test_single_class_returns_zero(self):
        y_true = np.array([1, 1, 1])
        y_proba = np.array([0.5, 0.6, 0.7])
        self.assertEqual(m.roc_auc(y_true, y_proba), 0.0)


class TestBestThreshold(unittest.TestCase):
    def test_standard_threshold_search(self):
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.4, 0.6, 0.9])
        # Best threshold for F1 should be 0.6 (gives predictions [0, 0, 1, 1] -> F1=1.0)
        thresh = m.best_threshold(y_true, y_proba, metric="f1")
        self.assertAlmostEqual(thresh, 0.6)

    def test_tie_breaker_returns_lowest(self):
        y_true = np.array([0, 1])
        y_proba = np.array([0.1, 0.9])
        # Both thresholds 0.1 and 0.9 give recall=1.0. Lowest is 0.1.
        thresh = m.best_threshold(y_true, y_proba, metric="recall")
        self.assertAlmostEqual(thresh, 0.1)

    def test_empty_proba(self):
        thresh = m.best_threshold(np.array([]), np.array([]))
        self.assertEqual(thresh, 0.5)


class TestCrossValidateScores(unittest.TestCase):
    def test_cross_validate_runs(self):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(20, 2))
        y = np.array([0]*10 + [1]*10)
        model = DecisionTreeClassifier(random_state=42)
        scores = m.cross_validate_scores(X, y, model, n_splits=2)
        self.assertEqual(len(scores), 2)
        for s in scores:
            self.assertTrue(0.0 <= s <= 1.0)

    def test_stratification_preserves_ratio(self):
        # 30 samples: 10 positive, 20 negative (1:2 ratio)
        X = pd.DataFrame(np.random.randn(30, 2))
        y = np.array([0]*20 + [1]*10)
        model = DecisionTreeClassifier(random_state=42)
        # We can verify that cross_validate_scores runs successfully with StratifiedKFold
        scores = m.cross_validate_scores(X, y, model, n_splits=3)
        self.assertEqual(len(scores), 3)


class TestCompareToBaseline(unittest.TestCase):
    def test_lift_calculation(self):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(40, 2))
        # Make class 0 highly dominant (35 vs 5)
        y = np.array([0]*35 + [1]*5)
        model = DecisionTreeClassifier(random_state=42)
        res = m.compare_to_baseline(X, y, model, n_splits=3)
        self.assertIn("model_f1", res)
        self.assertIn("baseline_f1", res)
        self.assertIn("lift", res)
        self.assertAlmostEqual(res["lift"], res["model_f1"] - res["baseline_f1"])

    def test_baseline_always_predicts_majority(self):
        # If majority is 0, baseline F1 should be 0.0 because TP will be 0.
        X = pd.DataFrame(np.random.randn(20, 2))
        y = np.array([0]*16 + [1]*4)
        model = DecisionTreeClassifier(random_state=42)
        res = m.compare_to_baseline(X, y, model, n_splits=2)
        self.assertEqual(res["baseline_f1"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
