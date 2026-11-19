"""Tests for 51 - Evaluating LLM systems. Read them; they are the
specification."""

import importlib
import os
import unittest
import numpy as np

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestNormalizeAnswer(unittest.TestCase):
    def test_lowercase(self):
        self.assertEqual(m.normalize_answer("HELLO WORLD"), "hello world")

    def test_punctuation(self):
        self.assertEqual(m.normalize_answer("hello, world!!!"), "hello world")

    def test_articles(self):
        self.assertEqual(m.normalize_answer("the apple"), "apple")
        self.assertEqual(m.normalize_answer("an orange"), "orange")
        self.assertEqual(m.normalize_answer("a banana"), "banana")

    def test_article_substrings(self):
        # Words containing "the", "an", "a" as substrings should not be stripped
        self.assertEqual(m.normalize_answer("another theater canopy"), "another theater canopy")

    def test_whitespace(self):
        self.assertEqual(m.normalize_answer("   many    spaces   here  "), "many spaces here")


class TestExactMatch(unittest.TestCase):
    def test_identical(self):
        self.assertTrue(m.exact_match("test", "test"))

    def test_normalized_match(self):
        self.assertTrue(m.exact_match("The Eiffel Tower.", "eiffel tower"))

    def test_mismatch(self):
        self.assertFalse(m.exact_match("Eiffel Tower", "Louvre"))


class TestTokenLevelF1(unittest.TestCase):
    def test_perfect_match(self):
        self.assertAlmostEqual(m.token_level_f1("the quick brown fox", "quick brown fox"), 1.0)

    def test_partial_match(self):
        # Pred: "red ball", GT: "blue ball"
        # Common: "ball" (1)
        # Precision: 1/2, Recall: 1/2 -> F1: 0.5
        self.assertAlmostEqual(m.token_level_f1("the red ball", "the blue ball"), 0.5)

    def test_duplicate_tokens_pitfall(self):
        # Pred: "blue blue blue", GT: "blue"
        # Common: "blue" (1)
        # Precision: 1/3, Recall: 1/1 -> F1: 2 * (1/3) / (4/3) = 0.5
        self.assertAlmostEqual(m.token_level_f1("blue blue blue", "blue"), 0.5)

    def test_empty_cases(self):
        self.assertAlmostEqual(m.token_level_f1("", ""), 1.0)
        self.assertAlmostEqual(m.token_level_f1("the", "a"), 1.0)  # both normalize to empty
        self.assertAlmostEqual(m.token_level_f1("hello", ""), 0.0)


class TestRetrievalMetrics(unittest.TestCase):
    def test_recall_at_k_typical(self):
        retrieved = [1, 2, 3, 4, 5]
        gt = [2, 5, 9]
        # At K=3, retrieved[:3] is [1, 2, 3]. Intersection with gt is {2}. Recall = 1/3
        self.assertAlmostEqual(m.recall_at_k(retrieved, gt, k=3), 1/3)
        # At K=5, retrieved[:5] is [1, 2, 3, 4, 5]. Intersection is {2, 5}. Recall = 2/3
        self.assertAlmostEqual(m.recall_at_k(retrieved, gt, k=5), 2/3)

    def test_recall_at_k_large_k(self):
        retrieved = [1, 2]
        gt = [2, 5]
        self.assertAlmostEqual(m.recall_at_k(retrieved, gt, k=10), 0.5)

    def test_recall_at_k_empty_gt(self):
        self.assertAlmostEqual(m.recall_at_k([1, 2], [], k=5), 0.0)

    def test_mrr_typical(self):
        retrieved_batches = [
            [10, 20, 30],  # first match at rank 2 (20) -> RR = 0.5
            [40, 50, 60],  # first match at rank 1 (40) -> RR = 1.0
            [70, 80, 90]   # no match -> RR = 0.0
        ]
        gt_batches = [
            [20, 99],
            [40],
            [100]
        ]
        # Mean RR = (0.5 + 1.0 + 0.0) / 3 = 0.5
        self.assertAlmostEqual(m.mean_reciprocal_rank(retrieved_batches, gt_batches), 0.5)

    def test_mrr_empty(self):
        self.assertAlmostEqual(m.mean_reciprocal_rank([], []), 0.0)


class TestWinRate(unittest.TestCase):
    def test_all_wins(self):
        self.assertAlmostEqual(m.win_rate(["A", "A", "A"]), 1.0)

    def test_all_ties(self):
        self.assertAlmostEqual(m.win_rate(["tie", "tie"]), 0.5)

    def test_typical_distribution(self):
        # Wins: 2, Ties: 1, Losses: 1. Total: 4. Win rate: (2 + 0.5) / 4 = 0.625
        self.assertAlmostEqual(m.win_rate(["A", "tie", "B", "A"]), 0.625)

    def test_empty_judgements(self):
        self.assertAlmostEqual(m.win_rate([]), 0.0)


class TestBootstrapCI(unittest.TestCase):
    def test_bootstrap_ci_determinism(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        metric_fn = lambda x: sum(x) / len(x)
        
        # Run twice with same seed, should yield identical results
        ci1 = m.bootstrap_ci(data, metric_fn, num_resamples=100, confidence_level=0.90, seed=42)
        ci2 = m.bootstrap_ci(data, metric_fn, num_resamples=100, confidence_level=0.90, seed=42)
        self.assertEqual(ci1, ci2)
        
        # Check that bounds are reasonable
        self.assertTrue(1.0 <= ci1[0] <= ci1[1] <= 5.0)

    def test_bootstrap_ci_empty_data(self):
        with self.assertRaises(ValueError):
            m.bootstrap_ci([], lambda x: 0.0)

    def test_bootstrap_ci_with_win_rate(self):
        data = ["A"] * 10 + ["B"] * 10
        metric_fn = m.win_rate
        ci = m.bootstrap_ci(data, metric_fn, num_resamples=200, confidence_level=0.95, seed=123)
        # Lower bound should be < 0.5, upper bound should be > 0.5
        self.assertTrue(ci[0] < 0.5)
        self.assertTrue(ci[1] > 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
