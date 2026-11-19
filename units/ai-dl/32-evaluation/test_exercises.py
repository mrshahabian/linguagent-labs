"""Tests for 32 - evaluation metrics."""

import importlib
import math
import os
import unittest

import torch

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestPrecisionRecallF1(unittest.TestCase):
    def test_matches_hand_computed_values(self):
        # TP=3 (idx 0,3,6), FP=1 (idx 1), FN=1 (idx 5)
        predictions = [1, 1, 0, 1, 0, 0, 1, 0]
        labels =      [1, 0, 0, 1, 0, 1, 1, 0]
        precision, recall, f1 = m.precision_recall_f1(predictions, labels)
        self.assertAlmostEqual(precision, 0.75)
        self.assertAlmostEqual(recall, 0.75)
        self.assertAlmostEqual(f1, 0.75)

    def test_perfect_predictions(self):
        predictions = [1, 0, 1, 0]
        labels = [1, 0, 1, 0]
        precision, recall, f1 = m.precision_recall_f1(predictions, labels)
        self.assertEqual((precision, recall, f1), (1.0, 1.0, 1.0))

    def test_predict_everything_positive_hurts_precision_not_recall(self):
        predictions = [1, 1, 1, 1]
        labels = [1, 0, 1, 0]
        precision, recall, f1 = m.precision_recall_f1(predictions, labels)
        self.assertAlmostEqual(precision, 0.5)
        self.assertAlmostEqual(recall, 1.0)

    def test_no_positive_predictions_gives_zero_precision(self):
        predictions = [0, 0, 0]
        labels = [1, 0, 1]
        precision, recall, f1 = m.precision_recall_f1(predictions, labels)
        self.assertEqual(precision, 0.0)
        self.assertEqual(recall, 0.0)
        self.assertEqual(f1, 0.0)


class TestAveragePrecision(unittest.TestCase):
    def test_matches_sklearn_style_computation(self):
        confidences = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
        labels =      [1,   0,   1,   1,   0,   1,   0,   0]
        ap = m.average_precision(confidences, labels)
        self.assertAlmostEqual(ap, 0.7708333333333333, places=6)

    def test_perfect_ranking_gives_ap_of_one(self):
        confidences = [0.9, 0.8, 0.7, 0.6]
        labels = [1, 1, 0, 0]
        self.assertAlmostEqual(m.average_precision(confidences, labels), 1.0)

    def test_worst_ranking_gives_low_ap(self):
        confidences = [0.9, 0.8, 0.7, 0.6]
        labels = [0, 0, 1, 1]  # positives ranked last
        ap = m.average_precision(confidences, labels)
        self.assertLess(ap, 0.7)

    def test_no_positives_returns_zero(self):
        self.assertEqual(m.average_precision([0.9, 0.1], [0, 0]), 0.0)


class TestBleuN(unittest.TestCase):
    def test_identical_sentences_score_one(self):
        tokens = ["the", "cat", "sat", "on", "the", "mat"]
        self.assertAlmostEqual(m.bleu_n(tokens, tokens, n=4), 1.0)

    def test_completely_disjoint_scores_zero(self):
        candidate = ["dogs", "run", "fast"]
        reference = ["the", "cat", "sat", "on", "the", "mat"]
        self.assertEqual(m.bleu_n(candidate, reference, n=2), 0.0)

    def test_shorter_candidate_gets_brevity_penalty(self):
        candidate = ["the", "cat", "sat"]
        reference = ["the", "cat", "sat", "on", "the", "mat"]
        score = m.bleu_n(candidate, reference, n=2)
        self.assertAlmostEqual(score, math.exp(-1), places=5)

    def test_empty_candidate_is_zero(self):
        self.assertEqual(m.bleu_n([], ["a", "b"], n=2), 0.0)


class TestRougeL(unittest.TestCase):
    def test_identical_sentences_score_one(self):
        tokens = ["the", "cat", "sat", "on", "the", "mat"]
        self.assertAlmostEqual(m.rouge_l(tokens, tokens), 1.0)

    def test_no_common_subsequence_scores_zero(self):
        candidate = ["x", "y", "z"]
        reference = ["a", "b", "c"]
        self.assertEqual(m.rouge_l(candidate, reference), 0.0)

    def test_gives_partial_credit_for_reordered_content(self):
        # ROUGE-L doesn't require consecutive matches, unlike BLEU n-grams
        candidate = ["the", "mat", "sat", "cat"]
        reference = ["the", "cat", "sat", "on", "the", "mat"]
        score = m.rouge_l(candidate, reference)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_empty_inputs_score_zero(self):
        self.assertEqual(m.rouge_l([], ["a"]), 0.0)
        self.assertEqual(m.rouge_l(["a"], []), 0.0)


class TestRecallAtK(unittest.TestCase):
    def setUp(self):
        self.gallery = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])

    def test_hit_at_k1(self):
        query = torch.tensor([-1.0, 0.0])  # matches gallery[2] exactly
        self.assertTrue(m.recall_at_k(query, self.gallery, correct_index=2, k=1))

    def test_miss_at_k1(self):
        query = torch.tensor([-1.0, 0.0])  # opposite of gallery[0]
        self.assertFalse(m.recall_at_k(query, self.gallery, correct_index=0, k=1))

    def test_larger_k_can_turn_a_miss_into_a_hit(self):
        # gallery[1] and gallery[3] are equally similar (0) to the query,
        # and rank ahead of gallery[0] (similarity -1) - so index 1 shows
        # up within the top 3 even though it's not the single best match.
        query = torch.tensor([-1.0, 0.0])
        self.assertTrue(m.recall_at_k(query, self.gallery, correct_index=1, k=3))

    def test_k_equal_to_gallery_size_always_hits(self):
        query = torch.tensor([5.0, 5.0])
        for index in range(len(self.gallery)):
            self.assertTrue(m.recall_at_k(query, self.gallery, correct_index=index, k=len(self.gallery)))


if __name__ == "__main__":
    unittest.main()
