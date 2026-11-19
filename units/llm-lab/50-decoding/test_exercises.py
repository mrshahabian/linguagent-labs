"""Tests for 50 - Decoding: greedy, temperature, top-k and top-p. Read them; they are the specification."""

import importlib
import os
import unittest
import numpy as np

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))

class TestSoftmaxWithTemperature(unittest.TestCase):
    def test_standard_softmax(self):
        logits = np.array([1.0, 2.0, 3.0])
        probs = m.softmax_with_temperature(logits, temperature=1.0)
        expected = np.exp(logits) / np.sum(np.exp(logits))
        np.testing.assert_allclose(probs, expected, rtol=1e-5)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0)

    def test_high_temperature(self):
        logits = np.array([1.0, 2.0, 3.0])
        probs = m.softmax_with_temperature(logits, temperature=100.0)
        # At very high temperature, should be close to uniform
        expected = np.array([1/3, 1/3, 1/3])
        np.testing.assert_allclose(probs, expected, atol=1e-2)

    def test_low_temperature(self):
        logits = np.array([1.0, 2.0, 3.0])
        probs = m.softmax_with_temperature(logits, temperature=0.01)
        # At very low temperature, should be close to one-hot for the max element
        expected = np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(probs, expected, atol=1e-5)

    def test_numerical_stability(self):
        # Large logits that would overflow standard exp()
        logits = np.array([1000.0, 1001.0, 1002.0])
        probs = m.softmax_with_temperature(logits, temperature=1.0)
        expected = np.array([1.0, np.e, np.e**2]) / (1.0 + np.e + np.e**2)
        np.testing.assert_allclose(probs, expected, rtol=1e-5)

    def test_invalid_temperature(self):
        logits = np.array([1.0, 2.0])
        with self.assertRaises(ValueError):
            m.softmax_with_temperature(logits, temperature=0.0)
        with self.assertRaises(ValueError):
            m.softmax_with_temperature(logits, temperature=-1.0)

class TestTopKFilter(unittest.TestCase):
    def test_top_k_basic(self):
        probs = np.array([0.1, 0.4, 0.3, 0.2])
        filtered = m.top_k_filter(probs, k=2)
        # Top 2 are 0.4 (idx 1) and 0.3 (idx 2).
        # Renormalized: 0.4/(0.4+0.3) = 4/7, 0.3/(0.4+0.3) = 3/7
        expected = np.array([0.0, 4/7, 3/7, 0.0])
        np.testing.assert_allclose(filtered, expected, rtol=1e-5)

    def test_top_k_all(self): 
        probs = np.array([0.1, 0.4, 0.3, 0.2])
        filtered = m.top_k_filter(probs, k=5)
        np.testing.assert_allclose(filtered, probs, rtol=1e-5)

    def test_top_k_invalid(self):
        probs = np.array([0.1, 0.9])
        with self.assertRaises(ValueError):
            m.top_k_filter(probs, k=0)

class TestTopPFilter(unittest.TestCase):
    def test_top_p_basic(self):
        probs = np.array([0.1, 0.5, 0.3, 0.1])
        # Sorted: 0.5 (idx 1), 0.3 (idx 2), 0.1 (idx 0), 0.1 (idx 3)
        # Cumsum: 0.5, 0.8, 0.9, 1.0
        # If p = 0.7: we need 0.5 and 0.3 to exceed 0.7.
        # Renormalized: 0.5/0.8 = 0.625, 0.3/0.8 = 0.375
        filtered = m.top_p_filter(probs, p=0.7)
        expected = np.array([0.0, 0.625, 0.375, 0.0])
        np.testing.assert_allclose(filtered, expected, rtol=1e-5)

    def test_top_p_low(self):
        probs = np.array([0.1, 0.5, 0.3, 0.1])
        # If p = 0.2, we must still keep at least the top element (0.5)
        filtered = m.top_p_filter(probs, p=0.2)
        expected = np.array([0.0, 1.0, 0.0, 0.0])
        np.testing.assert_allclose(filtered, expected, rtol=1e-5)

    def test_top_p_high(self):
        probs = np.array([0.1, 0.5, 0.3, 0.1])
        filtered = m.top_p_filter(probs, p=1.0)
        np.testing.assert_allclose(filtered, probs, rtol=1e-5)

    def test_top_p_invalid(self):
        probs = np.array([0.1, 0.9])
        with self.assertRaises(ValueError):
            m.top_p_filter(probs, p=0.0)
        with self.assertRaises(ValueError):
            m.top_p_filter(probs, p=1.5)

class TestApplyRepetitionPenalty(unittest.TestCase):
    def test_penalty_positive_logits(self):
        logits = np.array([2.0, 4.0, -1.0])
        # Penalty 2.0 on token 1 (value 4.0 -> 2.0)
        modified = m.apply_repetition_penalty(logits, [1], penalty=2.0)
        expected = np.array([2.0, 2.0, -1.0])
        np.testing.assert_allclose(modified, expected, rtol=1e-5)

    def test_penalty_negative_logits(self):
        logits = np.array([2.0, 4.0, -1.0])
        # Penalty 2.0 on token 2 (value -1.0 -> -2.0)
        modified = m.apply_repetition_penalty(logits, [2], penalty=2.0)
        expected = np.array([2.0, 4.0, -2.0])
        np.testing.assert_allclose(modified, expected, rtol=1e-5)

    def test_no_penalty(self):
        logits = np.array([2.0, 4.0, -1.0])
        modified = m.apply_repetition_penalty(logits, [0, 1, 2], penalty=1.0)
        np.testing.assert_allclose(modified, logits, rtol=1e-5)

    def test_invalid_penalty(self):
        logits = np.array([1.0, 2.0])
        with self.assertRaises(ValueError):
            m.apply_repetition_penalty(logits, [0], penalty=0.9)

class TestSampleNextToken(unittest.TestCase):
    def test_sampling_determinism(self):
        probs = np.array([0.1, 0.8, 0.1])
        generator = np.random.default_rng(42)
        samples = [m.sample_next_token(probs, generator) for _ in range(100)]
        # Since token 1 has 80% probability, it should dominate
        self.assertTrue(samples.count(1) > 60)
        # Check that it is deterministic with seed 42
        generator2 = np.random.default_rng(42)
        samples2 = [m.sample_next_token(probs, generator2) for _ in range(100)]
        self.assertEqual(samples, samples2)

class TestGreedyDecode(unittest.TestCase):
    def test_greedy_decode_basic(self):
        # Vocab size 3
        # 0 -> 1 (highest)
        # 1 -> 2 (highest)
        # 2 -> 0 (highest)
        transition_matrix = np.array([
            [-1.0, 0.0, -2.0],  # From 0: 1 is best
            [-3.0, -2.0, -0.5], # From 1: 2 is best
            [0.0, -4.0, -5.0],  # From 2: 0 is best
        ])
        seq = m.greedy_decode(transition_matrix, start_token=0, max_length=5)
        self.assertEqual(seq, [0, 1, 2, 0, 1])

class TestBeamSearch(unittest.TestCase):
    def test_beam_search_vs_greedy(self):
        # Vocab size 3: 0, 1, 2
        # Start token: 0
        # From 0:
        #   0 -> 1: log prob -0.5 (greedy choice)
        #   0 -> 2: log prob -1.0
        # From 1:
        #   1 -> 0, 1, 2: log prob -10.0 (dead end)
        # From 2:
        #   2 -> 0, 1, 2: log prob -0.1 (great path!)
        transition_matrix = np.array([
            [-10.0, -0.5, -1.0],  # From 0
            [-10.0, -10.0, -10.0], # From 1
            [-0.1, -0.1, -0.1],   # From 2
        ])
        
        # Greedy decode should get trapped in the bad path
        greedy_seq = m.greedy_decode(transition_matrix, start_token=0, max_length=3)
        self.assertEqual(greedy_seq, [0, 1, 0]) # 0 -> 1 -> 0 (total log prob: -10.5)
        
        # Beam search with beam_width=2 should find the better path
        beam_seq = m.beam_search(transition_matrix, start_token=0, beam_width=2, max_length=3)
        self.assertEqual(beam_seq, [0, 2, 0]) # 0 -> 2 -> 0 (total log prob: -1.1)

    def test_beam_search_width_one_is_greedy(self):
        # With beam_width=1, beam search should behave exactly like greedy decode
        transition_matrix = np.random.default_rng(123).normal(size=(5, 5))
        greedy_seq = m.greedy_decode(transition_matrix, start_token=0, max_length=10)
        beam_seq = m.beam_search(transition_matrix, start_token=0, beam_width=1, max_length=10)
        self.assertEqual(greedy_seq, beam_seq)

if __name__ == "__main__":
    unittest.main(verbosity=2)
