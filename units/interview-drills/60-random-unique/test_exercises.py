"""Tests for 60 — Random unique numbers. Read them; they are the specification."""

import importlib
import os
import unittest
import random

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestSampleWithoutReplacement(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(42)

    def test_basic_sampling(self):
        lst = [10, 20, 30, 40, 50]
        sample = m.sample_without_replacement(lst, 3, self.rng)
        self.assertEqual(len(sample), 3)
        self.assertTrue(all(x in lst for x in sample))
        self.assertEqual(len(set(sample)), 3)  # Unique

    def test_does_not_mutate_original(self):
        lst = [1, 2, 3, 4, 5]
        original = list(lst)
        m.sample_without_replacement(lst, 3, self.rng)
        self.assertEqual(lst, original)

    def test_k_equals_zero(self):
        lst = [1, 2, 3]
        self.assertEqual(m.sample_without_replacement(lst, 0, self.rng), [])

    def test_k_equals_length(self):
        lst = [1, 2, 3]
        sample = m.sample_without_replacement(lst, 3, self.rng)
        self.assertEqual(sorted(sample), [1, 2, 3])

    def test_invalid_k_raises_value_error(self):
        lst = [1, 2, 3]
        with self.assertRaises(ValueError):
            m.sample_without_replacement(lst, 4, self.rng)
        with self.assertRaises(ValueError):
            m.sample_without_replacement(lst, -1, self.rng)

    def test_deterministic_output(self):
        lst = list(range(100))
        rng1 = random.Random(100)
        rng2 = random.Random(100)
        res1 = m.sample_without_replacement(lst, 10, rng1)
        res2 = m.sample_without_replacement(lst, 10, rng2)
        self.assertEqual(res1, res2)


class TestReservoirSample(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(42)

    def test_stream_smaller_than_k(self):
        stream = iter([1, 2, 3])
        # Should return all elements in order
        self.assertEqual(m.reservoir_sample(stream, 5, self.rng), [1, 2, 3])

    def test_stream_exact_k(self):
        stream = iter([1, 2, 3, 4, 5])
        self.assertEqual(m.reservoir_sample(stream, 5, self.rng), [1, 2, 3, 4, 5])

    def test_stream_larger_than_k(self):
        stream = iter(range(100))
        sample = m.reservoir_sample(stream, 10, self.rng)
        self.assertEqual(len(sample), 10)
        self.assertEqual(len(set(sample)), 10)
        self.assertTrue(all(0 <= x < 100 for x in sample))

    def test_invalid_k_raises_value_error(self):
        with self.assertRaises(ValueError):
            m.reservoir_sample(iter([1, 2]), -1, self.rng)

    def test_deterministic_reservoir(self):
        rng1 = random.Random(12345)
        rng2 = random.Random(12345)
        res1 = m.reservoir_sample(iter(range(1000)), 5, rng1)
        res2 = m.reservoir_sample(iter(range(1000)), 5, rng2)
        self.assertEqual(res1, res2)


class TestUniqueRandomGenerator(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(42)

    def test_small_range_exhaustion(self):
        gen = m.UniqueRandomGenerator(5, self.rng)
        drawn = []
        for _ in range(5):
            self.assertTrue(gen.has_next())
            drawn.append(gen.draw())
        
        self.assertFalse(gen.has_next())
        self.assertEqual(sorted(drawn), [0, 1, 2, 3, 4])
        
        with self.assertRaises(IndexError):
            gen.draw()

    def test_huge_range_lazy_memory(self):
        # 1 billion elements. If it allocates an array, it will crash/timeout.
        gen = m.UniqueRandomGenerator(1_000_000_000, self.rng)
        self.assertTrue(gen.has_next())
        drawn = [gen.draw() for _ in range(100)]
        self.assertEqual(len(drawn), 100)
        self.assertEqual(len(set(drawn)), 100)
        self.assertTrue(all(0 <= x < 1_000_000_000 for x in drawn))

    def test_invalid_n_raises_value_error(self):
        with self.assertRaises(ValueError):
            m.UniqueRandomGenerator(-5, self.rng)

    def test_n_equals_zero(self):
        gen = m.UniqueRandomGenerator(0, self.rng)
        self.assertFalse(gen.has_next())
        with self.assertRaises(IndexError):
            gen.draw()

    def test_deterministic_draws(self):
        rng1 = random.Random(999)
        rng2 = random.Random(999)
        gen1 = m.UniqueRandomGenerator(10000, rng1)
        gen2 = m.UniqueRandomGenerator(10000, rng2)
        draws1 = [gen1.draw() for _ in range(50)]
        draws2 = [gen2.draw() for _ in range(50)]
        self.assertEqual(draws1, draws2)


class TestWeightedChoice(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(42)

    def test_single_choice(self):
        self.assertEqual(m.weighted_choice([10.0], self.rng), 0)

    def test_zero_weight_ignored(self):
        # Index 1 has 0 weight, should never be chosen
        weights = [10.0, 0.0, 10.0]
        choices = [m.weighted_choice(weights, self.rng) for _ in range(100)]
        self.assertNotIn(1, choices)
        self.assertTrue(all(c in (0, 2) for c in choices))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            m.weighted_choice([], self.rng)  # Empty
        with self.assertRaises(ValueError):
            m.weighted_choice([-1.0, 2.0], self.rng)  # Negative weight
        with self.assertRaises(ValueError):
            m.weighted_choice([0.0, 0.0], self.rng)  # Sum is zero

    def test_distribution_fairness(self):
        # Large sample to verify statistical correctness
        weights = [1.0, 3.0, 6.0]  # 10%, 30%, 60%
        counts = {0: 0, 1: 0, 2: 0}
        for _ in range(10000):
            idx = m.weighted_choice(weights, self.rng)
            counts[idx] += 1
        
        # Allow a reasonable margin of error for 10000 trials
        self.assertAlmostEqual(counts[0] / 10000, 0.10, delta=0.02)
        self.assertAlmostEqual(counts[1] / 10000, 0.30, delta=0.02)
        self.assertAlmostEqual(counts[2] / 10000, 0.60, delta=0.02)

    def test_deterministic_weighted_choice(self):
        weights = [1.5, 2.5, 10.0, 0.5]
        rng1 = random.Random(777)
        rng2 = random.Random(777)
        choices1 = [m.weighted_choice(weights, rng1) for _ in range(50)]
        choices2 = [m.weighted_choice(weights, rng2) for _ in range(50)]
        self.assertEqual(choices1, choices2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
