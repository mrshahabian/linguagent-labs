"""Level 2's worked example: histogram, outlier station, audit sample.

See docs/11-level2.md for the rules these check.
"""

import unittest

import numpy as np

from ._target import load
from .fixtures import NUM_DAYS, READINGS, STATION_IDS

level1 = load("1")
level2 = load("2")


def matrix():
    return level1.build_matrix(STATION_IDS, NUM_DAYS, READINGS)


class HistogramTests(unittest.TestCase):
    def test_counts_and_length(self):
        hist = level2.temperature_histogram(matrix(), 0, 15)
        self.assertEqual(len(hist), 15)
        self.assertEqual(int(np.sum(hist)), 12)  # 12 valid readings total
        expected = [0, 1, 3, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
        self.assertEqual(list(hist), expected)

    def test_out_of_range_readings_are_dropped(self):
        hist = level2.temperature_histogram(matrix(), 0, 10)  # west's 10-13 excluded
        self.assertEqual(len(hist), 10)
        self.assertEqual(int(np.sum(hist)), 8)  # 12 - the four >= 10


class EmptyBinsTests(unittest.TestCase):
    def test_indices_ascending(self):
        hist = level2.temperature_histogram(matrix(), 0, 15)
        self.assertEqual(list(level2.empty_bins(hist)), [0, 4, 6, 8, 14])

    def test_no_empty_bins(self):
        hist = np.array([1, 2, 3])
        self.assertEqual(list(level2.empty_bins(hist)), [])


class MostAnomalousStationTests(unittest.TestCase):
    def test_west_is_furthest_from_the_rest(self):
        # means: north 4.0, south 2.0, east 0.0, west 11.0 - west's total
        # absolute deviation from the other three is largest.
        self.assertEqual(level2.most_anomalous_station(matrix(), STATION_IDS), "west")

    def test_tie_goes_to_earlier_station(self):
        # Two stations, identical (non-NaN) readings: both have deviation 0,
        # so the tie-break is "earlier in station_ids".
        m = np.array([[1.0, 2.0], [1.0, 2.0]])
        self.assertEqual(level2.most_anomalous_station(m, ["a", "b"]), "a")


class AuditSampleTests(unittest.TestCase):
    def test_deterministic_for_a_given_seed(self):
        self.assertEqual(level2.audit_sample(STATION_IDS, 2, seed=7), ["south", "east"])

    def test_same_seed_same_result(self):
        first = level2.audit_sample(STATION_IDS, 3, seed=42)
        second = level2.audit_sample(STATION_IDS, 3, seed=42)
        self.assertEqual(first, second)

    def test_ascending_station_index_order(self):
        sample = level2.audit_sample(STATION_IDS, 2, seed=7)
        indices = [STATION_IDS.index(s) for s in sample]
        self.assertEqual(indices, sorted(indices))

    def test_no_repeats(self):
        sample = level2.audit_sample(STATION_IDS, 4, seed=1)
        self.assertEqual(len(sample), len(set(sample)))


if __name__ == "__main__":
    unittest.main()
