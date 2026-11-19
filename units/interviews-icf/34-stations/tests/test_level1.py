"""Level 1's worked example, asserted step by step.

See docs/10-level1.md for the rules these check.
"""

import math
import unittest

import numpy as np

from stations.api import StationStats

from ._target import load
from .fixtures import NUM_DAYS, READINGS, STATION_IDS

level1 = load("1")


class BuildMatrixTests(unittest.TestCase):
    def setUp(self):
        self.matrix = level1.build_matrix(STATION_IDS, NUM_DAYS, READINGS)

    def test_shape(self):
        self.assertEqual(self.matrix.shape, (len(STATION_IDS), NUM_DAYS))

    def test_reported_cells(self):
        self.assertEqual(self.matrix[0, 0], 1.0)  # north, day 0
        self.assertEqual(self.matrix[0, 2], 5.0)  # north, day 2
        self.assertEqual(self.matrix[3, 4], 13.0)  # west, day 4

    def test_missing_cells_are_nan(self):
        self.assertTrue(math.isnan(self.matrix[0, 3]))  # north never reported day 3
        self.assertTrue(np.all(np.isnan(self.matrix[2])))  # east never reported at all


class StationStatsTests(unittest.TestCase):
    def setUp(self):
        matrix = level1.build_matrix(STATION_IDS, NUM_DAYS, READINGS)
        self.stats = level1.station_stats(matrix, STATION_IDS)

    def test_every_station_present(self):
        self.assertEqual(set(self.stats), set(STATION_IDS))

    def test_north(self):
        self.assertEqual(self.stats["north"], StationStats(1.0, 7.0, 16.0, 4))

    def test_south(self):
        self.assertEqual(self.stats["south"], StationStats(2.0, 2.0, 6.0, 3))

    def test_west(self):
        self.assertEqual(self.stats["west"], StationStats(9.0, 13.0, 55.0, 5))

    def test_all_missing_station_is_zeroed_not_raised(self):
        # `east` never reported. np.min/np.max on an empty slice raise - the
        # spec says report zeros instead.
        self.assertEqual(self.stats["east"], StationStats(0.0, 0.0, 0.0, 0))


if __name__ == "__main__":
    unittest.main()
