"""Level 3's worked example: recursive aggregation over a station tree.

See docs/12-level3.md for the rules these check.

Tree (see tests/fixtures.py):

    ROOT
      MID
        NE  = {north, east}
        SW  = {south, west}
"""

import unittest

from stations.api import GroupStats, StationGroup, station_index

from ._target import load
from .fixtures import MID, NE, NUM_DAYS, READINGS, ROOT, STATION_IDS, SW

level1 = load("1")
level3 = load("3")


def matrix_and_index():
    matrix = level1.build_matrix(STATION_IDS, NUM_DAYS, READINGS)
    return matrix, station_index(STATION_IDS)


class AggregateTests(unittest.TestCase):
    def test_leaf_combines_its_own_stations(self):
        matrix, index = matrix_and_index()
        # north: total 16.0, count 4. east: all missing, contributes 0/0.
        self.assertEqual(level3.aggregate(NE, matrix, index), GroupStats(16.0, 4, 4.0))

    def test_leaf_with_no_valid_readings_anywhere(self):
        matrix, index = matrix_and_index()
        only_east = StationGroup("only-east", ("east",))
        self.assertEqual(level3.aggregate(only_east, matrix, index), GroupStats(0.0, 0, 0.0))

    def test_pass_through_node_sums_its_children(self):
        matrix, index = matrix_and_index()
        stats = level3.aggregate(MID, matrix, index)
        self.assertEqual(stats.total, 77.0)
        self.assertEqual(stats.count, 12)
        self.assertAlmostEqual(stats.mean, 77.0 / 12)

    def test_node_with_both_own_stations_and_children(self):
        matrix, index = matrix_and_index()
        mixed = StationGroup("mixed", ("north",), (SW,))
        stats = level3.aggregate(mixed, matrix, index)
        self.assertEqual(stats.total, 16.0 + 61.0)
        self.assertEqual(stats.count, 4 + 8)


class FindRegionalOutliersTests(unittest.TestCase):
    def test_threshold_two_flags_only_ne(self):
        matrix, index = matrix_and_index()
        self.assertEqual(level3.find_regional_outliers(ROOT, matrix, index, 2.0), ["NE"])

    def test_tighter_threshold_flags_both_leaves(self):
        matrix, index = matrix_and_index()
        self.assertEqual(level3.find_regional_outliers(ROOT, matrix, index, 1.0), ["NE", "SW"])

    def test_loose_threshold_flags_nothing(self):
        matrix, index = matrix_and_index()
        self.assertEqual(level3.find_regional_outliers(ROOT, matrix, index, 100.0), [])

    def test_root_itself_is_never_returned(self):
        matrix, index = matrix_and_index()
        outliers = level3.find_regional_outliers(ROOT, matrix, index, 0.0)
        self.assertNotIn("ROOT", outliers)


if __name__ == "__main__":
    unittest.main()
