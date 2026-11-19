"""Level 2 - histogram, outlier station, reproducible audit sample.

See docs/11-level2.md for the rules.
"""

from typing import List

import numpy as np


def temperature_histogram(matrix: np.ndarray, low: int, high: int) -> np.ndarray:
    """Count every valid reading, floored to an int degree, into bins
    [low, low+1), [low+1, low+2), ..., [high-1, high). Readings outside
    [low, high) are dropped. Returns an array of length (high - low)."""
    # TODO: your code here
    pass


def empty_bins(histogram: np.ndarray) -> np.ndarray:
    """Indices of the bins with zero readings, ascending."""
    # TODO: your code here
    pass


def most_anomalous_station(matrix: np.ndarray, station_ids: List[str]) -> str:
    """The station whose mean reading is furthest, in total absolute
    difference, from every other station's mean. Ties go to whichever
    station comes first in station_ids."""
    # TODO: your code here
    pass


def audit_sample(station_ids: List[str], k: int, seed: int) -> List[str]:
    """k station ids chosen without replacement, deterministically for a
    given seed, in ascending station-index order. Use
    np.random.RandomState(seed) - not the global np.random state."""
    # TODO: your code here
    pass
