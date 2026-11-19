"""Level 1 - build the matrix, then per-station stats.

See docs/10-level1.md for the rules.
"""

from typing import Dict, List

import numpy as np
from numpy.ma.core import minimum

from stations.api import Reading, StationStats


def build_matrix(station_ids: List[str], num_days: int, readings: List[Reading]) -> np.ndarray:
    """Return a (len(station_ids) x num_days) float array. A cell with no
    matching Reading is NaN."""
    # TODO: your code here
    index = {station_id: i for i , station_id in enumerate(station_ids)}
    grid = [[float("nan")] * num_days for _ in station_ids]
    for reading in readings:
        grid[index[reading.station_id]][reading.day] = reading.celsius
    return np.array(grid)


def station_stats(matrix: np.ndarray, station_ids: List[str]) -> Dict[str, StationStats]:
    """One StationStats per station, computed over that station's non-NaN
    days only."""
    # TODO: your code here
    stats = {}
    for row,  station_id in zip(matrix, station_ids):
        valid = row[~np.isnan(row)]
        if valid.size == 0:
            stats[station_id] = StationStats(0.0, 0.0, 0.0, 0)
        else:
            stats[station_id] = StationStats(
            minimum = float(np.min(valid)),
            maximum = float(np.max(valid)),
            total = float(np.sum(valid)),
            valid_count = int(valid.size)
            )


    return stats
