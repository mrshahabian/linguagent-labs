"""Level 1 - build the matrix, then per-station stats.

Rules:
  * build_matrix: one row per station (in station_ids order), one column
    per day. A cell starts as NaN; a Reading overwrites its cell.
  * station_stats: mask NaN out *before* calling np.min/np.max/np.sum - all
    three propagate NaN if you don't. A station with no valid days reports
    (0.0, 0.0, 0.0, 0) rather than raising (np.min of an empty array does).
"""

from typing import Dict, List

import numpy as np

from stations.api import Reading, StationStats, station_index


def build_matrix(station_ids: List[str], num_days: int, readings: List[Reading]) -> np.ndarray:
    index = station_index(station_ids)
    grid = [[float("nan")] * num_days for _ in station_ids]
    for reading in readings:
        grid[index[reading.station_id]][reading.day] = reading.celsius
    return np.array(grid)


def station_stats(matrix: np.ndarray, station_ids: List[str]) -> Dict[str, StationStats]:
    stats = {}
    for row, station_id in zip(matrix, station_ids):
        valid = row[~np.isnan(row)]
        if valid.size == 0:
            stats[station_id] = StationStats(0.0, 0.0, 0.0, 0)
        else:
            stats[station_id] = StationStats(
                minimum=float(np.min(valid)),
                maximum=float(np.max(valid)),
                total=float(np.sum(valid)),
                valid_count=int(valid.size),
            )
    return stats
