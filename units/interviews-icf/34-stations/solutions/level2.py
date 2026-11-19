"""Level 2 - histogram, outlier station, reproducible audit sample.

Rules:
  * temperature_histogram: np.bincount needs non-negative integer bin
    indices, so floor to int and subtract `low` before counting.
    `minlength` pads the tail so empty high bins still show up.
  * empty_bins: np.where(condition) on a 1-D array is the index list -
    take [0] to drop the tuple-of-one-array wrapper.
  * most_anomalous_station: broadcast means[:, None] against means[None, :]
    to get every pairwise difference in one shape-(n, n) array - no loop
    over station pairs.
  * audit_sample: np.random.RandomState(seed), never the module-level
    np.random - the global generator's state depends on what ran before it,
    so the same seed there does not reproduce the same sample.
"""

from typing import List

import numpy as np


def temperature_histogram(matrix: np.ndarray, low: int, high: int) -> np.ndarray:
    flat = matrix.reshape(-1)
    valid = flat[np.nonzero(~np.isnan(flat))[0]]
    in_range = valid[(valid >= low) & (valid < high)]
    bins = np.floor(in_range).astype(int) - low
    return np.bincount(bins, minlength=high - low)


def empty_bins(histogram: np.ndarray) -> np.ndarray:
    return np.where(histogram == 0)[0]


def most_anomalous_station(matrix: np.ndarray, station_ids: List[str]) -> str:
    means = []
    for row in matrix:
        valid = row[~np.isnan(row)]
        means.append(float(np.sum(valid)) / valid.size if valid.size else 0.0)
    means = np.array(means)

    diffs = np.abs(means[:, None] - means[None, :])
    total_deviation = np.sum(diffs, axis=1)

    winner_row = np.nonzero(total_deviation == np.max(total_deviation))[0][0]
    return station_ids[int(winner_row)]


def audit_sample(station_ids: List[str], k: int, seed: int) -> List[str]:
    rng = np.random.RandomState(seed)
    chosen = rng.choice(len(station_ids), size=k, replace=False)
    return [station_ids[i] for i in sorted(int(i) for i in chosen)]
