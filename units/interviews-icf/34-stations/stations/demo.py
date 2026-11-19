"""Interactive playground - not graded, just for exploring the dataset.

Run against your own code in levels/:

    python3 -m stations.demo

Or against the reference solution, to see the target behaviour before you
start:

    STATIONS_TARGET=solutions python3 -m stations.demo

Or drop into a REPL and call pieces yourself:

    >>> from tests.fixtures import STATION_IDS, NUM_DAYS, READINGS
    >>> from levels.level1 import build_matrix
    >>> from stations.api import render_matrix
    >>> print(render_matrix(build_matrix(STATION_IDS, NUM_DAYS, READINGS), STATION_IDS))
"""

import importlib
import os

from stations.api import station_index, render_matrix
from tests.fixtures import NUM_DAYS, READINGS, ROOT, STATION_IDS


def _load(level):
    target = os.environ.get("STATIONS_TARGET", "levels")
    return importlib.import_module(f"{target}.level{level}")


def main():
    level1, level2, level3 = _load(1), _load(2), _load(3)

    matrix = level1.build_matrix(STATION_IDS, NUM_DAYS, READINGS)
    print(render_matrix(matrix, STATION_IDS))
    print()

    stats = level1.station_stats(matrix, STATION_IDS)
    for station_id in STATION_IDS:
        print(f"{station_id:<6} {stats[station_id]}")
    print()

    hist = level2.temperature_histogram(matrix, 0, 15)
    print("histogram   ", [int(v) for v in hist])
    print("empty bins  ", [int(v) for v in level2.empty_bins(hist)])
    print("most anomalous station:", level2.most_anomalous_station(matrix, STATION_IDS))
    print("audit sample (seed=7): ", level2.audit_sample(STATION_IDS, 2, seed=7))
    print()

    index = station_index(STATION_IDS)
    print("ROOT aggregate:", level3.aggregate(ROOT, matrix, index))
    print("outliers @ threshold 2.0:", level3.find_regional_outliers(ROOT, matrix, index, 2.0))


if __name__ == "__main__":
    main()
