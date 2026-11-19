"""Types the candidate reads and works with — given code, don't edit it.

Every level operates on the same weather-station dataset: a set of stations,
each reporting a Celsius reading for some subset of days. A day a station
never reported is *missing*, not zero — level 1's job is turning that
absence into ``NaN`` inside a NumPy array, and every later level has to keep
treating it as "no data", not "a very cold day".

See ``docs/`` for the full rules of each level.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Tuple

import numpy as np


class Reading(NamedTuple):
    """One sensor report: station ``station_id`` measured ``celsius`` on
    ``day`` (a zero-based index into the run). A station with no Reading for
    a given day simply didn't report that day — there is no sentinel value
    for "missing" at this layer; that's what level 1 builds."""

    station_id: str
    day: int
    celsius: float


class StationStats(NamedTuple):
    """Summary of one station's valid (non-NaN) readings.

    A station with zero valid readings reports all-zero stats — see
    docs/10-level1.md for why that's the rule rather than raising."""

    minimum: float
    maximum: float
    total: float
    valid_count: int


class StationGroup(NamedTuple):
    """One node of the regional hierarchy used in level 3.

    ``stations`` are the station ids owned directly by this node.
    ``children`` are subgroups; a leaf has ``children == ()``. A node may
    have both its own stations *and* children at once."""

    name: str
    stations: Tuple[str, ...]
    children: Tuple["StationGroup", ...] = ()


class GroupStats(NamedTuple):
    """The aggregate of a StationGroup: every reading in its own stations
    plus every reading in every descendant, combined."""

    total: float
    count: int
    mean: float


def station_index(station_ids: List[str]) -> Dict[str, int]:
    """The row each station occupies in the matrix — build this once and
    pass it down instead of calling ``station_ids.index(...)`` in a loop."""
    return {station_id: row for row, station_id in enumerate(station_ids)}


def render_matrix(matrix: np.ndarray, station_ids: List[str]) -> str:
    """Pretty-print a station x day matrix for debugging. Missing (NaN)
    cells print as '.'. Not used by any test — call it yourself."""
    width = max(len(s) for s in station_ids)
    lines = []
    for station_id, row in zip(station_ids, matrix):
        cells = " ".join("   ." if np.isnan(v) else f"{v:4.1f}" for v in row)
        lines.append(f"{station_id:<{width}}  {cells}")
    return "\n".join(lines)
