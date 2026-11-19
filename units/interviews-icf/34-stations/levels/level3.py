"""Level 3 - recursive station-group aggregation.

See docs/12-level3.md for the rules.
"""

from typing import Dict, List

import numpy as np

from stations.api import GroupStats, StationGroup


def aggregate(group: StationGroup, matrix: np.ndarray, index: Dict[str, int]) -> GroupStats:
    """This group's own stations, plus every descendant, combined into one
    GroupStats. A group with zero valid readings anywhere below it has
    mean 0.0."""
    # TODO: your code here
    pass


def find_regional_outliers(
    group: StationGroup, matrix: np.ndarray, index: Dict[str, int], threshold: float
) -> List[str]:
    """Names of every subgroup at any depth (not `group` itself) whose
    aggregate mean differs from `group`'s aggregate mean by more than
    `threshold`. Depth-first, in child order."""
    # TODO: your code here
    pass
