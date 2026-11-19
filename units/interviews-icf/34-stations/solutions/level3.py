"""Level 3 - recursive station-group aggregation.

Rules:
  * aggregate: sum this node's own stations (masking NaN, same as level 1)
    plus the recursive aggregate of every child. A leaf is the base case;
    a node with children but no stations of its own still works - the
    `for station_id in group.stations` loop is just empty.
  * find_regional_outliers: `group`'s aggregate mean is fixed once, at the
    top - it is the baseline every descendant is compared against, not
    each node's own parent's mean. Recurse into every child regardless of
    whether that child was flagged.
"""

from typing import Dict, List

import numpy as np

from stations.api import GroupStats, StationGroup


def aggregate(group: StationGroup, matrix: np.ndarray, index: Dict[str, int]) -> GroupStats:
    total = 0.0
    count = 0

    for station_id in group.stations:
        row = matrix[index[station_id]]
        valid = row[~np.isnan(row)]
        total += float(np.sum(valid))
        count += int(valid.size)

    for child in group.children:
        child_stats = aggregate(child, matrix, index)
        total += child_stats.total
        count += child_stats.count

    mean = total / count if count else 0.0
    return GroupStats(total=total, count=count, mean=mean)


def find_regional_outliers(
    group: StationGroup, matrix: np.ndarray, index: Dict[str, int], threshold: float
) -> List[str]:
    root_mean = aggregate(group, matrix, index).mean

    def walk(node: StationGroup) -> List[str]:
        outliers = []
        for child in node.children:
            if abs(aggregate(child, matrix, index).mean - root_mean) > threshold:
                outliers.append(child.name)
            outliers.extend(walk(child))
        return outliers

    return walk(group)
