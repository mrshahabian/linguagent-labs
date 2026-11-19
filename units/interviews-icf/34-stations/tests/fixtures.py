"""The one dataset every level's tests share.

Four stations over five days (0..4):

    north  1.0  3.0  5.0    .  7.0
    south  2.0    .  2.0  2.0    .
    east      .    .    .    .    .
    west  10.0 12.0 11.0  9.0 13.0

`east` never reports - it exists to test the all-missing case at every
level. `.` is a missing day, not a zero reading.
"""

from stations.api import Reading, StationGroup

STATION_IDS = ["north", "south", "east", "west"]
NUM_DAYS = 5

READINGS = [
    Reading("north", 0, 1.0),
    Reading("north", 1, 3.0),
    Reading("north", 2, 5.0),
    Reading("north", 4, 7.0),
    Reading("south", 0, 2.0),
    Reading("south", 2, 2.0),
    Reading("south", 3, 2.0),
    Reading("west", 0, 10.0),
    Reading("west", 1, 12.0),
    Reading("west", 2, 11.0),
    Reading("west", 3, 9.0),
    Reading("west", 4, 13.0),
]

# The level-3 hierarchy: two leaf regions under one pass-through node under
# the root, so aggregate() and find_regional_outliers() each have to recurse
# more than one level deep.
NE = StationGroup("NE", ("north", "east"))
SW = StationGroup("SW", ("south", "west"))
MID = StationGroup("MID", (), (NE, SW))
ROOT = StationGroup("ROOT", (), (MID,))
