"""Which implementation the level tests exercise.

By default the tests run against `levels/` - the files you write. Set
STATIONS_TARGET=solutions (or run `./run --check`) to run the very same
tests against the reference implementation instead.
"""

import importlib
import os

TARGET = os.environ.get("STATIONS_TARGET", "levels")


def load(level: str):
    """Import levelN from whichever package is being tested."""
    return importlib.import_module(f"{TARGET}.level{level}")
