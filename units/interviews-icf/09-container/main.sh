#!/usr/bin/env bash
# Mirrors the platform's "Run Custom Script" option.
# Uncomment the line you want, or add your own.

cd "$(dirname "$0")"

# Everything:
python3 -m unittest discover -s tests -p '*.py' 2>&1

# One file:
# cd tests && python3 -m unittest containerTests 2>&1

# One single test — the fastest loop while you are getting started:
# cd tests && python3 -m unittest basicTests.DemoICATest.test_basic1 2>&1

# Scratch experimenting, no tests at all:
# python3 -c "from container import Container; c=Container(); c.add(1); print(c.get_median())"
