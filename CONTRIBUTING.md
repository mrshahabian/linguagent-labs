# Contributing

Thank you! A few things that keep the labs trustworthy:

- **Every exercise has a stub, a reference and tests**: the tests must fail on
  `exercises.py` and pass on `solutions.py`. `python -m labkit verify` checks
  the second half; CI runs it on every push and pull request.
- **Lessons and notebooks are built together.** Each unit's `lab.ipynb` is
  generated from its `README.md` and code, so a fix to the lesson text or the
  exercises should go in those files; mention it in your pull request and the
  notebook is rebuilt with it.
- **Found a wrong answer, a confusing sentence, or a test that passes for the
  wrong reason?** Open an issue with the unit number — that is the most useful
  contribution of all.
