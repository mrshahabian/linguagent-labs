# The CodeSignal ICA environment

The pre-certification task is a playground. Its real purpose is not the
Container class — it is **using the environment without thinking about it**, so
that on the day you spend your minutes on the problem instead of on the tooling.
Spend twenty minutes here and it pays for itself.

## The RUN button has three modes

| Mode | Use it when | Trade-off |
|---|---|---|
| **In Terminal** | almost always, while developing | raw output, **your `print()` calls show up**, test order is stable |
| **Structured** | a final check, or when there are many failures | tidy pass/fail list, **debug output is discarded** |
| **Run Custom Script** | you want one test, or no tests | creates `main.sh` and runs whatever you put there |

**Default to `In Terminal`.** Losing `print()` output to `Structured` while
debugging is a genuinely painful way to waste ten minutes.

## `main.sh` — your fast loop

`Run Custom Script` generates `main.sh` with commented examples. The line worth
knowing by heart runs a single test:

```bash
cd tests && python -m unittest basicTests.DemoICATest.test_basic1
```

The pattern is `module.Class.method`. Once one test is failing, narrowing to it
turns a three-second cycle into a fast one, and cuts the noise of twelve other
failures.

Other useful variants:

```bash
python -m unittest discover -s tests -p '*.py'        # everything
cd tests && python -m unittest containerTests          # one file
cd tests && python -m unittest containerTests.DemoICATest.test_05_repetitions2
```

Locally, `./run`, `./run basic`, `./run graded` and `./main.sh` do the same
things.

## Submit early, submit often

**Partial credit is granted per unit test passed.** This changes the optimal
strategy completely from a normal interview:

- Submit as soon as *anything* passes. A submission that passes 4/10 banks
  those 4.
- Never leave broken syntax at the end. A file that does not import scores
  **zero**, even for methods that were finished — the note "unless the project
  fails to build" in the basic-test docstring is doing real work.
- If a method is half-written when time runs out, comment it out and return
  something plausible. Partial beats broken.

## Two test files, two purposes

**`basicTests.py`** — a playground. Modify it, add cases, delete cases. Its
results do **not** affect your score. Its results appear first in the report.
Use it as your scratchpad: when you find a bug, add a two-line reproduction
here rather than re-running the whole graded file.

**`containerTests.py`** — the 10 scored tests, equally weighted, **read-only**.
You are not allowed to modify it, but you *are* invited to read it. Do read it.
The test file is the specification in a way the prose never is — for this task
it is where you learn that:

- `get_median()` on an empty container must raise, and
- `delete()` returns a bool rather than raising, and
- duplicates are kept (not deduplicated), and
- `test_05` and `test_10` are the ones with enough volume to expose a slow
  implementation.

## Read the tests to resolve ambiguity

The prose says "the leftmost integer from the two middle integers". That is one
sentence and it is easy to get backwards. The test says:

```python
self.container.add(30)
self.container.add(10)
self.assertEqual(self.container.get_median(), 10)   # [10, 30] -> 10
```

Unambiguous. **When prose and tests seem to differ, the tests are the spec.**
That habit is worth more than any single algorithm.

## `@timeout(0.1)`

Each test has a 100 ms budget (the task overall allows 3 seconds, 4 GB). This is
generous for the input sizes here — `test_10` does ~400 operations — but it is
tight enough to punish something accidentally quadratic-with-a-big-constant,
like re-sorting inside a loop that itself runs per element.

If you see a timeout rather than a wrong answer, you have an algorithmic
problem, not a logic one. See `02-median-structures.md`.

## The environment's import trick

Every test file starts with:

```python
import inspect, os, sys
current_dir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())
))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from container import Container
```

That is putting the project root on `sys.path` so `tests/` can import
`container.py` from the directory above. Worth recognising — if you ever add
your own helper module, it lives next to `container.py`, not inside `tests/`.

## IDE settings

The `IDE Settings` tab (lower left) has the editor keymap and hotkeys. Two
minutes there before the clock starts: check that your muscle-memory shortcuts
for *save*, *run*, *comment block* and *multi-cursor* do what you expect.
Fighting an unfamiliar keymap costs more time than any algorithm.

## Checklist before the real thing

- [ ] I can run one single test from `main.sh` without looking up the syntax.
- [ ] I know `In Terminal` keeps my `print()` output and `Structured` does not.
- [ ] I have read every graded test before writing code.
- [ ] I submit as soon as one test goes green, then keep going.
- [ ] My file always imports, even mid-edit.
