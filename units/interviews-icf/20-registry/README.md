# Mock Assessment 1 — Package Registry

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/interviews-icf/20-registry/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/icf/registry?utm_source=github&utm_medium=labs)

A five-level CodeSignal ICF-style exercise. 45 tests. 90 minutes.

You implement an in-memory package registry that grows, level by level, from
plain CRUD into a system with TTLs, snapshots and rollback, and download
rankings. It is the canonical ICF shape: each level adds a dimension that
forces you to reorganise what you built for the last one.

| Level | Adds | The one new idea |
|---|---|---|
| 1 | `publish` / `get` / `delete` | a dict, and returning bools rather than raising |
| 2 | `search` / `total_size` | compound sort key, capped at 5 |
| 3 | Timestamps and TTLs | half-open `[start, start+ttl)`, and delegation |
| 4 | `backup` / `restore` | saving *remaining* lifetime, not absolute expiry |
| 5 | `download_at` / `top_downloaded` | state attached to the name, surviving everything |

## Before you start

Set the room up like the real thing:

- Phone in another room. Notifications off. One monitor.
- **Disable Copilot / any AI completion in your editor.** Check it, don't
  assume it.
- No browser tabs except `https://docs.python.org/3/`. Nothing else.
- Do not open `locked/` or `solution.py` by hand. That is the whole point.
- Start `python3 timer.py` in a second terminal before you write a line.

## Running it

```
./unlock 1          # reveals spec/level1.md and tests/test_level1.py
./run 1             # runs levels 1..1
./run               # runs every unlocked level
./run --check       # runs ALL 45 tests against solution.py
```

Unlock the next level only once the current one is green. Re-run all
unlocked levels after every change — later levels break earlier ones.

## Rules

- Write everything in `registry.py`. Standard library only.
- The tests are the specification. Where the prose is ambiguous, the tests
  decide. Do not handle edge cases no test checks.
- If you get stuck, move on. A clean stop at Level 3 beats a broken Level 5.
- When the timer ends, stop typing mid-word.

## Suggested checkpoints

| Elapsed | You should be |
|---|---|
| 0:10 | Level 1 green |
| 0:25 | Level 2 green |
| 0:45 | Level 3 green |
| 1:10 | Level 4 green |
| 1:30 | Level 5 green |

If you are more than ten minutes behind a checkpoint, skip ahead to the next
level rather than grinding.

## The three ideas this mock is really testing

**1. Delegation, at Level 3.** Every Level 1–2 method gets a timestamped twin.
The spec says to treat the old ones as the new ones called at `timestamp = 0`.
Write that literally — `def get(self, name): return self.get_at(0, name)` — and
Level 3 costs you ten minutes. Duplicate the logic instead and every later
level costs you double.

**2. Half-open intervals, everywhere.** Alive means `start <= now < start + ttl`.
At exactly `start + ttl` it is gone. One `<=` in the wrong place fails a
scattering of tests across three levels and reads like a different bug each
time.

**3. Remaining vs absolute, at Level 4.** A backup stores how much life a
package has *left*, not when it expires. Restoring at a later timestamp
re-bases that remaining lifetime onto the new time. Storing the absolute expiry
passes the simple restore tests and fails the ones that restore at a different
timestamp.

## Check your understanding

1. **Level 3 adds a timestamp to every existing method. Why does
   `def get(self, name): return self.get_at(0, name)` cost ten minutes,
   while duplicating the logic costs double?**
   <details><summary>Answer</summary>Delegating means the old method's
   behavior comes free once the timestamped version is correct — only
   one implementation to get right and maintain. Duplicating means every
   later level's changes have to be applied to both copies, and they can
   silently drift apart.</details>

2. **"Alive means `start <= now < start + ttl`." What happens at exactly
   `now == start + ttl`?**
   <details><summary>Answer</summary>It's expired — the interval is
   half-open, so the upper bound is excluded. A single `<=` where a
   `<` belongs is a one-character bug that shows up as failures
   scattered across several tests and levels.</details>

3. **Why does storing a snapshot's *absolute* expiry timestamp instead
   of its *remaining* lifetime pass the simple restore tests but fail
   the others?**
   <details><summary>Answer</summary>A snapshot taken at one timestamp
   and restored at a later one needs its lifetime re-based onto the new
   time. Storing an absolute expiry ignores how much time has passed
   between snapshot and restore — it happens to work when restoring
   immediately, but not when restoring later.</details>

4. **The rules say "do not handle edge cases no test checks." Why is
   that the right instruction here, rather than just being
   thorough?**
   <details><summary>Answer</summary>Time spent handling untested edge
   cases is time not spent on the next level, and doesn't earn any
   credit — the tests *are* the specification, so an edge case with no
   test isn't actually part of the graded spec.</details>

## Afterwards

Write down, before you look at anything:

1. Which level cost the most time, and why.
2. What you had to rewrite from an earlier level.
3. Every typo or syntax error you hit — these are the cheapest minutes to
   recover.

Then read `solution.py` and compare **structure**, not syntax. If your Level 3
does not delegate, that is the lesson.

## Related

- `../08-drills/` — Blocks D, E and F drill exactly the primitives this mock
  needs. Do them first.
- `../09-container/` — a shorter warm-up: platform mechanics plus one task.
- `../21-scheduler/` — a harder five-level mock with a runnable
  simulator and a full course.
- `../34-stations/` — a shorter three-level mock, NumPy instead of
  standard-library Python.
- `../33-trees/` — the other assessment format: debug a broken NumPy
  implementation rather than build one from a spec.
