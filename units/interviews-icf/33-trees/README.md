# Mock Assessment 4 — Extremely Randomized Trees

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/interviews-icf/33-trees/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/icf/trees?utm_source=github&utm_medium=labs)

**A different format from the other mocks: nothing to build, everything to
fix.** You are given a working-ish implementation of a real machine-learning
algorithm with seven bugs in it. It sometimes crashes and sometimes gets poor
accuracy. Make the tests pass by fixing the underlying issues.

35 tests. 60 minutes. Requires NumPy.

```bash
pip install -r requirements.txt
./run             # test trees.py — your work
./run --check     # test solution.py — the reference, to see them pass
./run --reset     # restore the original buggy trees.py
```

You can also run the tests exactly as the assessment does:

```bash
python3 tests/test_trees.py
```

## Why this format matters

Every other mock in this repo asks you to write code from a specification.
This one asks you to **read code you did not write, form a hypothesis, and
test it** — which is what the job actually is most days, and a distinct
assessment format you should have practised at least once.

The skills it exercises are different too:

- Reading a docstring as a **specification** and spotting where the code below
  it quietly disagrees.
- Knowing NumPy's sharp edges: `nan != nan`, `0 * log2(0)` being NaN rather
  than 0, `nanmin` on an all-NaN column, scalar functions widening to float64,
  and `[expr] * n` repeating one object.
- Debugging a **stochastic** system, where "run it again" gives a different
  answer and a failure may only appear on some seeds.

## The rules

- Write your fixes in `trees.py`. The docstrings in it are **correct** — treat
  them as the spec and make the code do what they already say.
- Do not modify `tests/test_trees.py`. If you want to experiment with a case,
  copy it into a scratch script.
- Do not "hack" the tests — no hardcoding for particular inputs, no overriding
  assertion methods.
- Standard library plus **NumPy** only. No scikit-learn.
- Your code does not need to be fast, but don't make it dramatically slower —
  for example, don't replace vectorised NumPy with nested Python loops.
- Don't open `solution.py` until you're done.

## The algorithm, in one paragraph

An ordinary decision tree searches every candidate threshold on a feature for
the one that separates the labels best. An **extremely randomized** tree draws
a *single* threshold per candidate feature, uniformly at random between that
feature's minimum and maximum **within the current node**, and keeps the best
of those few random candidates. That sounds much worse and mostly isn't: each
tree is far cheaper to build so you can afford many more, and the extra
randomness decorrelates the trees, which is exactly what makes an ensemble
average well.

Splits are scored by normalised information gain, using Shannon entropy in
bits. Missing values are handled by giving every internal node a
`nans_go_left` flag: a row whose value for that feature is NaN follows the
flag instead of being compared against the threshold.

Full details in `docs/01-algorithm.md`.

## Suggested approach

1. **Run the tests first.** Read the failures before reading the code — they
   tell you which behaviours are wrong, which narrows the search enormously.
2. **Read each failing test's name and docstring.** They are written to name
   the property that is broken.
3. **Compare the two prediction paths.** `predict_single` and `predict_batch`
   are documented as equivalent. When two implementations of one rule
   disagree, one of them is the bug — and the tests will tell you which.
4. **Fix one thing, re-run.** Some failures share a cause; some causes produce
   several failures. Don't batch four changes and then wonder which helped.

## Checkpoints

| Elapsed | You should have |
|---|---|
| 0:05 | tests run, failures read, no code changed yet |
| 0:20 | the entropy and scoring bugs found |
| 0:40 | the prediction and feature-selection bugs found |
| 0:55 | all 35 green |

If you are stuck for more than ten minutes on one failure, move to another —
the bugs are independent, and each one you fix removes noise from the rest.

## Check your understanding

1. **Why is `0 * log2(0)` NaN in NumPy instead of the mathematically
   conventional 0?**
   <details><summary>Answer</summary><code>log2(0)</code> is
   <code>-inf</code>, and <code>0 * -inf</code> is an indeterminate form
   — NumPy (like IEEE 754 generally) returns NaN for it rather than
   silently applying the calculus convention that the limit is 0. Code
   that assumes multiplying by zero always zeroes out a term will
   propagate NaN instead.</details>

2. **Why does `nan != nan` evaluate to `True`, and what does that break
   if you're not expecting it?**
   <details><summary>Answer</summary>NaN is defined to compare unequal
   to everything, including itself — this is IEEE 754 behavior, not a
   bug. Code that checks "did this change?" with <code>!=</code> will
   report a false change whenever a value stays NaN across two
   reads.</details>

3. **What does `nanmin` return for a column that's entirely NaN, and why
   is that worth checking for explicitly?**
   <details><summary>Answer</summary>NaN, with a runtime warning —
   <code>nanmin</code> only skips NaNs when there's at least one real
   value to fall back on. An all-NaN column has nothing to fall back to,
   so the "ignore NaN" function still returns NaN, silently.</details>

4. **When `predict_single` and `predict_batch` disagree on the same
   input, how do you use the tests to figure out which one is
   actually wrong?**
   <details><summary>Answer</summary>Both are documented as equivalent,
   so the docstring is the spec — check which one's *behavior* matches
   what the docstring says the function should do. Whichever one the
   docstring's own description doesn't match is the bug, not necessarily
   whichever one "looks more correct" at a glance.</details>

5. **Why does `[expr] * n` silently break code that expects `n`
   independent objects?**
   <details><summary>Answer</summary><code>expr</code> is evaluated
   *once*, and the resulting single object is repeated `n` times by
   reference — mutating one of the `n` "copies" mutates all of them,
   since they're actually the same object.</details>

6. **The task is stochastic — "run it again" gives a different answer.
   How does that change how you should interpret a single failing
   test run?**
   <details><summary>Answer</summary>A failure might be a genuine bug,
   or it might be an unlucky seed producing a legitimately rare (but
   valid) outcome — re-running before concluding a fix worked, and being
   skeptical of a single passing run "proving" correctness, both
   matter more here than in a deterministic task.</details>

## Afterwards

`docs/02-bug-hunt.md` has the full catalogue with explanations — **spoilers**,
so only open it once you're done or genuinely stuck. It covers what each bug
was, why it produced the symptom it did, and the general NumPy lesson behind
it.

Then write down, before looking at anything:

1. Which bug took longest, and what finally gave it away.
2. Whether you found it by reading or by running.
3. Any moment you changed code without a hypothesis — those are the minutes
   you lose.

## Files

```
trees.py              you work here (seven bugs)
trees.template.py     pristine buggy copy (./run --reset restores it)
solution.py           reference implementation
tests/test_trees.py   35 tests — the specification
docs/01-algorithm.md  how the algorithm works, from scratch
docs/02-bug-hunt.md   debugging method + the bug catalogue (spoilers)
```

## Related

- `../08-drills/` — the primitives. Blocks A and B still apply.
- `../20-registry/` — the build-from-spec format this unit's debugging
  contrasts with.
- `../21-scheduler/` — the build-from-spec format, with a full course.
- `../09-container/` — a shorter warm-up on choosing a data structure.
- `../34-stations/` — build-from-spec instead of debug, but the same
  NumPy NaN semantics this unit tests.
