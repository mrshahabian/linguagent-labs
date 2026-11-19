# Prerequisite: running and reading the tests

You do not have to write tests here, but you have to *use* them, and the
difference between a slow session and a fast one is mostly how quickly you
turn a red test into a hypothesis.

## Running things

```
./run             every level test, against levels/
./run 1           only Level 1 (also: 2, 3a, 3b, 3c)
./run rules       the engine-rule tests
./run --check     the same tests against solutions/
```

Underneath, that is just pytest:

```
python3 -m pytest tests/ -q                              # quiet
python3 -m pytest tests/test_worked_examples.py -q -k level3b
python3 -m pytest tests/ -x                              # stop at first failure
python3 -m pytest tests/ -q --tb=line                    # one line per failure
```

`-k` matches against test names — `-k level3b` runs every test with `level3b`
in its name. `-x` is the one to reach for when you have twelve failures and
want to work on the first.

## Reading a failure

The level tests are written so the expected table is in the docstring, right
above the failure:

```
def test_level1_readme_example():
    """max_work=10; A(6 prompt, 3 max_tokens), B(3, 2), C(5, 2) at step 0.

    step  waiting  admitted  prefill    decode  finished
    0     A, B, C  -         A:6, B:3   -       -
    1     C        A, B      C:5        A, B    B
    2     -        A, C      -          A, C    A, C
    """
```

and the assertion prints your actual trace next to the expected row:

```
step 1 diverges
  expected (1, ('C',), ('A', 'B'), (), ('C:5',), ('A', 'B'), ('B',))
  actual   (1, ('C',), ('A', 'B'), (), (),       ('A', 'B'), ('B',))

step  waiting  admitted  preempt  prefill   decode  finished
0     A, B, C  -         -        A:6, B:3  -       -
1     C        A, B      -        -         A, B    B
...
```

Read it as: *everything matched until step 1, where I planned no prefill and
should have planned `C:5`*. That is a much smaller question than "why is my
scheduler wrong".

## The two failure modes

**A `SchedulingError`** means your plan was *illegal* — it broke a rule the
engine enforces. The message names the rule:

```
step 1: plan uses 12 units of work but max_work is 10 (2 decode + 10 prefill)
```

Look up the matching test in `tests/test_engine_rules.py`; each one documents
one rule with a minimal example.

**An assertion failure** means your plan was legal but *not the one the spec
asks for*. Compare your trace against the table.

The most common `SchedulingError` by far is:

```
step 0: no decode or prefill scheduled, but requests are waiting or admitted
        — the schedule cannot make progress
```

which almost always means you wrote the decode loop and not yet the prefill
loop. With nothing admitted at step 0, a decode-only scheduler plans nothing
at all, forever.

## Printing your own trace

The fastest debugging tool here. Drop this in a scratch file:

```python
from scheduler import Arrival, Engine, render
from levels.level3b import plan_step

engine = Engine(
    [Arrival("A", 2, 2), Arrival("B", 4, 2), Arrival("C", 2, 2)],
    max_work=10,
    kv_capacity=6,
)
print(render(engine.run(plan_step), kv=True))
```

```
step  waiting  admitted  kv held  preempt  prefill   decode  finished
0     A, B, C  -         0        -        A:2, B:4  -       -
1     C        A, B      6        B        C:2       A       A
2     B        C         2        -        -         C       C
3     B        -         0        -        B:5       -       B
```

Put that next to the table in the level doc and the divergence is visually
obvious. `Arrival(id, prompt_len, max_tokens, priority=0, at=0)` — `at` is the
step it shows up on.

If your scheduler stalls, `run()` gives up after 100 steps rather than hanging:

```
SchedulingError: still running after 100 steps — is the schedule making progress?
```

## Writing an assertion for your own hypothesis

You do not need a test file. In the same scratch script:

```python
records = engine.run(plan_step)
assert records[1].preempt == ("B",), records[1]
```

`StepRecord` fields: `step`, `waiting`, `admitted`, `kv_held`, `preempt`,
`prefill`, `decode`, `finished`. Everything is a tuple of strings; `prefill`
entries are rendered `"ID:tokens"`.

## Drill

1. Run `./run 3a` with an empty `levels/level3a.py`. What error, and why that
   one specifically?
2. Make `levels/level1.py` schedule decodes only. Which test fails first, and
   is it a `SchedulingError` or an assertion?
3. Use `render()` to print the Level 2 example trace and find the step where
   X's prefill completes.

<details>
<summary>Answers</summary>

1. `SchedulingError: step 0: no decode or prefill scheduled...` — the stub
   returns an empty plan, and at step 0 there are three waiting requests, so
   the engine refuses a plan that cannot make progress.
2. Still a `SchedulingError`, and still at step 0 — with nothing admitted yet
   a decode-only scheduler plans nothing. You reach assertion failures only
   once the plan is legal.
3. Step 1 (`X:7` completes the 13-token prompt begun with `X:6` at step 0).

</details>
