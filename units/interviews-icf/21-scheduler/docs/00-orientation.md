# Orientation

## What this exercise is

You are writing the scheduler for an LLM inference server. Each step, the
engine hands you a read-only snapshot of the world and asks one question:

> Given what is waiting, what is running, and how much GPU and memory you
> have — what should the GPU do *this* step?

You answer with a `StepPlan`. The engine applies it, advances one step, and
asks again. That is the whole job.

The exercise is delivered as five levels that each add one constraint:

| Level | Adds | The one new idea |
|---|---|---|
| 1 | Basic scheduling | a work budget, and stopping (not skipping) when it runs out |
| 2 | Chunked prefill | a prefill can be split across steps |
| 3a | KV-aware admission | memory, reserved pessimistically at peak |
| 3b | Preemption | memory reserved optimistically, evicting when you guess wrong |
| 3c | Priority tiers | some requests outrank others, for both admission and eviction |

The levels are cumulative in *understanding* but each lives in its own file.
You do not refactor Level 1 into Level 3c; you write five separate
`plan_step` functions, each solving a strictly harder problem.

## Why this shape shows up in interviews

This is a CodeSignal "Industry Coding Framework" task. The framework is
deliberately not an algorithms puzzle. There is no clever trick, no hidden
`O(n log n)` insight. Every level is a loop over two lists with a budget
counter. What is being measured is:

- **Can you read a precise specification and implement exactly it?** Not
  approximately it. Not a reasonable interpretation of it.
- **Can you keep working when requirement 6 contradicts your design from
  requirement 2?**
- **Do you notice the difference between "stop" and "skip"?** Between "peak"
  and "current"? Between "lower" and "strictly lower"?

Nearly every failure on tasks like this is a reading failure, not a coding
failure. The single highest-value habit this course tries to build is: **turn
each sentence of the spec into one line of code, and know which line.**

## How to use this repo

```
scheduler/        the engine and the types. Read api.py; engine.py is optional
                  but rewarding once you've finished.
levels/           YOU WRITE HERE. Five stubs waiting for a plan_step.
solutions/        reference implementations, each heavily commented.
tests/            the worked examples from the spec, as executable tests.
drills/           short exercises for the sub-skills, run with ./drills/drill
docs/             this course.
```

Commands:

```
./run             run every level test against your code in levels/
./run 1           just Level 1
./run 3b          just Level 3b
./run rules       the engine-rule tests (these pass without any of your code —
                  they document what the engine rejects)
./run --check     run the same tests against solutions/ to prove they pass
```

## The recommended path

If you are comfortable with Python but new to this problem, work in this
order. The prerequisites are short and every one of them is used directly.

1. **`01-prereq-dataclasses.md`** — what `@dataclass(frozen=True)` gives you
   and why the API is built from them.
2. **`02-prereq-sorting.md`** — `sorted(key=...)`, tuple keys, stability.
   Level 3c is unsolvable without this.
3. **`03-prereq-budget-loops.md`** — the accumulate-and-stop loop shape that
   every single level is built from.
4. **`04-prereq-pytest.md`** — reading a failure, running one test.
5. **`05-domain-llm-inference.md`** — what prefill, decode and the KV cache
   actually are. You can solve the levels without this, but you will solve
   them faster with it, because the rules stop being arbitrary.
6. **`06-reading-the-spec.md`** — the technique. Read this before Level 1 and
   again before Level 3c.
7. Then `10-level1.md` through `14-level3c.md`, one at a time. For each:
   read the doc, write the code in `levels/`, run `./run <level>`, and only
   look at `solutions/` once you are green — or once you have been stuck for
   fifteen minutes.
8. **`20-common-bugs.md`** — read after Level 2, and again at the end.
9. **`21-strategy.md`** — how to spend 90 minutes under time pressure.

## The one thing to internalise first

The view is a **snapshot**. It is taken before your `plan_step` runs and it
does not change as you build the plan.

```python
def plan_step(view):
    plan = StepPlan()
    plan.prefill.append(PrefillOp("A", 5))
    # view.max_work is STILL 10. Nothing decremented it. There is no
    # view.remaining_work. Tracking what you have spent is your job.
```

Every level therefore has the same skeleton: a local counter that starts at
the budget, and gets decremented by hand as you commit to work.

```python
budget = view.max_work
...
budget -= cost
```

If you remember one thing from this course, remember that the engine will not
stop you from planning 400 units of work — it will just reject the whole plan
afterwards, and you will be debugging a `SchedulingError` instead of a bug.
