# Level 1 — Basic scheduling

Write your answer in `levels/level1.py`. Check with `./run 1`.

## The spec

When serving inference, the two types of work the GPU can do are called
**prefill** and **decode**.

Prefill means the initial preprocessing of the input prompt and generation of
the first output token. For a prompt containing `prompt_len` tokens, this
requires `prompt_len` units of GPU work, all of which can be done in parallel —
in Level 1, every prefill completes within one time step.

Decode means generating additional output tokens after the first. Decode
reuses the intermediate results stored during prefill, so it only requires 1
unit of GPU work to produce 1 output token. A single request's decodes are
sequential — each output token depends on the previous one — so generating `n`
output tokens will require at least 1 time step for prefill and then `n-1`
more time steps for decode. (Different requests can each decode a token in the
same step.) Each request carries a `max_tokens` value — the most output tokens
it may generate.

A request starts in the `waiting` list, moves to `admitted` when its prefill is
scheduled, and finishes after its remaining decodes.

Your job is to look at which requests are `waiting` and `admitted`, and assign
prefill and decode work to the GPU.

The first scheduling algorithm we'll implement is:

- Schedule a decode for all admitted requests, oldest first (the order of
  `view.admitted`).
- Then, in arrival order, schedule each waiting prefill while the total work
  this step stays at most `view.max_work` units.
- Stop at the first waiting prefill that doesn't fit — don't skip ahead to a
  smaller, later one.

## Worked example

`max_work=10`, three requests arriving at step 0: A (6 prompt tokens,
max_tokens 3), B (3 prompt tokens, max_tokens 2), C (5 prompt tokens,
max_tokens 2).

```
step  waiting  admitted  prefill   decode  finished
0     A, B, C  -         A:6, B:3  -       -
1     C        A, B      C:5       A, B    B
2     -        A, C      -         A, C    A, C
```

- **Step 0.** Nothing is admitted, so no decodes. Prefill A costs 6; prefill B
  costs 3; that is 9 of the 10 units. C needs 5 more, which does not fit —
  stop.
- **Step 1.** Decode A (1 unit) and decode B (1 unit), then prefill C (5
  units) — 7 total, within budget. B's prefill in step 0 already produced the
  first of its two output tokens, so this decode produces the second and
  **finishes** it.
- **Step 2.** Decode A (its third token) and C (its second), completing both.

## Deriving the code

**Rule 1 — "a decode for all admitted requests, oldest first."**

`view.admitted` is documented as being in admission order, earliest first. So
"oldest first" is just the order it is already in. No sorting.

```python
for request in view.admitted:
    plan.decode.append(request.id)
    work += 1
```

Note there is no budget test on the decode loop. The spec does not gate
decodes on `max_work`, and by construction they always fit — the engine only
admits what it can keep decoding. Adding a test here is a common instinct and
a way to fail the tests.

**Rule 2 — "in arrival order, schedule each waiting prefill while total work
stays at most `max_work`."**

`view.waiting` is in arrival order already. "Total work" includes the decodes
you just planned, which is why `work` is seeded by the first loop rather than
starting at zero here.

```python
for request in view.waiting:
    cost = request.prefill_remaining
    if work + cost > view.max_work:
        ...
```

**Rule 3 — "stop at the first that doesn't fit — don't skip ahead."**

`break`. Not `continue`. See `03-prereq-budget-loops.md`; this is the rule
people lose marks on.

## The answer

```python
from scheduler.api import EngineView, PrefillOp, StepPlan


def plan_step(view: EngineView) -> StepPlan:
    plan = StepPlan()
    work = 0

    for request in view.admitted:
        plan.decode.append(request.id)
        work += 1

    for request in view.waiting:
        cost = request.prefill_remaining
        if work + cost > view.max_work:
            break
        plan.prefill.append(PrefillOp(id=request.id, tokens=cost))
        work += cost

    return plan
```

Fourteen lines. Every level from here is a variation on it.

## Why `prefill_remaining` and not `prompt_len`?

For a fresh waiting request they are equal, so both pass Level 1's tests.
Use `prefill_remaining` anyway:

- It is the field `PrefillOp.tokens` is actually checked against. The engine's
  rule is `tokens == prefill_remaining`, not `tokens == prompt_len`.
- From Level 3b onwards they *diverge*: a preempted request comes back with
  `prefill_remaining = prompt_len + tokens_already_generated`. Code written
  against `prompt_len` silently under-prefills and fails with a confusing
  error two levels later.

Getting into the habit now costs nothing and saves a debugging session.

## Pitfalls

- **`continue` instead of `break`.** Passes a casual read, fails the fairness
  test. `tests/test_worked_examples.py::test_level1_stops_rather_than_skipping_ahead`.
- **Forgetting decodes consume budget.** If `work` starts at 0 in the second
  loop, you will over-schedule and get
  `SchedulingError: plan uses N units of work but max_work is M`.
- **Expecting `view.max_work` to shrink.** It does not. Track spending
  yourself.
- **Thinking a request needs `max_tokens` decodes.** It needs `max_tokens - 1`;
  the prefill produced the first one. A request with `max_tokens=1` finishes on
  its prefill step and never decodes at all.
- **Sorting `view.admitted` or `view.waiting`.** Already in the right order,
  and they are tuples.

## Extra drills

1. **max_tokens=1.** One request, `prompt_len=3`, `max_tokens=1`,
   `max_work=10`. How many steps? What is in `decode`? Verify with `render()`.
2. **The deadlock.** One request with `prompt_len=25`, `max_work=10`. What
   happens, and what error do you get? This is the bug Level 2 exists to fix —
   see it for yourself before you fix it.
3. **Fairness.** A(4,1), B(8,1), C(1,1) with `max_work=10`. Which are
   prefilled at step 0 under `break`? Under `continue`? Which does the spec
   require?
4. **Budget interaction.** Four admitted requests and `max_work=4`. Can any
   prefill be scheduled? What does that imply about a server whose steady-state
   batch size approaches `max_work`?

<details>
<summary>Answers</summary>

1. One step. `decode` is empty — the prefill produces the only token, and the
   request finishes immediately.
2. Nothing is ever scheduled: `25 > 10` on every step, so the loop breaks
   immediately and the plan is empty. The engine raises
   `SchedulingError: no decode or prefill scheduled, but requests are waiting
   or admitted`. The request can never run.
3. `break` → only A (4 units; B needs 8, only 6 left, stop). `continue` → A and
   C (5 units). The spec requires `break`, so only A.
4. No — the four decodes consume the entire budget, so no new request is ever
   admitted until one finishes. This is real: decode-heavy steady state
   squeezes out prefill, which is why production schedulers reserve budget for
   prefill or interleave chunks of it (Level 2).

</details>
