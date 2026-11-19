# Prerequisite: the budget loop

Every level in this exercise is the same loop shape. Learn it once here and
each level becomes "the loop, plus one more condition".

## The shape

```python
budget = view.max_work            # 1. a local counter, seeded from the view
for request in view.waiting:      # 2. candidates in a defined order
    cost = request.prefill_remaining
    if budget < cost:             # 3. a fit test
        break                     # 4. stop (or clamp, or skip — see below)
    plan.prefill.append(...)      # 5. commit
    budget -= cost                # 6. pay for it
```

Six lines, five levels. What changes between levels is only:

- what goes in the **fit test** (work? memory? both?),
- what you do when it **fails** (stop? clamp? evict something and retry?),
- what **order** the candidates come in.

## `break` vs `continue` — the difference that costs marks

```python
for request in view.waiting:
    if budget < cost:
        break        # stop scheduling entirely
        continue     # skip this one, try the next
```

The spec is explicit, in every single level:

> Stop at the first waiting prefill that doesn't fit — don't skip ahead to a
> smaller, later one.

That is `break`. `continue` would be *better* for throughput — you would pack
the budget fuller — and it is wrong. This is a fairness rule: a large request
at the head of the queue must not be starved by an endless stream of small
ones behind it. In scheduling literature this is head-of-line blocking, and
here it is deliberate.

Both versions pass a naive smoke test. Only `break` passes the spec. When you
see "stop at the first X that doesn't fit", write `break` and move on.

## Clamping: `min(budget, need)`

Level 2 changes the failure mode from "stop" to "take what you can":

```python
chunk = min(budget, request.prefill_remaining)
plan.prefill.append(PrefillOp(id=request.id, tokens=chunk))
budget -= chunk
```

`min(budget, need)` is the whole of chunked prefill. Note what it does at the
boundaries: if `need` is small it takes all of it; if `budget` is small it
fills the budget exactly. The loop guard changes from "does it fit" to "is
there any budget left at all":

```python
for request in view.waiting:
    if budget <= 0:
        break
```

Use `<= 0`, not `== 0`. They are equivalent if your arithmetic is right, but
`<= 0` fails safe if it isn't, and it reads as "no budget left" rather than
"exactly zero budget left".

## Two budgets at once

From Level 3a there are two independent resources — work and memory — and a
candidate must satisfy both:

```python
if cost > work_budget or kv_used + cost > view.kv_capacity:
    break
```

Two rules for keeping this straight:

1. **Test before you commit, commit before you pay.** Never decrement first
   and check afterwards; you will end up "un-spending" and the bug will be
   invisible.
2. **Keep the two counters in the same direction.** In the reference solutions
   `work_budget` counts *down* from `max_work` and `kv_used` counts *up*
   towards `kv_capacity`, because that is how the spec phrases each one
   ("stays at most max_work units" vs "the slots in use"). Pick a direction
   per counter and never mix them mid-function.

## Partitioning before you loop

From Level 2, `view.admitted` contains two kinds of request and they need
different treatment:

```python
ready   = [r for r in view.admitted if r.prefill_remaining == 0]  # can decode
started = [r for r in view.admitted if r.prefill_remaining > 0]   # mid-prefill
```

This is worth doing even when you could test inside one loop, because the spec
states the priority order as three separate phases ("decodes; then already
started prefills; then new prefills"). Three phases in the spec, three loops in
the code — the mapping stays obvious and stays checkable.

A list comprehension over a tuple gives you a list, which you may then sort or
pop without touching the view.

## The snapshot rule, restated as a loop bug

The reason you need a local `budget` at all:

```python
# WRONG — there is no such thing
for request in view.waiting:
    if request.prefill_remaining <= view.max_work:   # max_work never changes
        plan.prefill.append(...)                     # so this admits everything
```

`view.max_work` is the *per-step* budget, not the *remaining* budget. It reads
10 on the first iteration and 10 on the fifth. The same applies to
`view.kv_capacity`, and to `r.kv` — none of them reflect what you have already
planned. If you find yourself reading a value off the view inside a loop and
expecting it to have changed, that is the bug.

## Refunding

Level 3c introduces the one case where a budget goes *up* mid-loop: evicting a
request cancels its decode, which gives back a unit of work and its memory.

```python
plan.decode.remove(victim.id)
work_budget += 1                       # the decode it would have done
kv_used -= victim.kv + 1               # its slots, plus the one it would grow into
```

The rule to remember: **whatever you added when you committed, subtract
exactly that when you undo.** Here the commit was `work_budget -= 1` and
`kv_used += r.kv + 1`, so the refund is the mirror image, `+ 1` included. Off
by one here is the single most common Level 3c bug.

## Drill

`drills/drills.py` Block C and D. Then:

1. Write the loop for "admit waiting requests while total work stays within
   `max_work`, stopping at the first that doesn't fit."
2. Change it to chunk instead of stopping.
3. `budget` is 3 and `prefill_remaining` is 10. What does each version plan?

<details>
<summary>Answers</summary>

1. ```python
   budget = view.max_work
   for r in view.waiting:
       if budget < r.prefill_remaining:
           break
       plan.prefill.append(PrefillOp(id=r.id, tokens=r.prefill_remaining))
       budget -= r.prefill_remaining
   ```
2. ```python
   budget = view.max_work
   for r in view.waiting:
       if budget <= 0:
           break
       chunk = min(budget, r.prefill_remaining)
       plan.prefill.append(PrefillOp(id=r.id, tokens=chunk))
       budget -= chunk
   ```
3. Version 1 plans nothing and stops — and with `max_work=3` and a 10-token
   prompt it will *never* schedule it, which is exactly the deadlock Level 2
   exists to fix. Version 2 plans `PrefillOp(id, 3)` and leaves 7 for later
   steps.

</details>
