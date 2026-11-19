# Level 3c — Priority tiers

Write your answer in `levels/level3c.py`. Check with `./run 3c`.

This is the hardest level. It has five interacting rules and the worked
example exercises four of them in a single row. Budget more time for reading
than for typing.

## The spec

Now that eviction is implemented, we can naturally handle requests of differing
`priority`, where a **larger number = more urgent**.

- **Victim selection:** always chooses the **lowest-priority** request, ties
  broken by **most recently admitted**, as before.
- **Preemption for growth:** as in Level 3b, if there aren't enough free slots
  for every admitted request to decode this step, preempt (in the victim order
  above) until they fit — **before** decoding.
- **Decodes:** unchanged — in the order of `view.admitted`. (That's admission
  order, which under priority admission is no longer always arrival order.)
- **Admission order:** admit **highest priority first, tiebreaker by oldest**.
  (`view.waiting` is already in arrival order, so a stable sort by descending
  `priority` keeps oldest-first within a tier.) As before, stop at the first
  candidate — in this order — that can't be admitted.
- **Preemption for admission:** if the candidate's prefill fits the remaining
  budget but there isn't enough free memory, evict admitted requests of
  **strictly lower priority** than the candidate (victim order above) as needed
  to make room. Only a *memory* shortfall triggers eviction; a candidate that
  doesn't fit the *budget* stops admission as usual. **Only do this if it
  actually suffices:** if evicting every strictly-lower-priority request still
  wouldn't free enough memory, evict no one and stop admitting. A victim does
  no decode this step — a plan that both preempts and decodes the same request
  is rejected — so each eviction frees its `kv` slots **plus** the one slot its
  decode would have grown into, and **refunds** the one unit of work budget its
  decode would have used (which may let further candidates be admitted this
  step).

Append to `plan.preempt` in the order you evict.

## Worked example

`max_work=10`, `kv_capacity=8`. LOW (prompt 2, max_tokens 3, priority 0),
MID (prompt 2, max_tokens 3, priority 1), HI (arrival step 2, prompt 4,
max_tokens 1, priority 2). LOW is listed first, so it is ahead of MID in
arrival order.

```
step  waiting   admitted  kv held  preempt  prefill       decode    finished
0     LOW, MID  -         0        -        MID:2, LOW:2  -         -
1     -         MID, LOW  4        -        -             MID, LOW  -
2     HI        MID, LOW  6        LOW      HI:4          MID       MID, HI
3     LOW       -         0        -        LOW:4         -         LOW
```

**Step 0.** MID is prefilled **before** LOW even though LOW is ahead in queue
order — priority beats arrival. Note the resulting admission order is
`MID, LOW`, which is *not* arrival order; that is what the "decodes: unchanged,
in the order of `view.admitted`" note is warning you about.

**Step 1.** Both decode. `(2+1) + (2+1) = 6 ≤ 8`, so no preemption.

**Step 2.** HI arrives. This row is the whole level:

- Growth check first: `(3+1) + (3+1) = 8 ≤ 8` — fits, so **no** preemption for
  growth. Both would decode.
- Admission: HI (priority 2) is the only candidate. Cost 4 ≤ budget 8 ✓. But
  free memory is `8 - 8 = 0`, so the shortfall is 4.
- Evictable pool = strictly lower priority than 2 = {LOW (p0), MID (p1)},
  in victim order **LOW first** (lowest priority).
- Would evicting everything suffice? `(3+1) + (3+1) = 8 ≥ 4` ✓ — proceed.
- Evict LOW: frees `3 + 1 = 4`, which meets the shortfall exactly. **Stop.**
  MID is spared even though it is also lower priority than HI.
- LOW is removed from `plan.decode`, refunding 1 unit of work.
- Prefill `HI:4`. HI has `max_tokens=1`, so the prefill produces its only token
  and it finishes immediately. MID's decode is its third token, so it finishes
  too.

**Step 3.** LOW is back in `waiting` with `prefill_remaining = 2 + 2 = 4` (its
2-token prompt plus the 2 tokens it had generated). Nothing else is admitted,
so it prefills, produces its third token, and finishes.

## The four sub-problems

### 1. The victim key

Two criteria: lowest priority, then most recently admitted. Recency is
position in `view.admitted`, so build an index once:

```python
rank = {request.id: index for index, request in enumerate(view.admitted)}

def victim_key(request):
    return (request.priority, -rank[request.id])
```

`min(pool, key=victim_key)` gives the victim: lowest `priority` wins; among
equals, the **largest** rank (most recent) wins because it is negated.

Why a dict rather than `.index()`? Because you will be removing requests from
a local `active` list as you evict, and `.index()` on that list would return
shifting positions. `rank` is computed once from the immutable view and stays
correct.

### 2. Admission order — one stable sort

```python
for candidate in sorted(view.waiting, key=lambda r: -r.priority):
```

That is all. `view.waiting` is in arrival order and Python's sort is stable, so
oldest-first inside each priority tier comes free. See
`02-prereq-sorting.md` if this is not obvious — it is worth being sure about.

### 3. The invariant that makes the accounting work

After the growth-preemption loop and the decode loop:

> **every request in `active` has a decode scheduled.**

So when you evict one for admission, you know exactly what it gives back:
`kv + 1` slots (its own, plus the one its decode would have grown into) and
1 unit of work. Hold that invariant in your head and the refund arithmetic
writes itself.

Newly admitted candidates are deliberately *not* added to `active` — they have
no decode, and since candidates are processed in descending priority order, no
later candidate could have strictly higher priority than them anyway. They can
never be victims.

### 4. The all-or-nothing check

> Only do this if it actually suffices.

Check the whole pool *before* evicting anyone:

```python
if sum(r.kv + 1 for r in pool) < shortfall:
    break            # evict no one, stop admitting
```

Without this you evict a few requests, still cannot fit the candidate, and
have destroyed work for nothing. This is the rule that is easiest to skip on a
first read and it has its own test
(`test_level3c_all_or_nothing_eviction`).

## Deriving the code

Order matters; the spec's phases are:

```python
# 1. preempt for growth (lowest priority first) — BEFORE decoding
active = list(view.admitted)
while active and sum(r.kv + 1 for r in active) > view.kv_capacity:
    victim = min(active, key=victim_key)
    active.remove(victim)
    plan.preempt.append(victim.id)

# 2. decode survivors, in view.admitted order
survivors = {r.id for r in active}
work_budget, kv_used = view.max_work, 0
for request in view.admitted:
    if request.id in survivors:
        plan.decode.append(request.id)
        work_budget -= 1
        kv_used += request.kv + 1

# 3. admit, highest priority first
for candidate in sorted(view.waiting, key=lambda r: -r.priority):
    cost = candidate.prefill_remaining

    if cost > work_budget:
        break                          # budget shortfall: eviction can't help

    shortfall = cost - (view.kv_capacity - kv_used)
    if shortfall > 0:
        pool = sorted(
            (r for r in active if r.priority < candidate.priority),
            key=victim_key,
        )
        if sum(r.kv + 1 for r in pool) < shortfall:
            break                      # all-or-nothing

        freed = 0
        for victim in pool:
            if freed >= shortfall:
                break
            plan.decode.remove(victim.id)
            plan.preempt.append(victim.id)
            active.remove(victim)
            work_budget += 1
            freed += victim.kv + 1
            kv_used -= victim.kv + 1

    plan.prefill.append(PrefillOp(id=candidate.id, tokens=cost))
    work_budget -= cost
    kv_used += cost
```

Two subtleties in step 3 worth naming:

- **The budget check comes before the eviction logic.** The spec: "Only a
  memory shortfall triggers eviction; a candidate that doesn't fit the budget
  stops admission as usual." Eviction buys memory, never work.
- **The refund helps *later* candidates, not this one.** "which may let further
  candidates be admitted this step". So we check `cost > work_budget` against
  the pre-refund budget, and the `+= 1` benefits the next iteration.

Note step 2 iterates `view.admitted` and filters, rather than iterating
`active` — because `active` has had victims removed by `min()`, which does not
preserve order. The spec says decodes go in `view.admitted` order.

## Pitfalls

- **`<=` instead of `<`** in the evictable pool. "Strictly lower priority."
  Evicting a peer fails `test_level3c_will_not_evict_equal_or_higher_priority`.
- **Forgetting to remove the victim from `plan.decode`.**
  `SchedulingError: in both preempt and decode`.
- **Forgetting the work refund**, or refunding the wrong amount. Mirror the
  commit exactly.
- **Forgetting `+ 1` in `freed`.** You evict more requests than necessary and
  MID does not get spared.
- **Skipping the all-or-nothing check.** Thrashing.
- **`sorted(..., reverse=True)` with a tuple key.** Reverses the tiebreaker.
- **Decoding in `active` order** instead of `view.admitted` order.
- **`max` instead of `min`** for the victim, or forgetting to negate the rank.

## Extra drills

1. **Victim key, from memory.** Write it, then explain why the rank is
   negated.
2. **Spare MID.** Trace step 2 of the worked example assuming eviction did
   *not* stop once the shortfall was met. Which cell of the table changes?
3. **All-or-nothing.** Build a case where the pool cannot cover the shortfall
   and confirm nothing is evicted. (See
   `test_level3c_all_or_nothing_eviction` for one.)
4. **Refund.** Construct a case where evicting one request refunds enough work
   budget to admit a *second* candidate in the same step.
5. **Theory.** Priority inversion: can a low-priority request block a
   high-priority one indefinitely here? What in the rules prevents it, and what
   does not?
6. **Theory.** This scheduler has no notion of a request's age when choosing
   victims, so a low-priority request can be evicted repeatedly. Sketch a rule
   that bounds this, and say what it costs.

<details>
<summary>Answers</summary>

1. `(r.priority, -rank[r.id])`. `min` picks the smallest, so `priority`
   ascending gives the lowest-priority victim. For the tiebreak we want the
   *largest* rank (most recently admitted) to be picked, and negating turns
   "largest" into "smallest".
2. `preempt` would be `LOW, MID`, `decode` would be empty, and `finished`
   would lose MID — it would be sent back to waiting instead of completing.
3. See the test: HUGE needs 10 slots, only 2 free; PEER holds 5 but is equal
   priority so untouchable; evicting LOW frees only 4 of the 8 needed, so
   nothing is evicted and HUGE waits.
4. Take a tight `max_work` where the decodes consume nearly all of it: e.g.
   `max_work` exactly equal to the number of admitted requests, so
   `work_budget` is 0 after decoding. Then no candidate can be admitted at all
   until an eviction refunds a unit. The first candidate's eviction frees both
   memory and one work unit, which the second candidate can then spend.
5. Not indefinitely for *admission*: a high-priority candidate can evict
   strictly-lower-priority holders to make room, and is always considered
   first. But it **can** be blocked by equal-or-higher-priority requests
   holding memory, since those are never evictable — and by a work-budget
   shortfall, which eviction cannot fix. So inversion is bounded by tier, not
   eliminated.
6. Anything that adds a cost to re-evicting the same victim: an eviction
   counter in the victim key, an age/starvation boost that raises effective
   priority the longer a request waits, or a minimum residency before a
   request becomes evictable. All of them cost throughput, because you are
   choosing a worse victim (one holding less memory, or that has done more
   work) to protect fairness. This is the classic aging mechanism from
   OS process schedulers.

</details>
