# Level 3b — Preemption

Write your answer in `levels/level3b.py`. Check with `./run 3b`.

## The spec

Level 3a's peak reservation guaranteed progress and fairness, but at the cost
of throughput.

Now we'll admit a request based on its **current** memory requirement, and if
we reach a step where not all admitted requests can make progress, we will
**preempt**.

Preemption means choosing an admitted request (the **victim**) and including
it in `plan.preempt`, causing the engine to automatically:

- Discard its KV cache, immediately freeing its memory (which can be used this
  step).
- Save any already-generated output tokens; these count towards
  `prefill_remaining`.
- Return the victim to `waiting` at a position determined by its original
  arrival (not at the back).

The full algorithm is:

- While there aren't enough memory slots for all decodes to make progress,
  preempt the **most recently admitted** request (append its id to
  `plan.preempt`).
- Decode all remaining requests (now guaranteed to fit).
- Then admit waiting prefills, oldest first, while the work budget and the free
  slots allow — counting the slot each scheduled decode grows into this step —
  and stop at the first that doesn't fit.

## Worked example

Same requests as 3a — `max_work=10`, `kv_capacity=6`, A(2,2), B(4,2), C(2,2) —
but now finishing at step 3 instead of step 5.

```
step  waiting  admitted  kv held  preempt  prefill   decode  finished
0     A, B, C  -         0        -        A:2, B:4  -       -
1     C        A, B      6        B        C:2       A       A
2     B        C         2        -        -         C       C
3     B        -         0        -        B:5       -       B
```

**Step 0.** No decodes. Admission is now against *current* usage: A costs 2
slots (`0 + 2 ≤ 6`), B costs 4 (`2 + 4 = 6 ≤ 6`, exactly full). C's 2 do not
fit. Compare with 3a, where B's *peak* of 5 blocked it entirely — that is the
throughput win, and it is also the risk we just took on.

**Step 1.** The risk lands. A holds 2 slots and B holds 4; if both decode they
need 3 and 5 = 8 slots, but there are only 6. So we preempt the most recently
admitted — B — freeing its 4 slots immediately. A decodes (2 → 3 slots) and
finishes. With 3 slots free, C's prefill fits.

**Step 2.** Only C is admitted. It decodes and finishes. B is back in
`waiting`, but its `prefill_remaining` is now **5**, not 4, and `3 + 5 > 6`,
so it does not fit yet.

**Step 3.** Nothing admitted, so all 6 slots are free. `B:5` runs — and since
B had already generated 1 token before being preempted, this prefill produces
its second, finishing it.

## The three ideas

### 1. `r.kv + 1` — the slot a decode grows into

This is the whole level. A decoding request needs one *more* slot than it
holds right now:

```python
sum(r.kv + 1 for r in survivors) > view.kv_capacity
```

Forget the `+ 1` and you never preempt, then fail with
`SchedulingError: plan needs 8 KV slots but kv_capacity is 6`.

The same `+ 1` appears when you account for the decodes before admitting:

```python
kv_used += request.kv + 1
```

### 2. The victim is the last element of `view.admitted`

`view.admitted` is in admission order, earliest first. So "most recently
admitted" is the last element. Copy to a list first — the view is a tuple, and
you must not mutate it:

```python
survivors = list(view.admitted)
victim = survivors.pop()
```

No sorting, no key function. Just `pop()`.

### 3. A victim does no work this step

It is not in `plan.decode`, and its memory is free *immediately* — usable by
this same step's prefills. That is why step 1 can preempt B and admit C in one
step. The engine enforces the first half:

```
SchedulingError: 'B' is in both preempt and decode — a victim does no work on
the step it is evicted
```

The clean way to get this right is to compute the survivor set **first**, then
build the decode list from it — rather than decoding everything and removing
victims afterwards.

## Deriving the code

The spec's three bullets, in order:

```python
# 1. "While there aren't enough memory slots for all decodes to make
#     progress, preempt the most recently admitted request."
survivors = list(view.admitted)
while survivors and sum(r.kv + 1 for r in survivors) > view.kv_capacity:
    victim = survivors.pop()
    plan.preempt.append(victim.id)

# 2. "Decode all remaining requests (now guaranteed to fit)."
work_budget = view.max_work
kv_used = 0
for request in survivors:
    plan.decode.append(request.id)
    work_budget -= 1
    kv_used += request.kv + 1

# 3. "Then admit waiting prefills, oldest first, while the work budget and the
#     free slots allow ... and stop at the first that doesn't fit."
for request in view.waiting:
    cost = request.prefill_remaining
    if cost > work_budget or kv_used + cost > view.kv_capacity:
        break
    plan.prefill.append(PrefillOp(id=request.id, tokens=cost))
    work_budget -= cost
    kv_used += cost
```

`while`, not `if` — one eviction may not be enough. The `survivors and` guard
stops an infinite loop if nothing is admitted.

`kv_used` starts at **0**, not at the current pool usage, because the decode
loop adds `r.kv + 1` for every survivor — which already accounts for all the
memory still held. Preempted victims are excluded, which is precisely how
their freed slots become available.

## Why `prefill_remaining` finally matters

A re-admitted victim has `prefill_remaining = prompt_len + tokens_generated`.
In the trace, B comes back needing **5** tokens (4 prompt + 1 generated), not
4. If you wrote `cost = request.prompt_len` back in Level 1, this is where it
breaks — you would plan `B:4` and get:

```
SchedulingError: prefill of 'B' is 4 tokens but chunking is off — a prefill
must cover the full remaining 5 tokens
```

That re-prefilled token is *wasted work*: it was computed once, thrown away,
and computed again. That is the real price of optimistic admission, and it is
why a good scheduler preempts as rarely as it can.

## Pitfalls

- **Missing `+ 1`.** Never preempts; overflows memory.
- **`if` instead of `while`.** One eviction is not always enough.
- **Popping the *first* element.** That is the oldest, not the most recent —
  exactly backwards, and it evicts the request that has done the most work.
- **Decoding victims.** Compute survivors first.
- **Seeding `kv_used` with the current pool usage** and then also adding
  `r.kv + 1` per survivor. Double-counting; nothing is ever admitted.
- **Using `prompt_len` as the prefill cost.** Breaks on the re-admitted victim.
- **Trying to be clever about *which* victim.** The spec says most recently
  admitted. Not smallest, not the one holding the most memory.

## Extra drills

1. **The `+ 1`, seen.** Delete `+ 1` from the `while` condition and run
   `./run 3b`. Which step fails and what is the error?
2. **Multiple evictions.** Construct a case where one eviction is not enough
   and `while` is required. (Hint: three admitted requests, tight capacity.)
3. **Wasted work, counted.** In the worked example, how many prefill tokens are
   processed in total? How many would 3a have processed? Where is the
   difference?
4. **Queue position.** Why must a victim rejoin at its arrival position rather
   than the back of the queue? Construct the starvation scenario that the
   other choice would allow.
5. **Theory.** 3b can preempt a request the very step after admitting it,
   wasting all its prefill work. Suggest a rule that would reduce that, and
   name what it costs.

<details>
<summary>Answers</summary>

1. Step 1: `SchedulingError: plan needs 8 KV slots but kv_capacity is 6` —
   A needs 3 and B needs 5 after their decodes.
2. E.g. `kv_capacity=6` with three admitted requests each holding 2 slots:
   decoding all three needs 9. Evicting one leaves 6 needed for two — still
   over 6? No: 3 + 3 = 6, which fits. So make it four requests holding 2 each
   (needs 12); evicting one leaves 9, still over; evicting a second leaves 6.
   Two evictions, so `while` is required.
3. 3b processes 2 + 4 + 2 + 5 = 13 tokens; 3a processes 2 + 4 + 2 = 8. The
   extra 5 is B's re-prefill after preemption. 3b trades 5 tokens of redundant
   work for finishing two steps sooner.
4. If victims went to the back, a request could be admitted, evicted, sent to
   the back, and repeatedly overtaken by newer arrivals — starving forever
   while doing (and losing) work each time. Rejoining at its arrival position
   bounds its wait by the requests that genuinely arrived before it.
5. Anything that raises the bar for admission — reserving a margin above
   current usage, requiring a request to survive *n* steps before it can be a
   victim, or preferring victims that have done less work. All of them cost
   utilisation: you hold back memory or keep a worse candidate running. This
   is the same peak-vs-current trade-off as 3a, just with a dial instead of a
   switch.

</details>
