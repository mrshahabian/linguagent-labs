# Level 3a — KV-aware admission

Write your answer in `levels/level3a.py`. Check with `./run 3a`.

## The spec

So far, nothing has modelled the *memory* a request occupies — admission has
only been bounded by the per-step `max_work`. In reality, the GPU needs memory
proportional to prompt length plus output length, to store the intermediate
results needed for attention. These intermediates are called the **KV cache**.

We'll model memory as a pool of "slots": `view.kv_capacity` is the total, and
each admitted request's `kv` field is the slots it currently holds. We assume
`kv_capacity` is always enough to complete any individual request on its own.

Each request requires one slot per prefilled input token plus one slot per
previously decoded output token, so a request's peak usage is on the last
step, when its final token is generated. Memory is held for the duration of a
step: a request that finishes this step still occupies (and grows) its slots
during the step, and they are reclaimed only afterwards.

The new logic:

- **Decodes first**, as before.
- **Prefill the oldest request as before, but admit it only if the peak usage
  of every admitted request plus the candidate — each counted at its peak, not
  its current usage — sums to at most `kv_capacity`.** `max_tokens` is an upper
  bound, not a prediction: a request may finish earlier and you can't know
  when, so you must plan for the worst case. (This is conservative, but
  guarantees every admitted request can run to completion.)
- **For fairness, stop at the oldest request that doesn't fit**; never skip
  ahead to newer requests.

In Level 3, we will **not** handle chunking; each prefill completes in one
step, as in Level 1.

## The peak formula

This is the one derivation in the level, and it is worth doing slowly.

A request holds one slot per **prefilled input token** — that is `prompt_len` —
plus one slot per **previously decoded output token**.

Walk a request with `prompt_len=2`, `max_tokens=3` through its life:

| step | event | tokens generated | slots held |
|---|---|---|---|
| 0 | prefill (2 tokens) → 1st output token | 1 | 2 |
| 1 | decode → 2nd output token | 2 | 3 |
| 2 | decode → 3rd output token | 3 | 4 |

The peak is 4, on the final step: `2 + 3 - 1`.

```
peak = prompt_len + max_tokens - 1
```

Why minus one? Because the *last* token generated does not need a slot of its
own to be produced — the slots hold the context that already exists. When the
final token is generated, the ones occupying memory are the prompt plus the
`max_tokens - 1` outputs that came before it.

Check against the worked example's requests: A(2,2) → 3. B(4,2) → 5.
C(2,2) → 3. Those are exactly the peaks the spec quotes.

*(Aside: the formula is invariant under preemption, which matters in 3b. A
request re-admitted with `k` tokens already generated has `prompt_len + k`
context and `max_tokens - k` tokens left, so its peak is still
`(prompt_len + k) + (max_tokens - k) - 1 = prompt_len + max_tokens - 1`.)*

## Worked example

`max_work=10`, `kv_capacity=6`; A(2 prompt, 2 max_tokens) → peak 3,
B(4, 2) → peak 5, C(2, 2) → peak 3.

```
step  waiting  admitted  kv held  prefill  decode  finished
0     A, B, C  -         0        A:2      -       -
1     B, C     A         2        -        A       A
2     B, C     -         0        B:4      -       -
3     C        B         4        -        B       B
4     C        -         0        C:2      -       -
5     -        C         2        -        C       C
```

Look at what this costs. **The pool is never more than 4/6 full, and only one
request runs at a time**, on a workload whose *actual* combined usage would
comfortably fit. That waste is the entire motivation for Level 3b.

Step 0 in detail: A's peak is 3, reserved is 0, `0 + 3 ≤ 6` → admit. B's peak
is 5, `3 + 5 = 8 > 6` → does not fit → **stop**. C is never considered, even
though its peak of 3 would have fitted in the remaining 3 slots. That is the
fairness rule doing its job.

## Deriving the code

**Decodes.** Unchanged from Level 1.

**The reservation.** Every admitted request was itself admitted at peak, so
the pool is reserved at peak:

```python
reserved = sum(peak_slots(r) for r in view.admitted)
```

Note this uses `peak_slots(r)`, **not** `r.kv`. Using `r.kv` is the classic
Level 3a bug: it is the *current* usage, so you under-count what is reserved
and over-admit. `r.kv` is what Level 3b wants; 3a wants the peak.

**The two-condition fit test.** Work *and* memory:

```python
for request in view.waiting:
    cost = request.prefill_remaining
    if cost > work_budget or reserved + peak_slots(request) > view.kv_capacity:
        break
    ...
    reserved += peak_slots(request)
```

## The answer

```python
def peak_slots(request) -> int:
    return request.prompt_len + request.max_tokens - 1


def plan_step(view: KvEngineView) -> StepPlan:
    plan = StepPlan()
    work_budget = view.max_work

    for request in view.admitted:
        plan.decode.append(request.id)
        work_budget -= 1

    reserved = sum(peak_slots(r) for r in view.admitted)

    for request in view.waiting:
        cost = request.prefill_remaining
        if cost > work_budget or reserved + peak_slots(request) > view.kv_capacity:
            break
        plan.prefill.append(PrefillOp(id=request.id, tokens=cost))
        work_budget -= cost
        reserved += peak_slots(request)

    return plan
```

Pulling `peak_slots` out as a named function is worth the three lines: it gets
used four times, and naming it means the `- 1` is written once instead of four
times.

## Pitfalls

- **`r.kv` instead of the peak.** Over-admits, then fails later with
  `SchedulingError: plan needs N KV slots but kv_capacity is M` — possibly
  several steps after the actual mistake, which makes it painful to find.
- **`prompt_len + max_tokens`** (no `- 1`). Under-admits; the trace is
  plausible but one request short.
- **Carrying chunking over from Level 2.** `min(budget, remaining)` here gives
  `SchedulingError: chunking is off — a prefill must cover the full remaining
  N tokens`.
- **`continue` on the memory condition.** Same fairness rule as always; the
  spec repeats "never skip ahead to newer requests" for exactly this reason.
- **Checking only one of the two budgets.** Both must hold. `or` between them,
  not `and`.

## Extra drills

1. **Derive it again.** `prompt_len=7`, `max_tokens=4`. Peak? Slots held after
   its prefill step? After its second decode?
2. **Exact fit.** `kv_capacity=4`, one request `prompt_len=2, max_tokens=3`.
   Does it admit? What is `kv_held` on each step? (Test:
   `test_level3a_peak_formula`.)
3. **The waste, measured.** Run the worked example under both `level3a` and
   `level3b` and compare step counts. What is the ratio? (Test:
   `test_level3b_beats_level3a_on_throughput`.)
4. **Theory.** The spec says "we assume `kv_capacity` is always enough to
   complete any individual request on its own." What would go wrong without
   that assumption, and how would you detect it?
5. **Theory.** Why is reserving at peak *safe* in a way that reserving at
   current usage is not? State the invariant it guarantees.

<details>
<summary>Answers</summary>

1. Peak = 7 + 4 − 1 = 10. After prefill: 7 slots (1 token generated). After
   its second decode: 9 slots (3 tokens generated, 7 + 2).
2. Yes: peak = 4 ≤ 4. `kv_held` goes 0 → 2 → 3 across the three steps, peaking
   at 4 *within* the final step before reclamation.
3. Six steps versus four — 3a takes 50% longer on the same workload with the
   same hardware.
4. A request whose peak exceeded `kv_capacity` could never be admitted by any
   scheduler, so it would wait forever and the engine would raise
   "no decode or prefill scheduled". You would detect it by checking
   `peak_slots(r) > kv_capacity` at admission and failing the request
   explicitly rather than starving it silently.
5. Invariant: **the sum of peaks of all admitted requests never exceeds
   `kv_capacity`**. Since each request's usage is bounded by its peak, actual
   usage can never exceed capacity either — so an admitted request can always
   take its next step, and no preemption mechanism is needed. Reserving at
   current usage gives no such bound: usage grows every step, so a set of
   requests that fits today may not fit tomorrow.

</details>
