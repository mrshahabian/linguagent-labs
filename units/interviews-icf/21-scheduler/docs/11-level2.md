# Level 2 — Chunked prefill

Write your answer in `levels/level2.py`. Check with `./run 2`.

## The problem with Level 1

Level 1 could not handle a prompt longer than `max_work`. A 25-token prompt on
a 10-unit budget never fits, so it is never scheduled, so it waits forever.
Run drill 2 from the Level 1 doc if you have not seen it happen.

There is a second, subtler problem. Even when a huge prefill *does* fit, it
consumes the whole budget for that step — so every request that was happily
decoding stalls. One new arrival with a long prompt freezes everyone else's
token stream.

## The spec

Our Level 1 logic was unable to handle prompts longer than `max_work` tokens.
To fix this, we will split prefill work across multiple time steps. The
priority order is now:

- **Decodes** (still oldest first; this avoids having a large prefill stall
  other requests for many steps).
- **Already started prefills** (we prefer to complete a partial prefill over
  starting a new prefill).
- Then **start waiting prefills**, oldest first, while the budget allows. A
  prefill too big for the remaining budget is chunked to fill it exactly.

## Worked example

`max_work=8`; W (prompt 2, max_tokens 4), X (prompt 13, max_tokens 2),
Y (prompt 3, max_tokens 2), all arriving at step 0.

```
step  waiting  admitted  prefill   decode  finished
0     W, X, Y  -         W:2, X:6  -       -
1     Y        W, X      X:7       W       -
2     Y        W, X      Y:3       W, X    X
3     -        W, Y      -         W, Y    W, Y
```

- **Step 0.** No decodes. W's prefill takes 2, leaving 6. X needs 13 — more
  than the 6 left — so it is **chunked to fill the budget exactly**: `X:6`.
  Budget exhausted, Y waits.
- **Step 1.** W decodes (1 unit). X is admitted but only partially prefilled,
  with 7 tokens outstanding; the remaining 7 units of budget cover exactly
  that, so `X:7` completes it. Nothing left for Y.
- **Step 2.** W and X both decode (X's prefill completed last step, producing
  its first token; this decode produces its second and finishes it). 6 units
  remain, so Y's 3-token prefill starts and completes.
- **Step 3.** W and Y decode, finishing both.

Note the fairness point the spec calls out: at step 1 we *could* have started
and fully completed Y's small 3-token prefill instead of topping up X. We
finish the older prefill X first.

## What is genuinely new

**Admitted no longer means "decoding".** In Level 1, everything in
`view.admitted` was fully prefilled. Now `view.admitted` holds two kinds of
request and they need opposite treatment:

```python
ready   = [r for r in view.admitted if r.prefill_remaining == 0]  # decode these
started = [r for r in view.admitted if r.prefill_remaining > 0]   # prefill these
```

This is the change that breaks a copied Level 1 solution. If you decode
everything in `view.admitted`, the engine rejects it:

```
SchedulingError: cannot decode 'X' — it still has 7 prefill tokens outstanding
```

**Chunking.** `min(budget, remaining)` instead of "does it fit".

## Deriving the code

Three bullets in the spec, three loops, in that order. Because the phases are
strictly ordered by priority, a countdown budget is cleaner here than the
running total of Level 1:

```python
budget = view.max_work
```

**Phase 1 — decodes, oldest first.**

```python
for request in ready:
    plan.decode.append(request.id)
    budget -= 1
```

**Phase 2 — already-started prefills.** These come before new ones. They are
in `view.admitted` order, which is admission order — and admission order is
arrival order at this level, so "oldest first" needs no sorting.

```python
for request in started:
    if budget <= 0:
        break
    chunk = min(budget, request.prefill_remaining)
    plan.prefill.append(PrefillOp(id=request.id, tokens=chunk))
    budget -= chunk
```

**Phase 3 — new prefills from the queue.** Identical body, different source
list.

```python
for request in view.waiting:
    if budget <= 0:
        break
    chunk = min(budget, request.prefill_remaining)
    plan.prefill.append(PrefillOp(id=request.id, tokens=chunk))
    budget -= chunk
```

The guard is now `budget <= 0` rather than a fit test: with chunking, *any*
remaining budget can be used, so the only reason to stop is having none left.

## The answer

See `solutions/level2.py`. It is exactly the three loops above.

## Should phases 2 and 3 be one loop?

You could concatenate `started + list(view.waiting)` and write one loop. The
reference keeps them separate for a reason worth internalising: **the spec
lists them as separate priority tiers, and later levels treat them
differently.** Code that mirrors the spec's structure stays checkable against
the spec. When Level 3 removes chunking entirely, you delete phase 2 rather
than untangling a fused loop.

## Pitfalls

- **Decoding a partially-prefilled request.** The number one Level 2 error.
  Split `view.admitted` on `prefill_remaining`.
- **Starting new prefills before topping up started ones.** Passes the simple
  cases, fails `test_level2_finishes_a_started_prefill_before_starting_a_new_one`.
- **Using `prompt_len` for the chunk size.** For a half-prefilled request that
  over-asks and the engine rejects it: `prefill of 'X' is 13 tokens but only 7
  remain`.
- **`if budget == 0`.** Use `<= 0`. Equivalent when correct, safer when not.
- **Emitting a zero-token `PrefillOp`.** If `budget` hits 0 and you do not
  guard, `min(0, n)` is 0 and the engine rejects
  `prefill of 'Y' must be at least 1 token`.
- **Carrying chunking into Level 3.** Chunking is Level 2 *only*. In 3a/3b/3c
  a prefill must cover the full remaining context.

## Extra drills

1. **The fix, verified.** `prompt_len=25`, `max_work=4`, `max_tokens=1`. How
   many steps? Print the trace.
2. **Exact fill.** `max_work=8`, one admitted request decoding, one waiting
   with a 20-token prompt. What is the chunk size at each step, and why not 8?
3. **Ordering.** Two admitted requests mid-prefill (5 and 3 outstanding) and
   one waiting (2), `max_work=6`. Write the plan by hand, then check.
4. **Theory.** Why does the spec put decodes *first*, given that this makes
   long prompts take longer to admit? Relate it to TTFT and TPOT from
   `05-domain-llm-inference.md`.
5. **Design.** Real servers cap the prefill chunk (e.g. 512 tokens) rather than
   letting it take the whole budget. What does that buy, and what does it cost?

<details>
<summary>Answers</summary>

1. Seven steps: `ceil(25/4) = 7` prefill steps (6 full chunks of 4 and one of
   1), and the last one produces the only output token, so no decode.
2. 7, then 8, then 8… — the first step is 7 because the decode consumed 1 unit
   of the 8. The budget is shared between decode and prefill.
3. Decodes first? No — both admitted requests are mid-prefill, so `ready` is
   empty. Phase 2: first started takes `min(6,5)=5`, budget 1; second takes
   `min(1,3)=1`, budget 0. Phase 3: budget is 0, so the waiting request gets
   nothing.
4. Decodes are cheap (1 unit) and directly visible to a user who is already
   streaming tokens — starving them spikes TPOT/inter-token latency, which
   reads as the response "freezing". Prefill only affects TTFT for a request
   nobody is watching yet. Serving the cheap, latency-critical work first is
   the right trade.
5. A cap bounds how much of any single step prefill can take, which bounds the
   worst-case inter-token latency for everyone decoding. It costs TTFT for
   long prompts (more steps to admit) and some GPU efficiency (smaller matrix
   multiplies).

</details>
