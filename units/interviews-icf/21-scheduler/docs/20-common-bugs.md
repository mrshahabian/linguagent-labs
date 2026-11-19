# The bug catalogue

Symptom-first. When something fails, find the message here.

## `SchedulingError` messages

### `no decode or prefill scheduled, but requests are waiting or admitted`

Your plan came back empty while there was work to do.

- **At step 0, always:** you wrote the decode loop but not the prefill loop.
  Nothing is admitted at step 0, so a decode-only scheduler plans nothing,
  forever. This is the first error nearly everyone hits.
- **At a later step:** every waiting request failed your fit test and every
  admitted request was preempted or skipped. Usually a budget you decremented
  twice, or a memory check that double-counts (see below).
- **On Level 1 with a long prompt:** genuinely unschedulable —
  `prompt_len > max_work` and there is no chunking. That is the deadlock
  Level 2 exists to fix.

### `plan uses N units of work but max_work is M`

- Decodes not counted against the same budget as prefills.
- `view.max_work` read inside the loop as if it were the *remaining* budget.
  It is the per-step total and never changes.
- Level 3c: you forgot the eviction refund, or refunded after spending.

### `plan needs N KV slots but kv_capacity is M`

- **Missing `+ 1`** on decoding requests. A decode grows its request by one
  slot; that is the most common cause.
- Level 3a: you used `r.kv` (current) where the spec says *peak*.
- You forgot that a request finishing this step still holds — and grows — its
  slots. Memory is reclaimed *after* the step.
- You assumed a preempted victim's memory frees next step. It frees
  immediately, this step. (That one causes over-conservatism, not this error —
  but it is the same misunderstanding.)

### `cannot decode 'X' — it still has N prefill tokens outstanding`

Level 2 only. `view.admitted` now contains partially-prefilled requests.
Split it:

```python
ready   = [r for r in view.admitted if r.prefill_remaining == 0]
started = [r for r in view.admitted if r.prefill_remaining > 0]
```

### `prefill of 'X' is N tokens but chunking is off`

You carried `min(budget, remaining)` from Level 2 into Level 3. Chunking is
Level 2 only; everywhere else a prefill must cover the full remaining context.

### `prefill of 'X' is N tokens but only M remain`

You used `prompt_len` as the cost for a request that has been partially
prefilled (Level 2) or preempted and re-queued (Level 3b/3c, where
`prefill_remaining = prompt_len + tokens_generated`). Always use
`request.prefill_remaining`.

### `'X' is in both preempt and decode`

A victim does no work on the step it is evicted. Compute the survivor set
first and build `plan.decode` from *that*, or remove the victim from
`plan.decode` when you evict it for admission (Level 3c).

### `cannot preempt 'X' — only admitted requests can be preempted`

You tried to evict something from `view.waiting`. Only admitted requests hold
memory.

### `prefill of 'X' must be at least 1 token`

`min(budget, remaining)` with `budget == 0`. Guard the loop with
`if budget <= 0: break`.

### `still running after 100 steps`

Livelock. Almost always Level 3c thrashing: evicting a request, admitting a
candidate, then evicting the candidate next step to re-admit the first. Check
the all-or-nothing rule and that you evict only as many as needed.

## Silent wrong answers

These produce a legal plan that is not the one the spec wants, so you get an
assertion failure and a trace comparison rather than an exception.

| Symptom in the trace | Cause |
|---|---|
| A small later request is prefilled while an earlier bigger one waits | `continue` instead of `break` |
| Fewer requests admitted than the table shows (Level 3a) | `prompt_len + max_tokens` without the `- 1` |
| More requests admitted than the table shows (Level 3a) | used `r.kv` instead of the peak |
| Never preempts (Level 3b) | missing `+ 1` in the growth check |
| Preempts the oldest instead of the newest | `pop(0)` instead of `pop()` |
| Preempts more requests than the table (Level 3c) | forgot `+ 1` in `freed`, or evicted the whole pool instead of stopping at the shortfall |
| Evicts an equal-priority request (Level 3c) | `<=` instead of `<` |
| Ties come out newest-first (Level 3c) | `reverse=True` on a tuple key, or forgot sort stability |
| Decodes in the wrong order (Level 3c) | iterated your filtered `active` list instead of `view.admitted` |
| Off by one step everywhere | forgot that prefill produces the first token, so `max_tokens` needs `max_tokens - 1` decodes |

## The five recurring root causes

Almost every bug above is one of these five:

1. **Treating the view as live state.** It is a snapshot. Budgets are yours to
   track.
2. **`prompt_len` vs `prefill_remaining`.** They are equal only for a fresh,
   untouched request.
3. **Peak vs current memory.** 3a wants peak. 3b and 3c want current, plus the
   `+ 1` growth.
4. **Stop vs skip.** `break`, essentially always.
5. **Forgetting prefill emits a token.** `max_tokens = n` means one prefill and
   `n - 1` decodes.

## A debugging routine that works

1. **Read the error message.** These messages name the rule; most of the
   diagnosis is in them.
2. **Print your trace** next to the spec's table (`render()`, see
   `04-prereq-pytest.md`). Find the **first** step that differs — later
   divergence is usually a consequence, not a cause.
3. **Hand-trace that one step** from the spec's rules. Do not read your code
   yet; derive what *should* happen.
4. **Then** compare against your code, one spec bullet at a time.
5. If the hand-trace disagrees with the spec's table, you have misread a rule.
   Re-read it. Debugging cannot help you.

Step 3 is the one people skip and it is the one that saves the most time.
