# The technique: reading a levelled spec

Most people who fail this kind of task can write the code. They fail on the
reading. This chapter is the method — it is short, and it is the difference
between finishing Level 3c and finishing Level 2.

## 1. Find the sentence that is the algorithm

A level spec is prose, but hidden inside it is a numbered list that *is* the
implementation. Level 1's is:

> - Schedule a decode for all admitted requests, oldest first.
> - Then, in arrival order, schedule each waiting prefill while the total
>   work this step stays at most `view.max_work` units.
> - Stop at the first waiting prefill that doesn't fit — don't skip ahead to
>   a smaller, later one.

Three bullets, three code blocks, in that order. Before writing anything,
find that list and copy it into your file as comments. Then fill in under each
comment. You now cannot forget a rule, and if a test fails you know which
bullet to re-read.

```python
def plan_step(view):
    plan = StepPlan()
    # decode all admitted, oldest first

    # then waiting prefills in arrival order, while work <= max_work

    # stop at the first that doesn't fit (break, not continue)

    return plan
```

This costs sixty seconds and it is the highest-return sixty seconds in the
task.

## 2. Words that are load-bearing

Specs like this are written tightly. These words are never decoration:

| Word | Means | If you miss it |
|---|---|---|
| **stop at** the first that doesn't fit | `break` | you `continue`, pack the budget fuller, and fail a fairness test |
| **strictly** lower priority | `<` not `<=` | you evict an equal-priority request and fail |
| at its **peak**, not its current usage | `prompt_len + max_tokens - 1` | you use `r.kv` and over-admit |
| **oldest first** / **most recently** admitted | list order, or reversed | victim selection is backwards |
| **only if it actually suffices** | check the total *before* evicting anyone | you evict, still can't fit, and thrash |
| a victim does **no** decode this step | remove it from `plan.decode` | `SchedulingError: in both preempt and decode` |
| **before** decoding | ordering between phases | memory accounting is off by the decode growth |
| **at most** / **at least** | `<=` / `>=` | classic off-by-one |

When a sentence contains one of these, that sentence maps to a *specific
character* in your code. Find the character.

## 3. The worked example is a test — use it as one

Every level ends with a step table. It is not illustration; it is the
acceptance test, and it is usually the *only* fully specified behaviour you
get. Before you write code, read the table and ask of each row: *why that and
not something else?*

The Level 3c table's step 2 is the whole level in one row:

```
step  waiting  admitted   preempt  prefill  decode  finished
2     HI       MID, LOW   LOW      HI:4     MID     MID, HI
```

Four separate rules are visible in that single row:

- `LOW` is evicted, not `MID` → victims are chosen by **lowest priority**.
- `MID` is *not* evicted even though it too is lower priority than `HI` →
  evict **only as many as needed**.
- `LOW` appears in `preempt` but not in `decode` → **a victim does no work**
  this step.
- `HI:4` is admitted the same step the eviction happens → freed memory is
  usable **immediately**.

If your output differs from the table in exactly one cell, you have broken
exactly one of these rules. That is a much faster diagnosis than re-reading
your whole function.

## 4. Trace the first two steps by hand before you run anything

Cheaper than a test cycle, and it catches misunderstanding rather than typos.
For Level 3a's example (`max_work=10`, `kv_capacity=6`; A(2,2), B(4,2), C(2,2)):

> Step 0. Nothing admitted, so no decodes. Waiting is A, B, C in arrival
> order. A's peak is 2 + 2 − 1 = 3; reserved is 0; 0 + 3 ≤ 6, so admit,
> prefill A:2, reserved = 3. B's peak is 5; 3 + 5 = 8 > 6, so it does not fit
> — **stop**, do not try C. Plan: prefill A:2. ✓ matches the table.

If your hand-trace disagrees with the table, you have misread a rule and no
amount of debugging will help. Re-read before you type.

## 5. Later levels rewrite earlier rules — read for what *changed*

The levels are not additive. Level 3b does not extend 3a; it **replaces** its
admission rule (peak reservation becomes current-usage reservation). Level 3c
replaces 3b's victim rule (most-recent becomes lowest-priority-then-most-recent).

So read each new level asking "what did they *take away*?" as well as "what
did they add". A useful habit: keep a one-line note per level.

| Level | Admission test | Victim rule | Chunking |
|---|---|---|---|
| 1 | work only | — | no |
| 2 | work only | — | **yes** |
| 3a | work + **peak** memory | — | no |
| 3b | work + **current** memory | most recently admitted | no |
| 3c | work + current memory, **may evict to fit** | lowest priority, then most recent | no |

Notice Level 2 is the only one with chunking. It is easy to carry `min(budget,
remaining)` forward into 3a out of habit and get a `SchedulingError` about
chunking being off.

## 6. Where the tests disagree with the prose, the tests win

Prose is ambiguous; the test suite is not. If you cannot decide between two
readings, write the one you believe, run it, and let the failure tell you.
Do not spend five minutes reasoning about a sentence when thirty seconds of
`./run 3b` will answer it.

Corollary: **do not handle edge cases no test checks.** Defensive code for
situations the spec never produces costs time and adds bugs.

## 7. When you are stuck, say the rule out loud

Genuinely. Most stuck-ness on these tasks is a rule you have half-internalised
in a form that is subtly wrong. Saying "evict the lowest priority request,
breaking ties towards the most recently admitted" forces you to notice that
your code says `max` where it should say `min`.

## Drill

Take the Level 3c rules in `docs/14-level3c.md`, and *without looking at the
code*, write out the comment skeleton — one comment per rule, in execution
order. Then compare against `solutions/level3c.py`. Any rule you did not write
down is a rule you would have missed.
