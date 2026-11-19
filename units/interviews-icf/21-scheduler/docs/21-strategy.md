# Strategy under time pressure

The real assessment is ~90 minutes for all five levels, graded on **tests
passed**, not on style, elegance or completeness. That grading rule should
change how you work.

## Checkpoints

| Elapsed | You should be |
|---|---|
| 0:05 | read the whole README once, all levels, before writing anything |
| 0:15 | Level 1 green |
| 0:35 | Level 2 green |
| 0:55 | Level 3a green |
| 1:15 | Level 3b green |
| 1:30 | Level 3c green, or a partial 3c that passes some tests |

More than ten minutes behind? **Skip to the next level.** Levels are separate
files with separate tests. A green 3a is worth more than a nearly-green 2.

## Read all the levels first

Five minutes, before a single line. You are not looking for the solution; you
are looking for **what changes**. Knowing that Level 3b will re-queue
preempted requests with a larger `prefill_remaining` is what makes you write
`request.prefill_remaining` in Level 1 instead of `request.prompt_len` — and
that one habit saves a confusing failure an hour later.

Cheap things to notice on the first read:

- chunking exists only in Level 2,
- memory arrives in 3a and the definition *changes* in 3b,
- priority arrives in 3c and affects **two** separate decisions (admission and
  eviction), not one.

## Copy forward, then delete

Each level starts from the previous one. Paste, then **delete what the new
level removed** before adding what it added. Level 3a's first edit is deleting
Level 2's `min(budget, ...)` chunking. Skipping that deletion is the single
most common way to fail 3a.

## Comment the rules before coding them

Sixty seconds, described in `06-reading-the-spec.md`. Paste the spec's bullet
list into the function as comments and fill in underneath. You cannot forget a
rule, and a failure tells you which bullet to re-read.

## Do not refactor

You will be tempted to extract a shared helper across levels. Don't. The
levels' rules diverge (peak vs current, stop vs chunk, one victim rule vs
another), so a shared helper accumulates flags, and a bug in it breaks levels
that were already green. Five separate 20-line functions with duplicated
structure is the correct engineering trade-off *for this task*.

The same goes for defensive code. If no test exercises an empty queue or a
zero-token prompt, do not write code for it.

## When you are stuck: the fifteen-minute rule

If a level has been red for fifteen minutes:

1. Print the trace and find the first differing step. (Two minutes. Do this
   first, always.)
2. Hand-trace that step from the spec.
3. If you now see it, fix it. If not, **move to the next level** and come back.

Later levels are not strictly harder to *pass* — 3b is arguably easier to get
right than 2, because its rules are more mechanical. Leaving a level red is a
tactical choice, not a defeat.

## Partial credit is real

Grading is per test. A Level 3c that handles growth-preemption and priority
admission but skips preemption-for-admission still passes several tests. Write
the parts you are sure of, run, and bank them.

Similarly: if you run out of time mid-level, make sure the code you leave
behind **runs**. A syntax error can take a file from partial credit to zero.
Comment out the half-written block before the timer ends.

## Typing speed matters more than you think

The bottleneck in these tasks is rarely thinking; it is the accumulated cost
of typos, and of re-deriving `sorted(xs, key=lambda x: (-x[1], x[0]))` for the
fourth time. That is what `drills/` is for. Ten minutes of drills before a
practice run pays for itself.

## After a practice run

Write down, before you look at anything:

1. Which level cost the most time, and why.
2. What you had to rewrite from an earlier level.
3. Every typo or syntax error you hit — these are the cheapest minutes to
   recover.

Then re-read `20-common-bugs.md` and see how many of your bugs are already in
it. Usually most of them, which is the point: the failure modes are a short,
finite list.

## Practising under real conditions

If you are using this as a mock rather than a tutorial:

- Phone in another room. One monitor. Notifications off.
- **Disable Copilot or any AI completion.** Check it; don't assume.
- Only `https://docs.python.org/3/` open.
- Do not open `solutions/` or `docs/1*.md`. Start from `levels/` and the
  level specs only.
- Set a 90-minute timer and stop when it ends, mid-word.
