# ICF Warm-up Drills

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/interviews-icf/08-drills/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/icf/drills?utm_source=github&utm_medium=labs)

27 tests across 7 blocks. Total budget: ~75 minutes. These are not puzzles —
they are the primitives you will use over and over in the real assessment.
The goal is speed and zero typos, not cleverness.

    ./drill        # run everything
    ./drill B      # run one block, verbose

Implement in `drills.py`. Work block by block, in order.

| Block | Topic | Budget |
|---|---|---|
| A | Dictionaries as the default structure | 10 min |
| B | Compound sort keys | 15 min |
| C | Prefix matching and grouping | 8 min |
| D | Interval / TTL arithmetic, half-open | 12 min |
| E | Snapshot and rollback | 10 min |
| F | Small class design + delegation | 12 min |
| G | Formatting and parsing | 8 min |

## Rules

Same conditions as the real thing: no AI, no Stack Overflow, no autocomplete.
`https://docs.python.org/3/` only.

Block B is the one to over-practice. `sorted(xs, key=lambda x: (-x[1], x[0]))`
appears in nearly every ICF task, and fumbling it costs minutes each time.

Block F contains the delegation pattern — a plain method that defers to a
timestamped one — which is exactly the shape Level 3 of a real ICF task
demands. Getting this reflex now means you won't be refactoring under time
pressure later.

## Check your understanding

1. **`sorted(xs, key=lambda x: (-x[1], x[0]))` — what does each half of
   the tuple key actually do?**
   <details><summary>Answer</summary><code>-x[1]</code> sorts by the
   second field descending (negating flips ascending into descending
   for numbers). <code>x[0]</code> is the tiebreaker, sorted ascending,
   applied only when the first field is equal.</details>

2. **Why does the delegation pattern (a plain method calling a
   timestamped one at `timestamp=0`) save real time under the clock,
   rather than just being tidier code?**
   <details><summary>Answer</summary>When a later level adds a
   timestamp to every existing method, delegating means only the
   timestamped version needs new logic — the plain method's behavior
   for free. Duplicating the logic instead means every later level
   costs double, since both copies need updating.</details>

3. **What does "half-open" mean for a TTL interval, and which end is
   excluded?**
   <details><summary>Answer</summary>A half-open interval
   <code>[start, start+ttl)</code> includes <code>start</code> but
   excludes <code>start+ttl</code> — an item is expired exactly at (not
   just after) its expiry timestamp.</details>

4. **Block E is "snapshot and rollback." What's the failure mode if a
   snapshot stores a reference instead of a copy?**
   <details><summary>Answer</summary>Later mutations to the live state
   would also silently change the "snapshot," since it was never
   actually a separate copy — restoring it would restore whatever the
   live state happens to be now, not the state at snapshot time.</details>

5. **Why is Block B — compound sort keys — worth over-practicing
   specifically, more than the others?**
   <details><summary>Answer</summary>It appears in nearly every ICF
   task, in some form — getting the tuple-key shape wrong (or fumbling
   the syntax under time pressure) costs minutes every single time it
   comes up, not just once.</details>

## When you're done

Note which blocks ran over budget. Those are what you drill again on Friday.

## Related

- `../09-container/` — the pre-certification playground these drills feed
  straight into.
- `../20-registry/` — Blocks D, E and F drill exactly the primitives this
  five-level mock needs.
- `../21-scheduler/` — a harder five-level mock with its own drill set
  (`21-scheduler/drills/`) once you've done these.
- `../33-trees/` — a NumPy debugging mock, a different format from
  anything drilled here.
- `../34-stations/` — a NumPy build-from-spec mock, shorter than
  `21-scheduler`.
