# Scheduler warm-up drills

29 tests across 7 blocks. Total budget: ~75 minutes. These are not puzzles —
every one is a line you will write inside `plan_step`, extracted so you can
practise it without the surrounding problem. The goal is speed and zero typos.

```
./drill          run everything
./drill B        run one block, verbose
./drill --check  run against solutions.py
```

Implement in `drills.py`. Work block by block, in order.

| Block | Topic | Budget | Used by |
|---|---|---|---|
| A | Dataclasses and immutable views | 8 min | every level |
| B | Compound sort keys | 12 min | Level 3c |
| C | Budget loops | 12 min | every level |
| D | KV arithmetic | 10 min | Levels 3a, 3b |
| E | Eviction accounting | 12 min | Level 3c |
| F | Small class design + delegation | 12 min | general ICF |
| G | Formatting and transformation | 8 min | general ICF |

## Which blocks matter most

**Block B** is the one to over-practise. `sorted(xs, key=lambda x: (-x[1], x[0]))`
appears in nearly every ICF task, and fumbling it costs minutes each time. The
two drills that matter here are `by_priority_desc` (one key, relying on sort
stability) and `lowest_priority_newest` (the `(priority, -rank)` victim key) —
between them they are the whole of Level 3c's ordering logic.

**Block C** is the difference between `take_while_fits` and `take_greedily`.
They are four characters apart — `break` versus `continue` — and only one is
correct for this exercise. Writing both, side by side, is the point.

**Block E** is Level 3c's hardest part in isolation: the all-or-nothing check
and evicting as few as possible. If Level 3c is going badly, come back here.

**Block F** contains the delegation pattern — a plain method that defers to a
timestamped one, without duplicating the logic. That is exactly the shape
Level 3 of a real ICF task tends to demand, and having the reflex means you
won't be refactoring under time pressure.

## Rules

Same conditions as the real thing: no AI, no Stack Overflow, no autocomplete.
`https://docs.python.org/3/` only.

## When you're done

Note which blocks ran over budget. Those are what you drill again next session.
`./drill --check` shows you the reference answers passing; `solutions.py` shows
you how they're written.
