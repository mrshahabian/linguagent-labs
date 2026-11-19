# Mock Assessment 2 — LLM Inference Scheduler

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/interviews-icf/21-scheduler/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/icf/scheduler?utm_source=github&utm_medium=labs)

A five-level CodeSignal ICF-style exercise, with a **runnable simulator** so you
can execute and test every level locally, and a full course explaining it.

You write the scheduler for an LLM inference server. Each step the engine hands
you a read-only snapshot and asks: given what is waiting, what is running, and
how much GPU and memory you have, what should the GPU do *this* step?

| Level | Adds | The one new idea |
|---|---|---|
| 1 | Basic scheduling | a work budget, and stopping (not skipping) when it runs out |
| 2 | Chunked prefill | a prefill can be split across steps |
| 3a | KV-aware admission | memory, reserved pessimistically at peak |
| 3b | Preemption | memory reserved optimistically, evicting when you guess wrong |
| 3c | Priority tiers | some requests outrank others, for admission *and* eviction |

## Running it

```
./run             every level test, against your code in levels/
./run 1           one level (1, 2, 3a, 3b, 3c)
./run rules       the engine-rule tests — these pass without your code and
                  document what the engine rejects and why
./run --check     the same tests against solutions/, to see them pass
./drills/drill    warm-up drills for the sub-skills
```

## Two ways to use this

**As a course.** Start at `docs/00-orientation.md` and work through in order.
The prerequisites are short and every one is used directly. Roughly four hours
end to end.

**As a mock.** 90 minutes, timer running. Read only the spec at the top of each
`docs/1*.md`, write in `levels/`, and stay out of `solutions/`. See
`docs/21-strategy.md` for checkpoints and the room setup.

## Layout

```
scheduler/     api.py (read this), engine.py (read it after you finish —
               every rule in the docs is enforced by a check in there),
               trace.py (render a run as the spec's step table)
levels/        YOU WRITE HERE. Five stubs.
solutions/     reference implementations, heavily commented
tests/         the spec's worked examples as executable tests, plus the
               engine-rule tests
drills/        sub-skill drills — ./drills/drill, ./drills/drill B
docs/          the course
```

## The course

| Doc | |
|---|---|
| `00-orientation.md` | what this is, how it's assessed, the reading path |
| `01-prereq-dataclasses.md` | frozen dataclasses, `default_factory`, why the view is immutable |
| `02-prereq-sorting.md` | `key=`, tuple keys, stability. **Level 3c depends on this** |
| `03-prereq-budget-loops.md` | the accumulate-and-stop loop every level is built from |
| `04-prereq-pytest.md` | reading a failure, printing your own trace |
| `05-domain-llm-inference.md` | what prefill, decode and the KV cache actually are |
| `06-reading-the-spec.md` | the technique — load-bearing words, using the worked example as a test |
| `10-level1.md` … `14-level3c.md` | one per level: rules, derivation, trace, pitfalls, drills |
| `20-common-bugs.md` | symptom-first catalogue. Every `SchedulingError` explained |
| `21-strategy.md` | 90-minute checkpoints, partial credit, when to skip a level |

## Debugging loop

Print your trace next to the spec's table and find the first step that differs:

```python
from scheduler import Arrival, Engine, render
from levels.level3b import plan_step

engine = Engine(
    [Arrival("A", 2, 2), Arrival("B", 4, 2), Arrival("C", 2, 2)],
    max_work=10, kv_capacity=6,
)
print(render(engine.run(plan_step), kv=True))
```

```
step  waiting  admitted  kv held  preempt  prefill   decode  finished
0     A, B, C  -         0        -        A:2, B:4  -       -
1     C        A, B      6        B        C:2       A       A
2     B        C         2        -        -         C       C
3     B        -         0        -        B:5       -       B
```

## The five ideas the whole thing rests on

1. **The view is a snapshot.** It does not change as you build the plan.
   Tracking budgets is your job.
2. **Prefill produces the first token.** `max_tokens = n` needs one prefill and
   `n - 1` decodes.
3. **`prefill_remaining`, not `prompt_len`.** They differ for a partially
   prefilled or previously preempted request.
4. **Stop, don't skip.** `break` at the first candidate that doesn't fit — a
   fairness rule, deliberately costing throughput.
5. **Peak vs current memory.** 3a reserves at peak; 3b and 3c reserve current
   usage plus the `+ 1` each decode grows into.

## Check your understanding

1. **Why does "the view is a snapshot" matter for how you track
   budgets while building your answer?**
   <details><summary>Answer</summary>The snapshot doesn't update as you
   add requests to the plan in your head — if you don't track remaining
   budget yourself, you'll keep reading a stale number and overcommit
   resources the snapshot never actually reflected being used.</details>

2. **For `max_tokens = n`, how many prefill and decode steps does that
   require, and why isn't it `n` decodes?**
   <details><summary>Answer</summary>One prefill produces the first
   token, and each decode step produces one more — so `n` total tokens
   needs 1 prefill + `n - 1` decodes, not `n` decodes.</details>

3. **Why does the scheduler need `prefill_remaining` instead of just
   using `prompt_len` directly?**
   <details><summary>Answer</summary>They only match for a request
   that's never been partially processed. A request that was previously
   preempted or partially prefilled has less prefill work remaining than
   its original prompt length — using `prompt_len` would overcount the
   remaining work.</details>

4. **"Stop, don't skip" at the first candidate that doesn't fit — why
   `break` instead of `continue`, given `continue` would pack more
   requests into the same batch?**
   <details><summary>Answer</summary>It's a deliberate fairness rule,
   not an oversight — skipping over a request that doesn't fit to pack
   in a smaller one that does would let smaller requests perpetually cut
   the line ahead of larger ones waiting their turn.</details>

5. **What's the actual difference between "peak" and "current" memory
   reservation, and why do different levels use different ones?**
   <details><summary>Answer</summary>Peak reserves for the maximum a
   request could ever grow to; current reserves only what's used right
   now, plus headroom for the immediate next step
   (<code>+ 1</code> per decode). Which one a level specifies changes
   how many requests can be admitted concurrently — using the wrong one
   passes some tests and fails others that specifically probe the
   difference.</details>

## Related

- `../08-drills/` — the original typing-speed drills.
- `../20-registry/` — a five-level package-registry mock.
- `../09-container/` — the pre-certification playground: the platform
  mechanics (RUN modes, `main.sh`, partial credit) plus the Container task.
- `../34-stations/` — a NumPy-based mock, shorter and with a different
  failure mode (silent NaN propagation instead of a moving spec).
- `../33-trees/` — a NumPy debugging mock, for
  assessments built on `numpy` rather than plain standard library.
