# 15 — Concurrency and async

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/python/15-concurrency/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/python/concurrency?utm_source=github&utm_medium=labs)

*~55 minutes · 34 tests*

Concurrency in Python is one decision repeated: **is this work waiting, or is
it computing?** Waiting overlaps with threads or `asyncio`. Computing does
not, because of the GIL, and needs processes or a library that releases the
lock.

Getting that first classification right is most of the interview. The rest is
knowing the standard shapes: bounded concurrency, backpressure, cancellation,
batching. Those four are exactly what a model-serving or data-loading layer
is made of.

Work in `exercises.py`. Check your work with `python -m labkit check 15`, or open the notebook (`lab.ipynb`) in Colab.

Each section below ends with **Test it** — the exact command for just that
piece, run from `units/python/15-concurrency/`. No need to finish the whole
file before checking your work.

## What you'll build

Seven functions and classes in `exercises.py`, in this order. Keep this
table open as you go — each section below tells you which row it's
building:

| Name | You'll implement | Shows up in real code as |
|---|---|---|
| `fetch_all` | `fetch_all` | Concurrent IO — fanning out to several feature stores or downstream APIs, or pulling multiple dataset shards, before assembling a batch |
| `Counter` | `increment` | A thread-safe counter — request or token counts updated from several worker threads in a serving layer |
| `producer_consumer` | `producer_consumer` | A bounded worker pool with backpressure — a preprocessing pipeline that feeds a model without buffering the whole input in memory |
| `gather_limited` | `gather_limited` | Bounded async fan-out — calling a rate-limited API or a GPU-backed service without swamping it |
| `with_timeout` | `with_timeout` | A timeout wrapper that actually cancels — the standard shape for bounding a slow downstream call |
| `Batcher` | `add`, `flush` | The batching layer of an inference server — amortising a fixed per-call cost (one GPU forward pass) over many requests |
| `cpu_bound_is_not_helped_by_threads` | `cpu_bound_is_not_helped_by_threads` | The interview answer for "why isn't this threaded preprocessing step actually faster?" |

Two more, covered below in §5 — `gather_limited` and `with_timeout` —
round out the seven. (`exercises.py` is generated from `solutions.py` by
`units/_make_stubs.py`; these two were briefly missing from that
generated file because the generator didn't recognise `async def` at
module level. That's a bug in the generator, now fixed — regenerating
produces both stubs correctly.)

---

## 1. The GIL, precisely

CPython's Global Interpreter Lock allows **one thread to execute Python
bytecode at a time**. So:

- **CPU-bound Python does not get faster with threads.** Two threads doing
  arithmetic take as long as one doing both — plus switching overhead.
- **IO-bound work does.** A thread blocked on a socket, a file, or a
  `time.sleep` releases the GIL, so the waits overlap. Ten 100 ms requests
  take ~100 ms, not one second.
- **NumPy, PyTorch and most C extensions release the GIL** inside their heavy
  kernels. A `np.dot` on a large matrix runs with the lock dropped, so
  threads *do* help there. This is why "the GIL makes threads useless" is too
  strong an answer, and saying so is worth marks.

The escape hatches: `multiprocessing` / `ProcessPoolExecutor` (real
parallelism, pays serialisation cost per task), the C-extension route, and —
worth naming as current context — **free-threaded CPython** (PEP 703),
available as an official build from 3.13 and no longer experimental in 3.14,
which removes the GIL at some single-thread cost.

**Choose:**

| Work | Tool |
|---|---|
| Many network calls | `asyncio` (thousands) or threads (dozens) |
| Blocking library with no async version | `ThreadPoolExecutor` |
| Pure-Python number crunching | `ProcessPoolExecutor` |
| NumPy / Torch kernels | Threads are fine — the GIL is released |

That reasoning — the GIL serialises CPU-bound bytecode, so reach for
processes instead of threads — is also the entire content of one exercise: a
function that states it in a single sentence.

```python
>>> cpu_bound_is_not_helped_by_threads()
'The GIL lets only one thread execute Python bytecode at a time, so CPU-bound work is serialised; use processes instead.'
```

**In `exercises.py`:** `cpu_bound_is_not_helped_by_threads` — the whole point
is the one sentence it returns; everything above is the reasoning behind it.

**Test it:** `python3 -m unittest test_exercises.TestGilExplanation -v`

## 2. Threads: the pool, and ordering

```python
with ThreadPoolExecutor(max_workers=8) as pool:
    return list(pool.map(fetch, urls))
```

`Executor.map` **preserves input order** even though completion order varies
— so you never need to sort results yourself. Use `as_completed` only when
you genuinely want results as they arrive.

Two behaviours to know:

- **Exceptions are re-raised when you consume the result**, not when the task
  runs. `pool.map` returns a lazy iterator; a failure surfaces at the
  `list(...)`. With `submit`, it surfaces at `future.result()`. A pool whose
  futures are never consumed swallows every error silently.
- **The `with` block joins on exit.** Leaving the context waits for
  everything submitted.

```python
>>> import time
>>> def fetch(n):
...     time.sleep((3 - n) * 0.01)   # n=2 finishes first, n=0 finishes last
...     return n * 2
...
>>> fetch_all([0, 1, 2], fetch)
[0, 2, 4]
```

Completion order here is `2, 1, 0`, yet the returned list is `[0, 2, 4]` —
input order, not completion order.

**In `exercises.py`:** `fetch_all`.

**Test it:** `python3 -m unittest test_exercises.TestFetchAll -v`

## 3. `+= 1` is not atomic

```python
self.value += 1
```

is `LOAD`, `ADD`, `STORE`. The interpreter can switch threads between those
bytecodes, so two threads can read the same value and both write back the
same increment — a lost update. The GIL guarantees only that a single
bytecode does not tear; it guarantees nothing about a sequence.

```python
def increment(self):
    with self._lock:
        self.value += 1
```

The test here runs 4 threads × 20k increments precisely because an unlocked
version loses updates reliably at that scale, not occasionally.

Alternatives worth naming: `itertools.count()` (its `__next__` is a single C
call), a `queue.Queue` (already locked internally), or per-thread partials
combined at the end — the last being the pattern that actually scales, since
it removes the contention rather than protecting it.

Also know **`threading.Lock` is not reentrant** — acquiring it twice in the
same thread deadlocks. `RLock` is the one that allows it.

```python
>>> c = Counter()
>>> c.increment()
>>> c.increment()
>>> c.value
2
```

**In `exercises.py`:** `Counter.increment`.

**Test it:** `python3 -m unittest test_exercises.TestCounter -v`

## 4. Producers, consumers and backpressure

```python
task_queue = queue.Queue(maxsize=consumers * 2)
```

The `maxsize` is the point. An unbounded queue lets a fast producer read the
entire input into memory while the consumers fall behind — the pipeline that
works on the test file and OOMs on the real one. A bounded queue makes
`put()` block once it is full, so the producer runs at the consumers' pace.
That is **backpressure**, and it is the single most useful word to have in
this answer.

Shutdown is the other half. Consumers blocked in `queue.get()` never return
on their own, so the producer sends one **sentinel per consumer**:

```python
for _ in threads:
    task_queue.put(sentinel)
```

A unique `object()` is the right sentinel — `None` may be a legitimate item.
Getting this wrong is how a script hangs at 100% complete, which the test
here catches by joining with a timeout.

```python
>>> sorted(producer_consumer([1, 2, 3, 4], lambda n: n * n))
[1, 4, 9, 16]
```

**In `exercises.py`:** `producer_consumer`.

**Test it:** `python3 -m unittest test_exercises.TestProducerConsumer -v`

## 5. asyncio: cooperative, single-threaded

One thread, one event loop, tasks that yield control at every `await`.
Nothing is preempted, so there is no data race between awaits — but a single
blocking call freezes **everything**. `time.sleep(1)` inside a coroutine
stalls the whole loop; `await asyncio.sleep(1)` does not. For a blocking
library with no async version, hand it to a thread with
`asyncio.to_thread(...)`.

### Bounded concurrency

`asyncio.gather` starts *everything* at once. Against a rate-limited API or a
GPU-backed service that is a denial-of-service on your own infrastructure.
The semaphore is the standard cap:

```python
semaphore = asyncio.Semaphore(limit)

async def run(factory):
    async with semaphore:
        return await factory()

return await asyncio.gather(*(run(f) for f in coro_factories))
```

`gather` returns results **in input order** regardless of completion order —
same guarantee as `Executor.map`.

Note the signature takes **factories, not coroutines**. A coroutine object
can be awaited exactly once; building them all up front means you cannot
retry, and any you never await produce a "coroutine was never awaited"
warning. Passing zero-argument callables keeps the work re-creatable.

```python
>>> import asyncio
>>> def make(n):
...     async def run():
...         await asyncio.sleep(0.01)
...         return n
...     return run
...
>>> asyncio.run(gather_limited([make(n) for n in range(5)], limit=2))
[0, 1, 2, 3, 4]
```

**In `exercises.py`:** `gather_limited`.

**Test it:** `python3 -m unittest test_exercises.TestGatherLimited -v`

### Cancellation and timeouts

```python
try:
    return await asyncio.wait_for(coro, timeout=seconds)
except asyncio.TimeoutError:
    return default
```

`wait_for` **cancels** the task on timeout — it does not leave it running.
Catching `TimeoutError` from something that does not cancel leaves the work
executing in the background, still holding its connection: the classic leak.
The test asserts on the cancellation itself, not on elapsed time.

Cancellation arrives as `CancelledError` raised at the coroutine's current
`await`. Two rules: clean up in `finally`, and **re-raise it** — swallowing
`CancelledError` means the task refuses to die. Since 3.8 it inherits from
`BaseException`, so a bare `except Exception` no longer catches it by
accident.

On 3.11+, `asyncio.timeout()` as a context manager and `TaskGroup` (which
cancels siblings when one fails) are the modern spellings. Knowing both the
old and the new names is worth a sentence in an interview.

```python
>>> import asyncio
>>> async def quick():
...     return "ok"
...
>>> async def slow():
...     await asyncio.sleep(5)
...     return "ok"
...
>>> asyncio.run(with_timeout(quick(), 1.0))
'ok'
>>> asyncio.run(with_timeout(slow(), 0.05, default="fallback"))
'fallback'
```

**In `exercises.py`:** `with_timeout`.

**Test it:** `python3 -m unittest test_exercises.TestWithTimeout -v`

## 6. Batching

```python
def add(self, item):
    self.pending.append(item)
    if len(self.pending) >= self.size:
        return self.flush()
```

Amortising a fixed per-call cost over many inputs — the core of every batched
inference server. One forward pass over 32 prompts costs far less than 32
passes over one.

The detail the test pins down: `flush` must **detach** the batch before
handing it over.

```python
batch, self.pending = self.pending, []      # detached
self.flush_fn(batch)

self.flush_fn(self.pending); self.pending.clear()    # wrong — clears the batch
```

Passing the live list and clearing it afterwards hands the callback an empty
list if it holds onto the reference.

What a real batcher adds: a **timer**, so a partial batch does not wait
forever for a 32nd request. That is the latency/throughput trade-off, and
it is the natural follow-up question — larger batches mean better GPU
utilisation and worse tail latency.

```python
>>> batcher = Batcher(2, lambda batch: len(batch))
>>> batcher.add("x")            # 1st item - batch not full yet
>>> batcher.add("y")            # 2nd item - flushes, returns flush_fn's result
2
>>> batcher.pending, batcher.batches
([], 1)
```

**In `exercises.py`:** `Batcher.add` and `Batcher.flush`.

**Test it:** `python3 -m unittest test_exercises.TestBatcher -v`

---

## The exercises

| Function | What it teaches |
|---|---|
| `fetch_all` | Thread pools; `map` preserves order; exceptions surface late |
| `Counter` | `+=` is not atomic; locking |
| `producer_consumer` | Bounded queues, backpressure, sentinel shutdown |
| `gather_limited` | Semaphores; factories vs coroutine objects |
| `with_timeout` | `wait_for` cancels; the leak if it does not |
| `Batcher` | Amortising per-call cost; detaching the batch |
| `cpu_bound_is_not_helped_by_threads` | Saying the GIL answer in one sentence |

Every function in this table has a stub in `exercises.py` and test
coverage in `test_exercises.py` — a clean 1:1 mapping.

## Check your understanding

1. **"The GIL makes threads useless in Python" — why is that wrong?**
   <details><summary>Answer</summary>The GIL only serializes execution of
   Python *bytecode*. Threads still genuinely help for I/O-bound work
   (the GIL is released while waiting), and libraries like NumPy and
   PyTorch release it inside their compiled kernels — so threads help
   there too.</details>

2. **Why isn't `counter += 1` atomic, even though it looks like one
   operation?**
   <details><summary>Answer</summary>It's actually read, add, write as
   three separate steps. Two threads can both read the same value before
   either writes back, and one thread's update gets silently lost — a
   lock is needed to make the three steps atomic together.</details>

3. **In a bounded producer/consumer queue, what stops the producer from
   running far ahead of the consumer and exhausting memory?**
   <details><summary>Answer</summary>Backpressure — a bounded queue's
   <code>put</code> blocks once it's full, until a consumer makes room
   by taking something off. The producer is forced to slow down to the
   consumer's pace.</details>

4. **Why does `asyncio` need only one thread to run many coroutines
   concurrently?**
   <details><summary>Answer</summary>Coroutines cooperatively yield
   control at each <code>await</code> instead of being preemptively
   interrupted — the single-threaded event loop switches between them at
   those yield points, so nothing needs a second thread to make
   progress.</details>

5. **`gather_limited` takes coroutine *factories*, not coroutine objects.
   Why does that distinction matter?**
   <details><summary>Answer</summary>A coroutine object can only be
   awaited once — building them all up front would mean none could be
   retried, and any never awaited would raise a "coroutine was never
   awaited" warning. A factory can be called fresh each time one is
   actually needed.</details>

6. **What does batching actually trade off in an inference server?**
   <details><summary>Answer</summary>It amortizes a fixed per-call cost
   (like one GPU forward pass) across many requests, improving
   throughput — at the cost of added latency, since a request may wait
   for the batch to fill (or a timeout) before it's processed.</details>

## Interview questions this module answers

- *What is the GIL, and when does it actually matter?*
- *Threads, processes, or asyncio — for this workload, which and why?*
- *Is `counter += 1` thread-safe?*
- *Limit this to 10 concurrent requests.*
- *What happens to the coroutine when the timeout fires?*
- *Why did this pipeline run out of memory?*
- *How does batching help an inference server, and what does it cost?*

## A note on the tests

Every test here asserts on **structure** — ordering, counts, peak concurrency
recorded by the workers, whether a task was actually cancelled — never on
elapsed wall-clock time. Timing assertions are flaky on shared machines and
in CI, and a flaky test is worse than no test. The few `sleep` calls exist to
*force* interleaving, not to be measured.
