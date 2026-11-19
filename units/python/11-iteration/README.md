# 11 — Iteration, closures and decorators

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/python/11-iteration/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/python/iteration?utm_source=github&utm_medium=labs)

*~50 minutes · 30 tests*

Three features that look unrelated and are really the same idea: **Python
functions carry state**. A generator carries its suspended stack frame, a
closure carries the enclosing scope, a decorator carries the function it
wrapped. Once you see the shared shape, the whole family stops being magic.

This is also where the most-quoted Python interview traps live — late
binding, mutable defaults, generators that can only be consumed once.

Work in `exercises.py`. Check your work with `python -m labkit check 11`, or open the notebook (`lab.ipynb`) in Colab.

Each section below ends with **Test it** — the exact command for just that
piece, run from `units/python/11-iteration/`. No need to finish the whole
file before checking your work.

## What you'll build

Nine functions in `exercises.py`, in this order. Keep this table open as you
go — each section below tells you which row it's building:

| Function | You'll implement | Shows up in real code as |
|---|---|---|
| `Countdown` | `__iter__` | A restartable iterable — the same shape a PyTorch `Dataset`/`IterableDataset` needs, since every training epoch calls `iter(loader)` again, not just the first |
| `sliding` | `sliding` | Overlapping context windows — turning a token stream or price series into fixed-width samples for a sequence model |
| `chunked` | `chunked` | Batching a stream into minibatches without holding the whole dataset in memory — what a `DataLoader` does over an `IterableDataset` |
| `flatten` | `flatten` | Flattening nested batches or nested JSON records that come out of a data pipeline before they're fed to a model |
| `unique` | `unique` | Deduplicating a token stream while preserving order — building a vocabulary from a corpus in one pass, without materialising it first |
| `make_counters` | `make_counters` | The closure-capture bug behind registering one forward hook per layer in a loop and having every hook report the same, last layer index |
| `retry` | `retry` | Retrying a flaky call to an LLM API through rate limits and transient 5xxs, without swallowing the final error |
| `counted` | `counted` | Instrumentation — counting calls to an inference function or cache lookups without touching its body |
| `atomic` | `atomic` | Transactional updates to shared state — an in-memory feature store or run-metadata dict that must never end up half-written |

---

## 1. Iterable vs iterator — the distinction people fail on

An **iterable** has `__iter__` and can produce a fresh iterator on demand.
An **iterator** has `__iter__` *and* `__next__`, and is consumed exactly
once:

```python
>>> numbers = [1, 2, 3]        # iterable
>>> it = iter(numbers)         # iterator
>>> next(it)
1
>>> list(it), list(it)         # second pass over the same iterator: empty
([2, 3], [])
```

A list can be looped over twice; the iterator you pulled out of it cannot —
`next()` moves it forward permanently. That distinction is exactly what
bites in code like this:

```python
>>> rows = (x for x in [1, 2, 3])   # a generator IS an iterator
>>> total = sum(rows)
>>> total
6
>>> count = sum(1 for _ in rows)    # rows is already exhausted
>>> count
0
```

That pattern — build a generator, consume it once, then try to reuse it —
is a bug that reaches production constantly (`total = sum(r.size for r in
rows); count = sum(1 for r in rows)` over a file-parsing generator, say).
The fix is either materialising (`rows = list(...)`) or making the *source*
an iterable whose `__iter__` hands back a new generator every time — which
is exactly what `Countdown` does:

```python
class Countdown:
    def __init__(self, start):
        self.start = start

    def __iter__(self):              # a fresh generator per loop
        current = self.start
        while current > 0:
            yield current
            current -= 1
```

Because `__iter__` is itself a generator function, *calling* it returns a
brand-new generator object each time, so the instance itself never gets
consumed:

```python
>>> from solutions import Countdown
>>> c = Countdown(3)
>>> list(c)
[3, 2, 1]
>>> list(c)          # same instance, second pass — works fine
[3, 2, 1]
>>> list(Countdown(0))
[]
```

**In `exercises.py`:** `Countdown.__iter__` is the only method this section
asks for.

**Test it:** `python3 -m unittest test_exercises.TestCountdown -v`

## 2. Generators are lazy, and laziness is the point

`yield` suspends the function, keeping its locals alive, and resumes on the
next `next()` — nothing in the body runs until the first pull:

```python
>>> def numbers():
...     print("start")
...     yield 1
...     print("middle")
...     yield 2
...
>>> gen = numbers()      # nothing printed yet — the body hasn't run
>>> next(gen)
start
1
>>> next(gen)
middle
2
```

That laziness is what lets `sliding`, `chunked`, `flatten` and `unique` all
work on data too large — or too infinite — to hold in memory at once. It
also has a sharp edge: a generator function that validates its arguments
does so *far* later than you'd expect, because the validation code is just
more body that doesn't run until the first `next()`. `sliding` and `chunked`
both demonstrate this below.

### `sliding`: overlapping windows

```python
def sliding(sequence, size):
    if size <= 0:
        raise ValueError("size must be positive")
    for start in range(len(sequence) - size + 1):
        yield tuple(sequence[start:start + size])
```

```python
>>> from solutions import sliding
>>> list(sliding([1, 2, 3, 4], 2))
[(1, 2), (2, 3), (3, 4)]
>>> list(sliding([1, 2], 5))    # window bigger than the input: nothing
[]
```

Watch the validation-timing trap in practice: `size <= 0` is checked inside
the generator body, so *calling* `sliding(..., 0)` raises nothing at all —
only consuming it does:

```python
>>> gen = sliding([1, 2, 3], 0)
>>> gen                          # no exception yet — the body hasn't run
<generator object sliding at 0x...>
>>> list(gen)                    # first next() runs the body -> raises now
ValueError: size must be positive
```

**In `exercises.py`:** `sliding` is this subsection's target.

**Test it:** `python3 -m unittest test_exercises.TestSliding -v`

### `chunked`: batching without materialising the input

```python
def chunked(iterable, size):
    if size <= 0:
        raise ValueError("size must be positive")
    iterator = iter(iterable)
    while True:
        batch = list(itertools.islice(iterator, size))
        if not batch:
            return
        yield batch
```

`itertools.islice(iterator, size)` pulls at most `size` items from whatever
iterator it's handed and stops — it never asks the underlying iterable how
long it is. That's the difference between working and hanging forever on an
infinite source:

```python
>>> from solutions import chunked
>>> import itertools
>>> list(chunked(range(7), 3))
[[0, 1, 2], [3, 4, 5], [6]]
>>> batches = chunked(itertools.count(), 2)   # an infinite iterable
>>> next(batches)
[0, 1]
>>> next(batches)
[2, 3]
```

Same delayed-validation behaviour as `sliding`: `chunked([1, 2, 3], 0)`
returns a generator object with no error; only `list(...)` or `next(...)`
on it raises.

**In `exercises.py`:** `chunked` is this subsection's target.

**Test it:** `python3 -m unittest test_exercises.TestChunked -v`

### `flatten`: `yield from` and the string trap

`yield from` delegates to a sub-iterator, one item at a time, which is what
makes a recursive flatten read like the plain description of the problem:

```python
def flatten(nested):
    for item in nested:
        if isinstance(item, (list, tuple)):
            yield from flatten(item)
        else:
            yield item
```

```python
>>> from solutions import flatten
>>> list(flatten([1, [2, [3, [4]]], 5]))
[1, 2, 3, 4, 5]
```

Testing "is it iterable?" instead of "is it a list/tuple?" recurses forever
on strings: `"a"[0]` is `"a"`, so a one-character string contains itself.
Strings must be treated as leaves, not containers:

```python
>>> list(flatten(["ab", ["cd"]]))
['ab', 'cd']
```

**In `exercises.py`:** `flatten` is this subsection's target.

**Test it:** `python3 -m unittest test_exercises.TestFlatten -v`

### `unique`: laziness must survive filtering

```python
def unique(iterable):
    seen = set()
    for item in iterable:
        if item not in seen:
            seen.add(item)
            yield item
```

The `seen` set is what makes this different from `sorted(set(...))` — order
is preserved, and, because it's still a generator, consuming *some* of the
output must not silently consume the *whole* input behind your back:

```python
>>> from solutions import unique
>>> list(unique([3, 1, 3, 2, 1]))
[3, 1, 2]
>>> source = iter([1, 1, 2, 3, 4, 5])
>>> result = unique(source)
>>> next(result)
1
>>> next(result)
2
>>> next(source)    # unique only pulled as far as it needed to — 3 is still there
3
```

**In `exercises.py`:** `unique` is this subsection's target.

**Test it:** `python3 -m unittest test_exercises.TestUnique -v`

## 3. Closures capture variables, not values

The single most-asked Python trap:

```python
>>> fns = [lambda: i for i in range(3)]
>>> [f() for f in fns]
[2, 2, 2]
```

Every lambda closes over the *same* variable `i`, not the value it happened
to hold when the lambda was created. By the time you call any of them, the
loop has finished and `i` is `2`. The idiomatic fix binds the value at
*definition* time, using a default argument:

```python
>>> fns = [lambda i=i: i for i in range(3)]
>>> [f() for f in fns]
[0, 1, 2]
```

Default arguments are evaluated **once, when the function is defined** —
which is what makes that fix work, and is also the reason `def f(items=[])`
is a bug: every call that doesn't pass `items` shares that one list.

You can see exactly what a closure captured after the fact, which is worth
being able to show in an interview:

```python
>>> def outer():
...     x = 5
...     def inner():
...         return x
...     return inner
...
>>> f = outer()
>>> f.__closure__[0].cell_contents
5
```

`make_counters` is the same fix applied on purpose:

```python
>>> from solutions import make_counters
>>> counters = make_counters(3)
>>> [c() for c in counters]
[0, 1, 2]
>>> make_counters(0)
[]
```

**In `exercises.py`:** `make_counters` is the only function this section
asks for.

**Test it:** `python3 -m unittest test_exercises.TestClosures -v`

## 4. Decorators are functions returning functions

A decorator takes a function and returns a replacement. Three things to get
right every time:

1. **`functools.wraps`.** Without it the wrapper loses the original's
   `__name__`, `__doc__`, `__module__`, type hints and `__wrapped__`. That
   breaks `help()`, breaks introspection-based frameworks, and makes stack
   traces lie. It costs one line.
2. **`*args, **kwargs`.** The wrapper must be signature-agnostic — it
   doesn't know what the wrapped function takes.
3. **State goes on the wrapper**, not in a global. `wrapper.calls` is
   readable from outside as `decorated.calls`, and stays scoped to that one
   decorated function.

### `retry`: decorator factories

`@retry(3)` is a *call*, so `retry` itself isn't the decorator — it's a
factory that builds and returns one. That's why it needs three nested
`def`s, not a mistake:

```python
def retry(times, exceptions=(Exception,)):   # factory: takes config
    def decorate(func):                      # decorator: takes the function
        @functools.wraps(func)
        def wrapper(*args, **kwargs):        # replacement function
            last = None
            for _ in range(times):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last = exc
            raise last
        return wrapper
    return decorate
```

`@retry(3)` evaluates `retry(3)` first, producing `decorate`, and *then*
applies `decorate` to the function underneath. On exhaustion it **re-raises
the last exception** — a retry decorator that swallows the final failure and
returns `None` converts a loud error into a silent wrong answer:

```python
>>> from solutions import retry
>>> calls = []
>>> @retry(3)
... def flaky():
...     calls.append(1)
...     if len(calls) < 3:
...         raise ValueError("not yet")
...     return "ok"
...
>>> flaky()
'ok'
>>> len(calls)
3
>>> @retry(2)
... def always_fails():
...     raise ValueError("nope")
...
>>> always_fails()
ValueError: nope
```

**In `exercises.py`:** `retry` is this subsection's target.

**Test it:** `python3 -m unittest test_exercises.TestRetry -v`

### `counted`: state lives on the wrapper

```python
def counted(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        return func(*args, **kwargs)
    wrapper.calls = 0
    return wrapper
```

```python
>>> from solutions import counted
>>> @counted
... def work(x):
...     return x * 2
...
>>> work.calls
0
>>> work(1), work(2)
(2, 4)
>>> work.calls
2
```

**In `exercises.py`:** `counted` is this subsection's target.

**Test it:** `python3 -m unittest test_exercises.TestCounted -v`

## 5. Context managers without writing a class

`@contextlib.contextmanager` turns a generator into a context manager: the
code before `yield` is `__enter__`, the value yielded is what `as` binds,
and the code after `yield` is `__exit__`.

```python
@contextmanager
def atomic(mapping):
    snapshot = dict(mapping)
    try:
        yield mapping
    except Exception:
        mapping.clear()
        mapping.update(snapshot)
        raise
```

Two details that matter:

- **`raise` at the end.** Not re-raising would *suppress* the exception,
  which is almost never what a rollback should do — the caller needs to
  know the block failed.
- **`mapping.clear()` + `update()`, not rebinding.** Assigning
  `mapping = snapshot` would only rebind the local name; the caller's dict
  would keep the mutations. Restoring in place is the whole job.

```python
>>> from solutions import atomic
>>> data = {"a": 1}
>>> with atomic(data) as scratch:
...     scratch["b"] = 2
...
>>> data
{'a': 1, 'b': 2}
>>> try:
...     with atomic(data) as scratch:
...         scratch["c"] = 3
...         raise ValueError("abort")
... except ValueError:
...     pass
...
>>> data
{'a': 1, 'b': 2}
```

For cleanup that must run on the exception path too, put it in a `finally`,
not after the `except`.

**In `exercises.py`:** `atomic` is the only function this section asks for.

**Test it:** `python3 -m unittest test_exercises.TestAtomic -v`

---

## The exercises

| Function | What it teaches |
|---|---|
| `Countdown` | Restartable iterable vs one-shot iterator |
| `sliding` | Generators over sequences; empty-result edge case |
| `chunked` | Streaming with `itertools.islice`; never call `len()` |
| `flatten` | `yield from`; strings are leaves |
| `unique` | Laziness — consuming two items must not drain the source |
| `make_counters` | Late binding, and the default-argument fix |
| `retry` | Decorator factories; re-raising the last failure |
| `counted` | `functools.wraps`; state on the wrapper |
| `atomic` | `@contextmanager`; restoring in place; letting errors propagate |

## Check your understanding

1. **What's the difference between an iterable and an iterator, and which
   one can you loop over twice?**
   <details><summary>Answer</summary>An iterable's <code>__iter__</code>
   returns a fresh iterator each time, so it can be looped over
   repeatedly. An iterator *is* that fresh state — once exhausted,
   looping over it again yields nothing.</details>

2. **Why does `unique` need to be lazy, not just correct?**
   <details><summary>Answer</summary>If it eagerly consumed the whole
   source to compute the result, it would defeat the point of using a
   generator at all — an unbounded or expensive-to-produce source would
   never finish, or would be fully materialized in memory regardless of
   how many items the caller actually needed.</details>

3. **`[lambda: i for i in range(3)]` — why do all three lambdas return
   `2`, and how do you fix it?**
   <details><summary>Answer</summary>Closures capture the *variable*
   <code>i</code>, not its value at creation time — by the time any
   lambda is called, the loop has finished and <code>i</code> is
   <code>2</code>. Fix it by binding the current value as a default
   argument: <code>lambda i=i: i</code>.</details>

4. **What does `functools.wraps` actually do, and what breaks without
   it?**
   <details><summary>Answer</summary>It copies the wrapped function's
   <code>__name__</code>, <code>__doc__</code> and other metadata onto
   the wrapper. Without it, a decorated function's introspection
   (its name in tracebacks, help text, <code>__wrapped__</code>) shows
   the wrapper's, not the original's.</details>

5. **Why does `retry` need to be a decorator *factory* (`retry(times=3)`)
   rather than a plain decorator?**
   <details><summary>Answer</summary>A plain decorator only takes the
   function being wrapped. A factory takes configuration
   (<code>times</code>) first and returns the actual decorator, so the
   number of retries can vary per use instead of being hard-coded.</details>

6. **`atomic` is written with `@contextmanager`, `try`/`except`/`yield`
   rather than a class. What would silently swallow the caller's
   exception here, and how does `atomic` avoid it?**
   <details><summary>Answer</summary>Catching the exception in the
   <code>except</code> block to restore the snapshot, then *not*
   re-raising it, would swallow it. <code>atomic</code> avoids this with
   a bare <code>raise</code> at the end of the <code>except</code> block,
   after restoring — the rollback happens, then the original exception
   still propagates.</details>

## Interview questions this module answers

- *Why does my list of lambdas all return the same number?*
- *What is the difference between an iterable and an iterator?*
- *Why did my second pass over this generator produce nothing?*
- *What does `functools.wraps` do, and what breaks without it?*
- *Write a retry decorator that takes the number of attempts.*
- *How do you process a 50 GB file without loading it?*
- *Why is `def f(x=[])` dangerous?*
