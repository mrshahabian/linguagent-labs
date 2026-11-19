# 17 — Functional patterns: itertools, functools, decorators

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/python/17-functional-patterns/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/python/functional-patterns?utm_source=github&utm_medium=labs)

*~45 minutes · 31 tests*

`itertools` and `functools` are the standard library's answer to "I could
write this loop by hand, but there's a named, tested, lazy-where-it-matters
version already." Knowing the names is most of the value — `groupby`,
`product`, `accumulate`, `reduce`, `partial`, `singledispatch` each replace
a small pile of boilerplate with one call, and reaching for the standard
name instead of reinventing it is exactly the kind of fluency an interview
is checking for.

The second half is decorators with configuration — `@retry(3)` instead of
`@retry` — which is a different shape from the plain decorators in module
11 and trips people up the first time.

Work in `exercises.py`. Check your work with `python -m labkit check 17`, or open the notebook (`lab.ipynb`) in Colab.

Each section below ends with **Test it** — the exact command for just that
piece, run from `units/python/17-functional-patterns/`.

## What you'll build

| Name | You'll implement | Shows up in real code as |
|---|---|---|
| `flatten` | `flatten` | Merging shards, batches, or paginated results into one sequence |
| `group_adjacent` | `group_adjacent` | The groupby gotcha itself — grouping only consecutive runs |
| `group_all` | `group_all` | The sort-then-group fix — the recipe you actually want most of the time |
| `all_pairs` | `all_pairs` | Every combination of two option sets — hyperparameter grids, test matrices |
| `running_total` | `running_total` | Cumulative sums/products without a hand-rolled loop |
| `product_of` | `product_of` | `reduce` for folding a binary op across a sequence |
| `make_multiplier` | `make_multiplier` | `partial` — pre-binding arguments without a wrapping `def` |
| `describe` | `describe`, three `@describe.register` overloads | `singledispatch` — dispatch on type without an `isinstance` chain |
| `retry` | `retry` | A configurable decorator — the `@retry(3)` shape |
| `call_counter` | `call_counter` | A plain decorator that attaches state to the function it wraps |

---

## 1. `itertools.chain`: flattening without nesting

```python
list(itertools.chain.from_iterable(iterables))
```

`chain.from_iterable` is lazy — it does not build the flattened list until
something consumes it — unlike `sum(iterables, [])`, which is a common but
genuinely `O(n²)` way to flatten (each `+` copies the accumulator so far).

```python
>>> flatten([[1, 2], [3], [4, 5]])
[1, 2, 3, 4, 5]
```

**In `exercises.py`:** `flatten`.

**Test it:** `python3 -m unittest test_exercises.TestFlatten -v`

## 2. `itertools.groupby`: consecutive only

```python
[(key, list(group)) for key, group in itertools.groupby(items)]
```

This is the itertools trap worth memorising: **`groupby` only merges a run
while the key keeps matching the item right before it.** It does not scan
ahead across the whole sequence.

```python
>>> group_adjacent(["a", "a", "b", "a"])
[('a', ['a', 'a']), ('b', ['b']), ('a', ['a'])]
```

Three groups, not two — the second `"a"` starts a fresh group because a
`"b"` came between it and the first run. The fix, when you want every
occurrence merged regardless of position, is to sort by the key first so
every match becomes adjacent before `groupby` ever sees it:

```python
ordered = sorted(items, key=key)
{k: list(group) for k, group in itertools.groupby(ordered, key=key)}
```

```python
>>> group_all(["a", "a", "b", "a"], key=lambda x: x)
{'a': ['a', 'a', 'a'], 'b': ['b']}
```

**In `exercises.py`:** `group_adjacent`, `group_all`.

**Test it:** `python3 -m unittest test_exercises.TestGroupAdjacent test_exercises.TestGroupAll -v`

## 3. `itertools.product` and `accumulate`

```python
>>> all_pairs([1, 2], ["x", "y"])
[(1, 'x'), (1, 'y'), (2, 'x'), (2, 'y')]
```

`product` is the Cartesian product, generated lazily — the shape behind
"every combination of these hyperparameters" or "every pair of test
fixtures." A nested list comprehension does the same job; `product` names
it and does not force materialising both inputs up front.

```python
>>> running_total([1, 2, 3, 4])
[1, 3, 6, 10]
```

`accumulate` defaults to running sum but takes any two-argument function —
`accumulate(numbers, operator.mul)` gives a running product with the exact
same call shape, no separate loop needed.

**In `exercises.py`:** `all_pairs`, `running_total`.

**Test it:** `python3 -m unittest test_exercises.TestAllPairs test_exercises.TestRunningTotal -v`

## 4. `functools.reduce` and `partial`

```python
functools.reduce(operator.mul, numbers, 1)
```

`reduce` folds a binary function across an iterable, carrying a running
result forward — `operator.mul` here instead of `lambda a, b: a * b`,
since the `operator` module's functions are both faster and more
readable at the call site than an equivalent lambda. The `1` initial
value is what makes an empty input return `1` instead of raising
`TypeError: reduce() of empty iterable with no initial value`.

```python
functools.partial(operator.mul, n)
```

`partial` pre-binds `operator.mul`'s first argument and returns a new
callable — no `def`, no lambda. It's a real object, not a closure, which
is why `partial` objects are picklable (usable with `multiprocessing`)
where an equivalent closure typically is not.

```python
>>> product_of([1, 2, 3, 4])
24
>>> product_of([])
1
>>> times_three = make_multiplier(3)
>>> times_three(4)
12
```

**In `exercises.py`:** `product_of`, `make_multiplier`.

**Test it:** `python3 -m unittest test_exercises.TestProductOf test_exercises.TestMakeMultiplier -v`

## 5. `functools.singledispatch`: dispatch by type

```python
@functools.singledispatch
def describe(obj):
    return f"a {type(obj).__name__}"        # the fallback

@describe.register
def _(obj: int):
    return f"the integer {obj}"
```

Each `@describe.register` adds a type-specific implementation without an
`if isinstance(obj, int): ... elif isinstance(obj, str): ...` chain
growing inside one function body. Adding support for a new type later
means adding a new `@describe.register`-decorated function, not editing
an existing one — the same "open for extension" property polymorphism
gives you with classes, here applied to plain functions and types you may
not control (you can register a handler for a built-in or third-party
type without subclassing it).

```python
>>> describe(42)
'the integer 42'
>>> describe("hi")
'a string of length 2'
>>> describe(3.14)
'a float'
```

**In `exercises.py`:** `describe` (the base case) and the three
`@describe.register` functions for `int`, `str`, `list`.

**Test it:** `python3 -m unittest test_exercises.TestDescribe -v`

## 6. Parameterised decorators: `@retry(3)` vs `@call_counter`

`call_counter` is a plain decorator — it takes the function directly:

```python
def call_counter(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        return func(*args, **kwargs)
    wrapper.calls = 0
    return wrapper
```

`retry(times)` needs configuration, so it has one more layer. `@retry(3)`
is not `retry` used as a decorator — it is **`retry(3)` called first**,
and *that return value* is the decorator applied to the function:

```python
def retry(times):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            ...
        return wrapper
    return decorator
```

Three layers, three purposes: `retry` takes the configuration,
`decorator` takes the function, `wrapper` runs on every call. Forgetting
the middle layer — writing `retry` as if it were `call_counter` — is the
standard mistake here, and it fails at decoration time with `TypeError:
'function' object is not callable` (or similar), since `@retry` would
apply `retry` itself as the decorator, passing it the function where it
expects `times`.

`functools.wraps(func)` on the inner `wrapper` matters more here than it
looks: without it, `wrapper.__name__` becomes `"wrapper"` for *every*
decorated function, which breaks logging, debuggers, and any other
decorator stacked on top that reads `__name__`.

```python
>>> @retry(3)
... def flaky():
...     ...
>>> flaky()  # retries internally, raises only if all 3 attempts fail
```

**In `exercises.py`:** `retry`, `call_counter`.

**Test it:** `python3 -m unittest test_exercises.TestRetry test_exercises.TestCallCounter -v`

---

## The exercises

| Function | What it teaches |
|---|---|
| `flatten` | `itertools.chain.from_iterable` — lazy flattening |
| `group_adjacent`, `group_all` | `groupby`'s consecutive-only behaviour, and the sort-first fix |
| `all_pairs` | `itertools.product` |
| `running_total` | `itertools.accumulate` |
| `product_of` | `functools.reduce` with an initial value |
| `make_multiplier` | `functools.partial` |
| `describe` | `functools.singledispatch` — type-based dispatch without an `if` chain |
| `retry` | A parameterised decorator — the extra factory layer |
| `call_counter` | A plain decorator that attaches state to the wrapper |

Every function in this table has a stub in `exercises.py` and test
coverage in `test_exercises.py` — a clean 1:1 mapping.

## Check your understanding

1. **Why does `itertools.groupby(["a", "a", "b", "a"])` produce three
   groups instead of two?**
   <details><summary>Answer</summary><code>groupby</code> only merges an
   item into the current group while its key matches the item
   immediately before it — it never looks ahead across the whole
   sequence. The second run of <code>"a"</code> is separated from the
   first by a <code>"b"</code>, so it starts a new group. Getting "all
   items with this key together" regardless of position requires sorting
   by the key first.</details>

2. **What's the practical difference between `functools.partial(f, x)`
   and `lambda *a: f(x, *a)`?**
   <details><summary>Answer</summary>They behave the same when called,
   but <code>partial</code> returns a real object (with
   <code>.func</code>, <code>.args</code> attributes) rather than a
   closure, and it's picklable — usable with
   <code>multiprocessing</code> or cached to disk — where an equivalent
   lambda generally is not.</details>

3. **Why does `@retry(3)` need an extra layer of nesting compared to a
   plain decorator like `@call_counter`?**
   <details><summary>Answer</summary><code>@retry(3)</code> calls
   <code>retry(3)</code> first, and that call must itself return the
   actual decorator function, because Python evaluates the parenthesised
   expression before applying <code>@</code>. A plain decorator receives
   the function directly as its only argument; a parameterised one
   receives its configuration first and returns a decorator that then
   receives the function.</details>

4. **What does `functools.wraps` actually fix, and what breaks without
   it?**
   <details><summary>Answer</summary>Without it, the wrapper function
   replaces the original's <code>__name__</code>, <code>__doc__</code>,
   and other metadata with its own generic values (e.g. every decorated
   function reports its name as <code>"wrapper"</code>). This breaks
   logging, debugging output, documentation tools, and any other
   decorator stacked on top that reads those attributes to make
   decisions.</details>

5. **When would you reach for `functools.singledispatch` instead of an
   `if isinstance(...) / elif isinstance(...)` chain?**
   <details><summary>Answer</summary>When you expect to add support for
   more types over time, or when you want to register a handler for a
   type you don't control (a built-in or a third-party class) without
   subclassing it. Each new type gets its own
   <code>@describe.register</code>-decorated function instead of another
   branch inside a single, ever-growing function body.</details>

## Interview questions this module answers

- *`groupby` isn't giving me the groups I expect — why?*
- *What's the difference between `functools.partial` and a lambda that
  does the same thing?*
- *Write a decorator that retries a flaky function N times.*
- *Why does `functools.wraps` matter?*
- *How would you avoid a long `isinstance` chain when handling several
  input types?*
