# 10 — The data model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/python/10-data-model/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/python/data-model?utm_source=github&utm_medium=labs)

*~50 minutes · 34 tests*

Python has no interfaces and almost no keywords for behaviour. Instead, the
language asks your object questions by calling methods with dunder names.
`len(x)` calls `x.__len__()`. `a + b` calls `a.__add__(b)`. `for y in x` calls
`x.__iter__()`. That is nearly the whole language design.

Knowing this well is the difference between "writes Python" and "knows
Python", and it is the thing senior interviews probe first — usually as
*"make this object work with `sorted()`"* or *"why does `in` work here?"*

Work in `exercises.py`. Check your work with `python -m labkit check 10`, or open the notebook (`lab.ipynb`) in Colab.

Each section below ends with **Test it** — the exact command for just that
piece, run from `units/python/10-data-model/`. No need to finish the whole
file before checking your work.

## What you'll build

Seven classes in `exercises.py`, in this order. Keep this table open as you
go — each section below tells you which row it's building:

| Class | You'll implement | Shows up in real code as |
|---|---|---|
| `Vec` | `__repr__`, `__eq__`, `__hash__`, `__add__`/`__sub__`/`__mul__`, `__iter__` | Any small value type — a shape, a coordinate, an RGB pixel. The same dunders are why `torch.Size` compares and hashes sanely |
| `Bag` | `add`, `__len__`, `__contains__`, `__iter__` | A counting structure — token/word frequency, a histogram, dedup with multiplicities (this is the shape of the standard library's `collections.Counter`) |
| `Version` | `_padded`, `__eq__`, `__lt__`, `__hash__` | Dependency resolution, comparing checkpoint or package versions in a registry |
| `RateLimiter` | `__call__` | Throttling — capping requests per user against an inference API or LLM endpoint |
| `collecting` | `__enter__`, `__exit__` | Scoped state management — the same shape as `torch.no_grad()`, a DB transaction, or temporarily redirecting logs |
| `Positive` / `Config` | `__get__`, `__set__`, `__set_name__` | Validated attributes — the descriptor protocol behind `property`, and how a config object can reject a bad learning rate the moment it's assigned, not just at construction |
| `Lazy` | `__getattr__` | Deferred initialisation — loading a large checkpoint or dataset only on first access, not eagerly at construction |

---

## 1. Representation: `__repr__` before `__str__`

```python
def __repr__(self):
    return f"Vec({self.x}, {self.y})"
```

Both are just methods Python calls for you, on different triggers. `repr(x)` —
and anywhere Python needs an unambiguous stand-in, like the REPL or an item
inside `[Vec(1, 2)]` — calls `x.__repr__()`. `str(x)`, `print(x)` and
f-strings call `x.__str__()`:

```python
def __str__(self):
    return f"({self.x}, {self.y})"
```

`__repr__` is for you — unambiguous, ideally `eval`-able. `__str__` is for
users, so it can drop information `__repr__` keeps. **Define `__repr__`
first**: if `__str__` is missing Python falls back to `__repr__`, but not the
reverse:

```python
>>> v = Vec(1, 2)
>>> repr(v)          # calls __repr__ directly
'Vec(1, 2)'
>>> str(v)            # no __str__ defined -> falls back to __repr__
'Vec(1, 2)'
>>> [v]                # containers print their items with repr(), not str()
[Vec(1, 2)]
```

Now flip it around: a class with only `__str__` and no `__repr__` inherits
`object`'s default, and prints as `<Vec object at 0x7f8f2c1a5d90>` inside a
list — which is exactly where you actually need a readable form.

**In `exercises.py`:** `Vec.__repr__` is the only method this section asks
for — there's no separate `__str__` stub; the demo above works precisely
*because* `__str__` is missing.

You'll also notice `Vec` declares `__slots__ = ("x", "y")`. That's not part
of this module's material — it's a memory/speed optimisation that replaces
each instance's `__dict__` with fixed storage for exactly those attributes,
covered properly in `units/python/14-performance/README.md` (§7). Here it's
just marking that a vector should only ever have `x` and `y`.

**Test it:** `python3 -m unittest test_exercises -k repr -v`

## 2. Equality and hashing travel together

### Equality: `__eq__`

`a == b` calls `a.__eq__(b)`. `object`'s default implementation compares
identity (`a is b`), so two separately built objects with identical fields
are `!=` until you say otherwise:

```python
>>> class Point:
...     def __init__(self, x, y): self.x, self.y = x, y
...
>>> Point(1, 2) == Point(1, 2)   # default __eq__: same object or bust
False
```

Override it to compare by value instead:

```python
def __eq__(self, other):
    if not isinstance(other, Point):
        return NotImplemented
    return (self.x, self.y) == (other.x, other.y)
```

```python
>>> Point(1, 2) == Point(1, 2)
True
>>> Point(1, 2) == "not a point"
False
```

Returning `NotImplemented` — not raising, not `False` — for a type you don't
recognise matters: it tells Python to ask the *other* object whether it knows
how to compare, and only fall back to "not equal" (or raise `TypeError`, for
operators like `<`) once nothing does. `Point(1, 2) == "not a point"` above
works out to `False` this way, cleanly, rather than crashing on
`"not a point".x`.

**In `exercises.py`:** `Vec.__eq__` is this subsection's target (`Version`'s
`__eq__` needs `_padded` too, so it's covered as part of §5 instead).

**Test it:** `python3 -m unittest test_exercises -k comparison -v`

This checks only `Vec.__eq__`'s `NotImplemented` behaviour — nothing else in
`Vec` needs to be finished first.

### Hashing travels with it

`hash(x)` is just a number — a "fingerprint" of a value, and equal values
always give the same one:

```python
>>> hash(5)
5
>>> hash("cat")
-2041914558
>>> hash("cat") == hash("cat")
True
```

`dict` and `set` use that number to jump straight to where an item *should*
be, instead of checking every item one by one — that's what makes `x in
some_set` fast regardless of how big `some_set` is. But two different values
can produce the same number by chance (a "collision"), so the number only
narrows the search to one spot; `==` is still what confirms a match.

That means the two must agree, or lookups quietly break. Watch what happens
if `__eq__` says two objects are equal but `__hash__` disagrees:

```python
>>> class Bad:
...     def __init__(self, n): self.n = n
...     def __eq__(self, other): return True   # everything is "equal"
...     def __hash__(self): return id(self)    # but every hash differs
...
>>> a, b = Bad(1), Bad(2)
>>> a == b
True
>>> s = {a}
>>> b in s
False   # wrong: b == a, but hash(b) != hash(a), so `in` checked the wrong spot
```

**The contract: if `a == b` then `hash(a) == hash(b)`.** (Not the reverse —
unequal objects sharing a hash is just a slower lookup, not a bug.)

Python's default `__eq__`/`__hash__` compare by identity (same object, hash
from `id()`), which trivially satisfies this. The moment you write your own
`__eq__`, Python can no longer be sure that guarantee still holds — so it
sets `__hash__ = None` for you automatically, and using the object in a set
or as a dict key raises instead of silently misbehaving:

```python
>>> class Point:
...     def __init__(self, x, y): self.x, self.y = x, y
...     def __eq__(self, other): return (self.x, self.y) == (other.x, other.y)
...
>>> hash(Point(1, 2))
TypeError: unhashable type: 'Point'
```

The fix, if your object is immutable: define `__hash__` too, over the same
fields `__eq__` compares.

```python
def __hash__(self):
    return hash((self.x, self.y))
```

`Vec` and `Version` in this module both need this pairing.

**In `exercises.py`:** `Vec.__eq__` and `Vec.__hash__` are the simple case —
compare/hash `(self.x, self.y)` directly. `Version.__eq__` and
`Version.__hash__` are the subtle case: `__eq__` needs `_padded` (§5) so
`"1.0"` and `"1.0.0"` compare component-by-component; `__hash__` does *not*
need `_padded` or `__eq__` — it works straight off `self.parts`, stripping
trailing zeros (see the comment above the stub), which is deliberate so it
doesn't depend on anything else in the class being finished first.

**Test it:** `python3 -m unittest test_exercises -k hash -v`

This isolates `__hash__` on both classes. The `Vec` half also needs
`Vec.__eq__` to pass, since it puts `Vec`s in a `set` — but `__eq__` sits
right above `__hash__` in the file, so you'll normally have it already.
`Version.__eq__` on its own isn't tested in isolation; it's checked together
with ordering in §5 (`test_missing_components_are_zero`), since both
`__eq__` and `__lt__` share `_padded`.

## 3. Truthiness: `__bool__`, falling back to `__len__`

`bool(obj)` — what `if obj:` calls — checks in order: `__bool__` if it's
defined, else `__len__() != 0`, else `True`. `__bool__` wins whenever both
are present:

```python
>>> class Weird:
...     def __len__(self): return 0       # would normally mean falsy
...     def __bool__(self): return True   # but this takes priority
...
>>> bool(Weird())
True
```

An object with **neither** method defined is always truthy, no matter what
it conceptually holds — the bug behind `if response:` silently passing for
an empty custom response type that never defined either one.

`Bag` defines `__len__` but not `__bool__`, so truthiness follows for free
the moment `__len__` works — no separate `__bool__` needed:

```python
>>> bool(Bag())       # __bool__ absent -> falls back to __len__() == 0
False
>>> bool(Bag("a"))    # __len__() == 1
True
```

**In `exercises.py`:** `Bag.__len__` is the only method this section asks
for — there's no `Bag.__bool__` stub, by design, exactly like `Vec` had no
`__str__` stub in §1.

**Test it:** `python3 -m unittest test_exercises -k falsy -v`

## 4. Iteration has an old path and a new one

The modern protocol is `__iter__` returning an iterator. But Python also
honours a legacy one: if there is no `__iter__` but there **is** a
`__getitem__` taking integers from 0 upward, `for` will call it until
`IndexError`:

```python
class Old:
    def __getitem__(self, i):
        return [10, 20, 30][i]     # no __iter__ at all

for x in Old():   # Python tries Old()[0], [1], [2], ... until IndexError
    print(x)      # 10 20 30
```

Worth recognising in old code; don't write it — neither exercise in this
module uses it. Both `Bag` and `Vec` use the modern `__iter__` protocol.

`in` uses `__contains__` if defined, otherwise falls back to iterating:

```python
"a" in bag   # with __contains__: one dict lookup, O(1)
             # without it: scans every item via __iter__, O(n)
```

**In `exercises.py`:** `Bag.__iter__` and `Bag.__contains__` are this
section's core targets. `Vec.__iter__` belongs here too, even though it's a
different class — it's what makes `x, y = Vec(7, 8)` unpacking work, since
unpacking is just iteration under the hood.

**Test it:** `python3 -m unittest test_exercises -k contains -k iterates -k unpacks -v`

## 5. Ordering: one method plus a decorator

Sorting needs only `__lt__`. For the full set (`<`, `<=`, `>`, `>=`), write
`__lt__` and `__eq__` and let `functools.total_ordering` fill in the rest:

```python
@functools.total_ordering
class Version:
    def __eq__(self, other): ...
    def __lt__(self, other): ...
```

```python
>>> Version("1.0") < Version("2.0")
True
>>> Version("1.0") <= Version("2.0")   # generated by total_ordering
True
```

Returning `NotImplemented` (not raising, not `False`) when the other operand
is a type you don't handle lets Python try the reflected operation and then
raise a proper `TypeError`, instead of a confusing `AttributeError` from
reaching into an operand that has no `.parts`:

```python
>>> Version("1.0") < 5
TypeError: '<' not supported between instances of 'Version' and 'int'
```

Both `__eq__` and `__lt__` have to compare `Version`s of different lengths
fairly — `"1.0"` is `(1, 0)`, `"1.0.1"` is `(1, 0, 1)`. `_padded` is the
helper they share: it pads the shorter tuple with zeros so the comparison
lines up position by position, which is what makes `Version("1.0") ==
Version("1.0.0")` and `Version("1.0") < Version("1.0.1")` both come out
right.

**In `exercises.py`:** `Version._padded`, `Version.__eq__` and
`Version.__lt__` are this section's targets. `_padded` is shared with §2's
`__eq__` — implement it once here, and the equality half from §2 uses the
same code.

**Test it:** `python3 -m unittest test_exercises -k sorts -k total_ordering -k missing_components -v`

This deliberately excludes the two hash tests that also live in
`TestVersion` (§2's material) — running the whole class would pull those in
too and make `Version.__hash__` look like a requirement for ordering, which
it isn't.

## 6. Callables

`__call__` makes an instance behave like a function while keeping state:

```python
>>> limiter = RateLimiter(limit=2)
>>> limiter("a")   # calls limiter.__call__("a")
True
>>> limiter("a")
True
>>> limiter("a")   # third call for "a" - over the limit
False
```

This is how PyTorch modules work — `model(x)` is `model.__call__(x)`, which
runs hooks and then `forward`. Worth being able to say in an interview.

**In `exercises.py`:** `RateLimiter.__call__` is the only method here.

**Test it:** `python3 -m unittest test_exercises.TestRateLimiter -v`

## 7. Context managers

`__enter__` returns the value bound by `as`; `__exit__(exc_type, exc, tb)`
runs on the way out **whether or not** an exception occurred:

```python
>>> with collecting(suppress=ValueError) as items:
...     items.append(1)
...     raise ValueError("boom")
...
>>> items
[1]
```

`__exit__` ran despite the `raise`, and returning a truthy value from it
**suppressed** the exception — a real footgun if you `return True` by
accident when you meant to let it propagate.

**In `exercises.py`:** `collecting.__enter__` and `collecting.__exit__`.

**Test it:** `python3 -m unittest test_exercises.TestCollecting -v`

## 8. Descriptors

A descriptor is any object defining `__get__`/`__set__`, used as a *class*
attribute. It intercepts attribute access on instances. This is the machinery
behind `property`, `classmethod`, and `nn.Parameter` registration.

```python
class Positive:
    def __set_name__(self, owner, name):
        self.name = name
    def __get__(self, obj, objtype=None):
        return obj.__dict__[self.name]
    def __set__(self, obj, value):
        if value <= 0:
            raise ValueError(f"{self.name} must be positive, got {value}")
        obj.__dict__[self.name] = value
```

`__set_name__` is what lets a descriptor learn the attribute name it was
assigned to, without you repeating it — which is why the error below can
name `height` on its own:

```python
>>> Config(width=10, height=-1)
ValueError: height must be positive, got -1
```

**In `exercises.py`:** `Positive.__get__`, `Positive.__set__` and
`Positive.__set_name__`. Note the code above is close to the full
implementation, not just a sketch — descriptors are the one genuinely
unfamiliar protocol in this module, so this section leans more on a worked
example than the others.

**Test it:** `python3 -m unittest test_exercises.TestDescriptor -v`

## 9. `__getattr__` is the fallback, `__getattribute__` is the intercept

`__getattr__` runs **only when normal lookup fails** — cheap and safe, ideal
for lazy loading and proxies:

```python
>>> lazy = Lazy(value=lambda: 42)
>>> lazy.build_count
0
>>> lazy.value          # not in __dict__ yet -> __getattr__ builds and caches it
42
>>> lazy.value          # now in __dict__ -> normal lookup finds it, __getattr__ skipped
42
>>> lazy.build_count    # only incremented once
1
```

`__getattribute__` runs on **every** access, including `self.x` inside
itself, so a naive implementation recurses infinitely. Reach for
`__getattr__` unless you genuinely need the other.

**In `exercises.py`:** `Lazy.__getattr__` is the only method here.

**Test it:** `python3 -m unittest test_exercises.TestLazy -v`

---

## The exercises

| Function | What it teaches |
|---|---|
| `Vec` | `__repr__`; the eq/hash contract; operator overloading; unpacking via `__iter__` |
| `Bag` | Truthiness falling back to `__len__`; `__contains__` vs. falling back to iteration |
| `Version` | `total_ordering` from `__eq__` + `__lt__`; equal values must hash equal |
| `RateLimiter` | `__call__`; state held per-instance, keyed per-call |
| `collecting` | `__enter__`/`__exit__`; suppressing one exception type, not all |
| `Positive` / `Config` | The descriptor protocol; `__set_name__` |
| `Lazy` | `__getattr__` as a fallback, not an intercept |

Two `Bag` methods don't belong to any numbered section above, because
they're plain support code rather than a data-model protocol being taught:
`Bag.add` (mutates `_counts`; covered by `test_add`) and `Bag.__repr__`
(covered by no test at all — nothing breaks if you leave it
`raise NotImplementedError`, but `repr(some_bag)` will crash until you
write it, so it's worth filling in anyway).

## Check your understanding

1. **Why does defining `__eq__` set `__hash__` to `None` automatically?**
   <details><summary>Answer</summary>Python's default identity-based
   <code>__eq__</code>/<code>__hash__</code> automatically satisfy the
   contract that equal objects hash equal. Once you override
   <code>__eq__</code> to compare by value, that guarantee is gone unless
   you also define a matching <code>__hash__</code> — so Python disables
   the old one rather than leave it silently inconsistent.</details>

2. **Why define `__repr__` before `__str__`, if you can only write
   one?**
   <details><summary>Answer</summary>If <code>__str__</code> is missing,
   Python falls back to <code>__repr__</code> — but not the reverse. A
   class with only <code>__str__</code> prints as
   <code>&lt;Vec object at 0x...&gt;</code> inside a container, which is
   exactly where you need a readable form.</details>

3. **`Bag` defines `__len__` but not `__bool__`. What makes an empty
   `Bag` falsy?**
   <details><summary>Answer</summary><code>bool(obj)</code> falls back to
   <code>__len__() != 0</code> when <code>__bool__</code> is absent — an
   empty container (<code>__len__</code> returns 0) is falsy for
   free.</details>

4. **`in` works even without `__contains__`. Why define it anyway?**
   <details><summary>Answer</summary>Without <code>__contains__</code>,
   <code>in</code> falls back to iterating the whole object — O(n).
   Defining it directly turns that into whatever you make it, O(1) for a
   dict-backed structure like <code>Bag</code>.</details>

5. **`Version` only implements `__eq__` and `__lt__`. Where do `<=`,
   `>`, and `>=` come from?**
   <details><summary>Answer</summary><code>functools.total_ordering</code>
   fills in the rest from just those two — sorting and the full
   comparison set both only need <code>__lt__</code> plus
   <code>__eq__</code>.</details>

6. **What's the actual difference between `__getattr__` and
   `__getattribute__`, and why is a naive `__getattribute__`
   dangerous?**
   <details><summary>Answer</summary><code>__getattr__</code> runs only
   when normal lookup fails; <code>__getattribute__</code> runs on
   *every* attribute access, including <code>self.x</code> inside itself
   — a naive implementation that reads <code>self.anything</code>
   recurses infinitely.</details>

## Interview questions this module answers

- *Why does defining `__eq__` break my object in a set?*
- *What is the difference between `__str__` and `__repr__`?*
- *How would you make this class work with `sorted()` / `in` / `with`?*
- *What does `model(x)` actually call in PyTorch?*
- *How does `@property` work under the hood?*
- *Why does `__getattribute__` recurse and `__getattr__` not?*
