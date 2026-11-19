# 16 — Typing, dataclasses and protocols

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/python/16-typing-and-dataclasses/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/python/typing-and-dataclasses?utm_source=github&utm_medium=labs)

*~45 minutes · 31 tests*

Type hints do not change how Python runs — they are erased at import time
and enforced by nothing at runtime. What they change is how much of your
design a reader (or a type checker, or an interviewer skimming your
screen-share) can see without running the code. `@dataclass` and `Protocol`
are the two tools that turn hints into actual leverage: less boilerplate for
the classes that are just data, and structural interfaces that do not need
inheritance to type-check.

This is the module that makes mid-to-senior code reviews go faster —
"why not just use a dataclass here" and "does this need to inherit from
anything" are both answered below.

Work in `exercises.py`. Check your work with `python -m labkit check 16`, or open the notebook (`lab.ipynb`) in Colab.

Each section below ends with **Test it** — the exact command for just that
piece, run from `units/python/16-typing-and-dataclasses/`.

## What you'll build

| Name | You'll implement | Shows up in real code as |
|---|---|---|
| `safe_divide` | `safe_divide` | Modelling failure as `Optional[T]` instead of an exception the caller must remember to catch |
| `Stack` | `push`, `pop`, `peek`, `is_empty`, `__len__` | A generic container a type checker can specialise per call site |
| `first_match` | `first_match` | A generic helper whose element type flows from input to predicate |
| `Point` | `distance_to` | An immutable, hashable value type — `frozen=True` |
| `SemVer` | `__str__` | Auto-generated ordering, with one field excluded from comparison |
| `sort_versions` | `sort_versions` | Sorting on a dataclass's generated `__lt__` |
| `ShoppingCart` | `add_item`, `total_items` | `default_factory` — the fix for the mutable-default-argument bug |
| `PositiveInt` | `__post_init__` | Validating a dataclass after construction |
| `Circle`, `Rectangle` | `area` | Two unrelated classes that satisfy the same `Protocol` |
| `total_area` | `total_area` | A function typed against a `Protocol`, not a base class |
| `satisfies_supports_area` | `satisfies_supports_area` | `isinstance` against a `@runtime_checkable` `Protocol` |

---

## 1. Type hints are erased, not enforced

```python
def safe_divide(a: float, b: float) -> Optional[float]:
    if b == 0:
        return None
    return a / b
```

Nothing stops you from calling `safe_divide("x", "y")` — Python never checks
the annotation at runtime. The hint exists for two audiences: a static
checker (`mypy`, `pyright`) run separately, and a human reading the
signature. `Optional[float]` (spelled `float | None` since 3.10) is doing
real work here regardless: it tells the caller **failure is a value they
must check**, not an exception they might forget to catch.

```python
>>> safe_divide(6, 3)
2.0
>>> safe_divide(1, 0)
>>> # None - printed as nothing at the REPL
```

**In `exercises.py`:** `safe_divide`.

**Test it:** `python3 -m unittest test_exercises.TestSafeDivide -v`

## 2. Generics: one definition, many types

```python
T = TypeVar("T")

class Stack(Generic[T]):
    def push(self, item: T) -> None: ...
    def pop(self) -> T: ...
```

`Generic[T]` is what lets `Stack[int]` and `Stack[str]` be *different types*
to a checker, both built from one class definition. At runtime `T` is inert
— `Stack()` behaves identically no matter what you told the checker it
holds. The payoff is entirely static: `stack.push("x")` on something typed
`Stack[int]` is flagged before the code runs, the same way a wrong function
argument would be.

`first_match` shows the same `TypeVar` tying two parameters together:

```python
def first_match(items: Iterable[T], predicate: Callable[[T], bool]) -> Optional[T]:
```

A checker infers `T` from `items` and then requires `predicate` to accept
that same `T` — passing a `list[int]` with a predicate typed for `str`
is an error, without either parameter needing to name the other's type.

```python
>>> stack = Stack()
>>> stack.push(1); stack.push(2)
>>> stack.pop()
2
>>> first_match([1, 3, 4, 6], lambda x: x % 2 == 0)
4
```

**In `exercises.py`:** `Stack.push`, `Stack.pop`, `Stack.peek`,
`Stack.is_empty`, `Stack.__len__`, `first_match`.

**Test it:** `python3 -m unittest test_exercises.TestStack test_exercises.TestFirstMatch -v`

## 3. `@dataclass(frozen=True)`: immutable and hashable

```python
@dataclass(frozen=True)
class Point:
    x: float
    y: float
```

One decorator generates `__init__`, `__repr__`, `__eq__` (by field values),
and — because `frozen=True` — a `__setattr__` that raises, plus `__hash__`.
That last part is the detail worth stating out loud: **a plain
`@dataclass` is unhashable by default the moment you define `__eq__`**
(same rule as hand-written classes — see module 10), but `frozen=True`
tells the generator the instance cannot change, so hashing by field values
is safe and it adds `__hash__` back.

```python
>>> p1, p2 = Point(0, 0), Point(3, 4)
>>> p1.distance_to(p2)
5.0
>>> p1.x = 99
Traceback (most recent call last):
    ...
dataclasses.FrozenInstanceError: cannot assign to field 'x'
>>> {p1, Point(0, 0)}          # hashable - the duplicate collapses
{Point(x=0, y=0)}
```

**In `exercises.py`:** `Point.distance_to`.

**Test it:** `python3 -m unittest test_exercises.TestPoint -v`

## 4. `order=True` and `field(compare=False)`

```python
@dataclass(order=True)
class SemVer:
    major: int
    minor: int
    patch: int
    prerelease: str = field(default="", compare=False)
```

`order=True` generates `__lt__`, `__le__`, `__gt__`, `__ge__` by comparing
fields **in declaration order**, as a tuple. `field(compare=False)` pulls
`prerelease` out of that tuple — and out of `__eq__` too — so two versions
that differ only in prerelease tag compare equal and sort adjacently.
Without it, `order=True` alone would try to compare `prerelease` strings
after the version numbers already decided the ordering — usually harmless,
but here it is explicitly not the intent, so it is excluded.

```python
>>> SemVer(1, 2, 3, "rc1") == SemVer(1, 2, 3, "rc2")
True
>>> sort_versions([SemVer(2, 0, 0), SemVer(1, 5, 0), SemVer(1, 0, 0)])
[SemVer(major=1, minor=0, patch=0, ...), SemVer(major=1, minor=5, patch=0, ...), SemVer(major=2, minor=0, patch=0, ...)]
```

**In `exercises.py`:** `SemVer.__str__`, `sort_versions`.

**Test it:** `python3 -m unittest test_exercises.TestSemVer -v`

## 5. `default_factory`: the mutable-default trap, fixed

```python
@dataclass
class ShoppingCart:
    items: list = field(default_factory=list)
```

`items: list = []` would bind **one list object**, created once when the
class body runs, shared by every instance that does not pass its own —
the same bug as `def f(x, items=[])`. Dataclasses actively forbid this: a
bare mutable literal default raises `ValueError` at class-definition time,
not silently at first use. `field(default_factory=list)` instead stores a
zero-argument callable, invoked fresh for every instance.

```python
>>> a, b = ShoppingCart(), ShoppingCart()
>>> a.add_item("apple")
>>> a.total_items(), b.total_items()
(1, 0)
>>> a.items is b.items
False
```

**In `exercises.py`:** `ShoppingCart.add_item`, `ShoppingCart.total_items`.

**Test it:** `python3 -m unittest test_exercises.TestShoppingCart -v`

## 6. `__post_init__`: validation after construction

```python
@dataclass
class PositiveInt:
    value: int

    def __post_init__(self) -> None:
        if self.value <= 0:
            raise ValueError(...)
```

The generated `__init__` only assigns fields — it has no way to express
"and also check this." `__post_init__` runs once, immediately after that
assignment, and is the standard place to put validation a field
declaration alone cannot: cross-field checks, range checks, normalising a
string's case. Raising here means an invalid instance is never
constructed — there is no window where `PositiveInt(-1)` exists half-built.

```python
>>> PositiveInt(5).value
5
>>> PositiveInt(-1)
Traceback (most recent call last):
    ...
ValueError: value must be positive, got -1
```

**In `exercises.py`:** `PositiveInt.__post_init__`.

**Test it:** `python3 -m unittest test_exercises.TestPositiveInt -v`

## 7. `Protocol`: typing by shape, not by ancestry

```python
@runtime_checkable
class SupportsArea(Protocol):
    def area(self) -> float: ...
```

`Circle` and `Rectangle` below share **no base class** — nothing inherits
from `SupportsArea`. It still type-checks to pass either one to
`total_area(shapes: Iterable[SupportsArea])`, because `Protocol` describes
a *shape* (anything with a no-argument `area() -> float`), not a lineage.
This is what "structural typing" means, and it is how Python narrows the
gap with languages that have interfaces natively — you get the type
safety without forcing unrelated classes into an inheritance hierarchy
just so a function can accept both.

`@runtime_checkable` is the detail that makes `isinstance(obj, SupportsArea)`
work at all. Without it, `Protocol` is a static-only construct and
`isinstance` against it raises `TypeError`. Even with it, `isinstance` only
checks that the *names* exist — it does not verify `area` is callable with
the right signature or actually returns a `float`.

```python
>>> total_area([Circle(1), Rectangle(2, 3)])
9.141592653589793
>>> satisfies_supports_area(Circle(1))
True
>>> satisfies_supports_area(object())
False
```

**In `exercises.py`:** `Circle.area`, `Rectangle.area`, `total_area`,
`satisfies_supports_area`.

**Test it:** `python3 -m unittest test_exercises.TestSupportsAreaProtocol -v`

---

## The exercises

| Function / class | What it teaches |
|---|---|
| `safe_divide` | `Optional[T]` as a failure-modelling return type |
| `Stack`, `first_match` | `Generic[T]` and `TypeVar` — zero runtime cost, real static leverage |
| `Point` | `frozen=True` gives immutability and restores `__hash__` |
| `SemVer`, `sort_versions` | `order=True`; `field(compare=False)` |
| `ShoppingCart` | `field(default_factory=...)` fixes the mutable-default bug |
| `PositiveInt` | `__post_init__` for validation |
| `Circle`, `Rectangle`, `total_area`, `satisfies_supports_area` | `Protocol` and structural typing; `@runtime_checkable` |

Every function in this table has a stub in `exercises.py` and test coverage
in `test_exercises.py` — a clean 1:1 mapping.

## Check your understanding

1. **Does `def f(x: int) -> int` stop you from calling `f("hello")`?**
   <details><summary>Answer</summary>No. Type hints are not enforced at
   runtime by the Python interpreter — they are checked only by a separate
   static tool (mypy, pyright) you run yourself, or ignored entirely.
   Calling <code>f("hello")</code> runs exactly as if the hint were not
   there, unless something inside <code>f</code> happens to fail on a
   string.</details>

2. **Why does `@dataclass(frozen=True)` restore `__hash__`, when a plain
   `@dataclass` with a custom `__eq__` does not have one?**
   <details><summary>Answer</summary>Defining <code>__eq__</code> normally
   disables the default <code>__hash__</code>, because a hash computed
   from mutable fields could go stale the moment a field changes, breaking
   dict/set invariants. <code>frozen=True</code> guarantees the instance
   can never change after construction, so hashing by field values is safe
   again, and the dataclass machinery adds <code>__hash__</code> back.</details>

3. **What does `field(default_factory=list)` fix that `items: list = []`
   would break?**
   <details><summary>Answer</summary><code>items: list = []</code> would
   create one list object when the class body executes, shared by every
   instance that doesn't pass its own — mutating it on one instance would
   affect all of them. <code>default_factory</code> stores a callable that
   dataclasses invokes fresh for each new instance, so every instance gets
   its own list.</details>

4. **What's the difference between `Protocol` and an abstract base class
   (`ABC`) for describing "anything with an `area()` method"?**
   <details><summary>Answer</summary>An ABC requires the concrete classes
   to explicitly inherit from it. A <code>Protocol</code> is structural —
   any class with a matching <code>area()</code> method satisfies it
   automatically, with no inheritance and no change to the concrete
   classes at all. This matters most when you don't control those
   classes, or don't want to force unrelated types into one hierarchy.</details>

5. **Why does `isinstance(obj, SomeProtocol)` raise `TypeError` unless the
   protocol is decorated `@runtime_checkable`?**
   <details><summary>Answer</summary><code>Protocol</code> is designed as
   a static-typing tool by default — it isn't meant to add runtime
   overhead to every class definition. <code>@runtime_checkable</code>
   opts a specific protocol into supporting <code>isinstance</code>,
   which then checks only for the presence of the named
   methods/attributes, not their signatures or return types.</details>

## Interview questions this module answers

- *Do type hints actually enforce anything?*
- *Why would you reach for `@dataclass` instead of writing `__init__` by
  hand?*
- *What's the mutable default argument bug, and how does `dataclass` avoid
  it?*
- *When would you use a `Protocol` instead of an abstract base class?*
- *Why is `frozen=True` the thing that makes a dataclass hashable?*
