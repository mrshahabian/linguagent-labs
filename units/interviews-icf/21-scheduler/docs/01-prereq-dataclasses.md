# Prerequisite: dataclasses

The whole API is dataclasses. You do not have to write any, but you have to
read them fluently, and one of them (`StepPlan`) you construct every step.

## The basics

A dataclass turns a class-level list of annotated names into a real class with
an `__init__`, an `__eq__` and a `__repr__`:

```python
from dataclasses import dataclass

@dataclass
class PrefillOp:
    id: str
    tokens: int

op = PrefillOp("A", 6)
op = PrefillOp(id="A", tokens=6)     # same thing, and clearer
print(op)                            # PrefillOp(id='A', tokens=6)
op == PrefillOp("A", 6)              # True — field-by-field equality, free
```

That free `__eq__` matters more than it looks. It is why a test can say

```python
assert plan.prefill == [PrefillOp("A", 6), PrefillOp("B", 3)]
```

and compare structure rather than identity.

## `frozen=True`

```python
@dataclass(frozen=True)
class Request:
    id: str
    prompt_len: int
```

Frozen means assignment raises:

```python
r = Request("A", 6)
r.prompt_len = 7      # dataclasses.FrozenInstanceError
```

The API uses `frozen=True` for `Request`, `PrefillOp`, `EngineView` and
`KvEngineView` — everything the engine hands *you* — and a plain mutable
`@dataclass` for `StepPlan`, the one thing you hand *back*.

That asymmetry is the API telling you something: **you cannot fix the world by
editing the view.** Decrementing `view.max_work` is not a way to track your
budget; it is an exception. The only channel from you to the engine is the
returned `StepPlan`.

## `field(default_factory=list)` — and the trap it avoids

```python
from dataclasses import dataclass, field

@dataclass
class StepPlan:
    preempt: list[str] = field(default_factory=list)
    decode: list[str] = field(default_factory=list)
    prefill: list[PrefillOp] = field(default_factory=list)
```

Why not `preempt: list[str] = []`? Because a default value is evaluated once,
at class-definition time, and would then be *shared by every instance* — the
classic Python mutable-default bug. Every `StepPlan()` would append into the
same three lists, and your Level 1 plan would still contain Level 1's decodes
when Level 3c ran. `dataclass` refuses to let you do it:

```python
@dataclass
class Bad:
    xs: list = []     # ValueError: mutable default <class 'list'> ...
```

`default_factory=list` says "call `list()` fresh for each new instance".

The practical consequence for you: `StepPlan()` gives you three independent
empty lists, so the working style is

```python
plan = StepPlan()
plan.decode.append(request.id)
plan.prefill.append(PrefillOp(id=request.id, tokens=cost))
return plan
```

You can also build them up front and pass them in, which is equally fine:

```python
return StepPlan(decode=decode_ids, prefill=prefill_ops)
```

## Inheritance and default ordering

`KvEngineView` extends `EngineView` to add `kv_capacity`. There is a rule that
bites here: **a field with a default cannot be followed by one without.**
Subclass fields come after the parent's, so if the parent has any defaults,
every subclass field needs one too. That is why the API's views give
everything a default (`waiting: tuple[Request, ...] = ()`).

You will not have to solve this, but it explains why the code looks the way it
does — and it is a genuinely common interview follow-up question.

## Why tuples in the view, not lists

```python
waiting: tuple[Request, ...] = ()
```

`frozen=True` only stops *rebinding a field*. It does not stop mutating what
the field points at. If `waiting` were a list, `view.waiting.append(...)` and
`view.waiting.sort()` would both work and would both corrupt the engine's
state. A tuple has no `append` and no `sort`, so the immutability is real.

The practical consequence: **`sorted(view.waiting, ...)` works and returns a
new list; `view.waiting.sort()` raises.** Level 3c needs the first one.

## `replace()`

Occasionally useful for building test fixtures: makes a modified copy of a
frozen instance.

```python
from dataclasses import replace
urgent = replace(request, priority=9)
```

## What to take into the levels

- `StepPlan()` then `.append(...)` is the normal working style.
- `PrefillOp(id=..., tokens=...)` — pass by keyword, it reads better and you
  cannot swap the arguments by accident.
- The view is read-only, in a way that is enforced. Track budgets in local
  variables.
- `view.waiting` and `view.admitted` are tuples: iterate them, `sorted()` them,
  never mutate them.

## Drill

`drills/drills.py` Block A. Then check yourself:

1. Why does `StepPlan` use `field(default_factory=list)` rather than `= []`?
2. `view.waiting` is a tuple of frozen `Request`s. Which of these raise?
   `view.waiting.append(r)` / `sorted(view.waiting, key=...)` /
   `view.waiting[0].kv = 5` / `list(view.waiting).pop()`
3. You need "the admitted requests, most recently admitted first." Write it
   without mutating `view.admitted`.

<details>
<summary>Answers</summary>

1. A bare `[]` would be created once at class-definition time and shared by
   every `StepPlan` instance. `dataclass` raises `ValueError` rather than let
   you. `default_factory=list` calls `list()` per instance.
2. `.append` raises `AttributeError` (tuples have no append).
   `sorted(...)` is fine — it returns a new list.
   `view.waiting[0].kv = 5` raises `FrozenInstanceError`.
   `list(view.waiting).pop()` is fine — it pops from the fresh copy.
3. `list(reversed(view.admitted))` or `view.admitted[::-1]`. Both build a new
   sequence; neither touches the view. (This is exactly the Level 3b victim
   order.)

</details>
