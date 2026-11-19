# Prerequisite: sorting, keys and stability

This is the highest-leverage prerequisite in the whole course. Level 3c is
essentially "can you express two different orderings correctly", and a
mis-specified sort key is the most common way to fail it.

## `key=` computes a value; sorting compares those values

```python
sorted(requests, key=lambda r: r.prompt_len)
```

Read it as: *for each element, compute this; sort by the results*. The key
function runs once per element, not once per comparison.

`sorted()` returns a new list and never touches the input — which is exactly
what you need for `view.waiting`, a tuple you are not allowed to mutate.

## Tuple keys: sort by A, then by B

Tuples compare left to right: compare the first items; only if they are equal,
compare the second; and so on.

```python
sorted(items, key=lambda x: (x.priority, x.arrival))
```

= "by priority ascending; ties broken by arrival ascending".

This is the single most useful sorting idiom in interview code, and it appears
in some form in almost every ICF task.

## Descending: negate the number

There is no per-field `reverse=`. `reverse=True` flips *the whole key*, which
is wrong the moment you have a tiebreaker that should stay ascending:

```python
# WRONG: reverses the tiebreaker too
sorted(items, key=lambda x: (x.priority, x.name), reverse=True)

# RIGHT: priority descending, name ascending
sorted(items, key=lambda x: (-x.priority, x.name))
```

Negation only works on numbers. For a descending string sort with an ascending
tiebreaker you need two passes (see "stability" below).

The Ledger-style `top(n)` from a warm-up drill is the canonical form:

```python
sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))   # total desc, name asc
```

Practise until you can type that without thinking. It is worth real minutes.

## Stability, and why Level 3c depends on it

Python's sort is **stable**: elements that compare equal keep their original
relative order.

This is not a curiosity here — it is load-bearing. The Level 3c rule reads:

> Admission order: admit highest priority first, tiebreaker by oldest.
> `view.waiting` is already in arrival order, so a stable sort by descending
> `priority` keeps oldest-first within a tier.

Which means this is enough:

```python
candidates = sorted(view.waiting, key=lambda r: -r.priority)
```

You do **not** need `(-r.priority, r.arrival_index)` — and you could not write
it anyway, because `Request` has no arrival field. The queue order *is* the
arrival order, and stability preserves it inside each priority tier.

```python
# waiting is [LOW(p0), MID(p1), OTHER(p1)]
sorted(waiting, key=lambda r: -r.priority)
# -> [MID(p1), OTHER(p1), LOW(p0)]
#     MID before OTHER because it was before it in the input
```

Stability is also why "sort by the secondary key first, then by the primary
key" works as a two-pass technique when you cannot express the key as one
tuple.

## `min` / `max` take the same `key`

```python
victim = min(active, key=lambda r: (r.priority, -rank[r.id]))
```

`min` with a key is often clearer than sorting and taking `[0]`, and it is
O(n) rather than O(n log n) — though at these sizes that is irrelevant and
clarity wins.

Note the shape of that key, because it is the Level 3c victim rule verbatim:

- `r.priority` ascending → lowest priority is the victim
- `-rank[r.id]` → among equal priorities, the *largest* rank (most recently
  admitted) sorts first

`rank` there is a dict built from the view:

```python
rank = {r.id: i for i, r in enumerate(view.admitted)}
```

`view.admitted` is in admission order, so its index *is* recency. Building an
index dict from `enumerate()` is a small habit worth having; it turns "where
was this in the list" from an O(n) `.index()` call into a lookup, and it keeps
working after you start removing things from a local copy of the list.

## Reversing without sorting

When the order you want is already the list's order backwards, do not invent a
key:

```python
# Level 3b: "preempt the most recently admitted request"
survivors = list(view.admitted)
victim = survivors.pop()        # last = most recently admitted
```

`list(...)` first, because `view.admitted` is a tuple and `pop` would fail —
and because you must not mutate the view even if you could.

## Common mistakes

| Mistake | Symptom |
|---|---|
| `reverse=True` with a tuple key | tiebreaker reversed too; ties come out newest-first |
| `sorted(view.waiting)` with no key | `TypeError: '<' not supported between Request instances` |
| `view.waiting.sort()` | `AttributeError` — it is a tuple |
| forgetting sort is stable, adding a redundant tiebreaker | usually harmless, but you cannot express it here and will waste time trying |
| `key=lambda r: -r.name` | `TypeError` — negation is numbers only |

## Drill

`drills/drills.py` Block B. Then, from memory:

1. Sort `requests` by `max_tokens` descending, ties by `id` ascending.
2. From `view.admitted`, pick the lowest-priority request, breaking ties
   towards the most recently admitted.
3. `view.waiting` is `[A(p0), B(p2), C(p2), D(p1)]` in arrival order. What is
   `sorted(view.waiting, key=lambda r: -r.priority)`?

<details>
<summary>Answers</summary>

1. `sorted(requests, key=lambda r: (-r.max_tokens, r.id))`
2. ```python
   rank = {r.id: i for i, r in enumerate(view.admitted)}
   victim = min(view.admitted, key=lambda r: (r.priority, -rank[r.id]))
   ```
3. `[B, C, D, A]` — B before C because stability preserves their arrival
   order within the p2 tier.

</details>
