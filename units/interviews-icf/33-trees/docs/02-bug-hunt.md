# Debugging someone else's code

The first half of this page is method — read it before you start. The second
half is the answer key. **Spoilers below the line.**

---

## The method

### 1. Read the failures before the code

Eighteen of the thirty-five tests fail at the start. That is not eighteen
problems; it is seven causes. The failure list is the most information you will
ever get for free, and reading it first tells you which *behaviours* are
broken, which narrows the search enormously.

```
$ ./run 2>&1 | grep -E "^(FAIL|ERROR)"
```

Group them. Several tests failing inside `TestEntropy` point at one function.
A single failure named `all_nan_column_is_constant` names its property
exactly. A test called `respects_max_depth` tells you where to look in one
word.

### 2. The docstrings are the specification

The brief says the docstrings are correct and the code is not. That turns
debugging into **diff-reading**: for each function, read what it claims, then
read what it does, and find the sentence the code contradicts.

> *"The score is the normalized difference between entropy before the split,
> and the weighted entropies after the split, where each subset's entropy is
> **weighted by the fraction of the examples that fall in that subset**."*

Now read the three lines under it. Is anything weighted by a fraction?

This is the single highest-yield technique on this kind of task, and it is why
the brief tells you not to modify docstrings — they are the only fixed point
you have.

### 3. Two implementations of one rule

`predict_single` and `predict_batch` are documented as equivalent. When a test
compares them and they disagree, you have localised the bug to one of two
functions without understanding either. Then ask which one matches the
docstring.

Look for this kind of redundancy deliberately: any pair of functions that must
agree, any invariant stated in two places, any property asserted in a
docstring that you can check directly.

### 4. Match the signature of the failure to the kind of cause

- **`nan` in an assertion message** — an arithmetic identity broke. In NumPy
  that usually means `0 * inf`, `0/0`, or a `nan` propagating through an
  aggregation.
- **A value outside its documented range** (a "score between 0 and 1" coming
  back negative) — a formula is missing a term, not a typo.
- **Shape mismatch** — something is ragged that should be rectangular.
- **Fails only with missing values** — NaN semantics.
- **Fails only on multiclass** — something assumes two classes.
- **Accuracy poor but nothing crashes** — routing, thresholds, or voting.

Matching signature to cause before you open the file saves more time than any
amount of stepping through a debugger.

### 5. An assertion message is a hypothesis, not a diagnosis

You will hit this one:

```
AssertionError: score_split should not be called with uniform labels
```

on labels like `[3, 0, 1, 4, 1, 4, 1, 3]` — five distinct classes, about as
non-uniform as eight labels get. The message is simply wrong.

Look at what the assertion actually tests:

```python
assert ent_before > 0, "score_split should not be called with uniform labels"
```

`ent_before > 0` is False in **two** cases: when `ent_before` is 0, and when it
is NaN. The author wrote the message for the first case and never considered
the second. So a NaN arriving from `entropy` gets reported as "uniform
labels", and you go looking in entirely the wrong place.

Whenever an assertion message contradicts what you can see in the data,
**stop reading the message and read the condition.** Then ask what else could
make that condition false. With floating point the answer is very often NaN,
because every comparison against NaN is False — including the ones you would
expect to be True.

### 6. Change one thing, re-run

The bugs here are independent — each one, fixed alone, turns a specific set of
tests green. Fix one, run, watch the failure count drop. If you make four
changes and the count drops by two, you have learned almost nothing and may
have added a bug.

### 7. Debugging something stochastic

**Reproduce before you diagnose.** Every test here fixes a seed. If you are
experimenting in a scratch script, fix yours too, or you will "fix" something
that was only a different draw.

**Distinguish "wrong" from "unlucky".** A single tree at 82% might be a bug or
might be that tree. A forest at 55% on linearly separable blobs is a bug.
Prefer the aggregate signal.

---

# Answers

**Stop here if you are still working.**

<details>
<summary>Bug 1 — <code>entropy</code>: zero times negative infinity</summary>

```python
counts = np.bincount(Y)
probs = counts / len(Y)
bits = -np.sum(probs * np.log2(probs))     # -> nan
```

`np.bincount([1, 1, 1, 1, 1])` returns `[0, 5]` — a zero count for class 0,
which is not present. Then `0 * log2(0)` is `0 * -inf`, which is **NaN**, and
one NaN poisons the whole sum.

Mathematically the `p = 0` term is defined as 0 (its limit), so absent classes
simply do not contribute.

**Fix:**

```python
probs = counts / len(Y)
probs = probs[probs > 0]
bits = -np.sum(probs * np.log2(probs))
```

**How you find it:** `test_score_split_perfect` reports `score=nan` for a
split whose right side is all 1s. Everything downstream of `entropy` inherits
the NaN, so this one bug produces failures in `score_split`, in `build_tree`
(where `max(scored, key=...)` compares NaN scores), and in accuracy.

**Lesson:** `0 * inf` is NaN, and NumPy will not warn you in a way you notice.
Whenever a formula has a `log`, a division, or a reciprocal, ask what it does
at zero.

</details>

<details>
<summary>Bug 2 — <code>score_split</code>: the weighting is missing</summary>

```python
info_gain = ent_before - (ent_left + ent_right)      # unweighted
```

The docstring says each subset's entropy is weighted by the **fraction of the
examples** in it. The code just adds them.

The consequence is not subtle: adding two unweighted entropies routinely
exceeds `ent_before`, so `info_gain` goes negative and the "score between 0
and 1" comes back as −0.13. A returned value outside its own documented range
is about as loud as a bug gets.

**Fix:**

```python
frac_left = np.count_nonzero(is_left) / len(Y)
weighted_after = frac_left * ent_left + (1 - frac_left) * ent_right
info_gain = ent_before - weighted_after
return info_gain / ent_before
```

**How you find it:** `test_score_split_unbalanced` gives the exact numbers —
expected `0.15106…`, got `−0.13191…`. Work the arithmetic by hand once and the
missing factor is obvious.

**Lesson:** when a docstring names a formula, implement the formula, not
something with the same shape. "Weighted average" and "sum" differ by exactly
the term that makes the metric meaningful.

</details>

<details>
<summary>Bug 3 — <code>find_non_constant_features</code>: an all-NaN column</summary>

```python
feat_min = np.nanmin(X, axis=0)
feat_max = np.nanmax(X, axis=0)
return np.where(feat_min != feat_max)[0]
```

For a column that is entirely NaN, `nanmin` and `nanmax` are both NaN (with a
RuntimeWarning you will scroll past). And **`nan != nan` is `True`** — so the
column is reported as non-constant and gets split on, even though it contains
no information at all.

**Fix:**

```python
all_nan = np.all(np.isnan(X), axis=0)
return np.where((feat_min != feat_max) & ~all_nan)[0]
```

**How you find it:** `test_all_nan_constant` shows `ACTUAL: array([0, 2])`
against `DESIRED: array([2])`. Column 0 is the all-NaN one.

**Lesson:** NaN is not equal to itself. Any equality or inequality test on
values that might be NaN needs an explicit `np.isnan` guard — this is the same
root cause as the `nan < split` behaviour in prediction.

</details>

<details>
<summary>Bug 4 — <code>predict_single</code> routes everything left</summary>

```python
if val < tree.split or tree.nans_go_left:
    tree = tree.left
```

The docstring says a row goes left when the value is *strictly less than the
split*, **or when the value is NaN and `nans_go_left` is True**. The code drops
the "value is NaN and" half. So whenever a node has `nans_go_left=True`, the
`or` short-circuits and **every** row goes left — the threshold is ignored.

`nans_go_left` is chosen per split, so roughly half the tree is affected. It
never crashes; it just quietly loses accuracy.

**Fix:**

```python
val = x[tree.feature]
if np.isnan(val):
    go_left = tree.nans_go_left
else:
    go_left = val < tree.split
tree = tree.left if go_left else tree.right
```

**How you find it:** `predict_batch` is correct and does exactly this:

```python
mask = values < node.split
mask[np.isnan(values)] = node.nans_go_left    # override for NaN only
```

Two implementations of one documented rule, and a test that compares them.

**Lesson:** `A or B` is not "A, with B as a fallback". When a condition should
apply only in a specific case, the case has to be in the condition.

</details>

<details>
<summary>Bug 5 — <code>build_tree</code>: <code>max_depth</code> off by one</summary>

```python
if max_depth is not None and depth > max_depth:
    return Leaf(majority(Y))
```

The docstring is precise: *"The root is at depth 0, so a tree built with
max_depth=d contains Nodes only at depths **strictly less than** d, and Leaves
at depths no greater than d."*

With `>`, a node at depth `d` still splits, so nodes reach depth `d` and the
tree reaches depth `d + 1`. Worse, `max_depth=0` — documented as "a single
Leaf" — builds a Node.

**Fix:** `depth >= max_depth`.

**How you find it:** `test_respects_max_depth` and `test_max_depth_zero_is_a_leaf`.
One-line failure, one-line fix — and exactly the kind that survives for months
in real code because the output still looks plausible.

**Lesson:** when a bound is expressed as a depth, level or count, decide
whether it is inclusive and check the boundary case. Zero-indexed depth with
an inclusive maximum is where off-by-one lives.

</details>

<details>
<summary>Bug 6 — <code>Ensemble.fit</code>: one tree, repeated <em>m</em> times</summary>

```python
self.trees = [
    build_tree(X, Y, self.n_min, self.k, self.rng, self.max_depth)
] * self.m
```

`[expr] * m` evaluates `expr` **once** and stores `m` references to the same
object. So the forest is one tree wearing `m` costumes — and the docstring
says the opposite: *"The same data is used for each tree, but each tree's
splits will differ due to the RNG."*

The whole point of an ensemble is that averaging cancels uncorrelated errors.
Averaging `m` copies of one prediction gives you that prediction. The model
still works — it is just no better than a single tree, and no amount of
raising `m` helps.

**Fix:**

```python
self.trees = [
    build_tree(X, Y, self.n_min, self.k, self.rng, self.max_depth)
    for _ in range(self.m)
]
```

**How you find it:** `test_trees_are_not_all_identical` renders every tree and
counts the distinct strings. `test_forest_beats_a_single_tree` catches the
consequence.

**Lesson:** `[x] * n` is list *repetition*, not *iteration*. It is the same
trap as `[[]] * 3` giving you three references to one list — dangerous
whenever the element is mutable or the expression has side effects. When you
want `n` fresh things, write a comprehension.

</details>

<details>
<summary>Bug 7 — the split value silently widens to float64</summary>

```python
vmin, vmax = np.nanmin(values), np.nanmax(values)
split = rng.uniform(vmin, vmax)          # always a Python float
```

`np.random.RandomState.uniform` returns a **Python float (64-bit)** no matter
what dtype you hand it. So a tree grown on `float16` or `float32` data stores
`float64` thresholds — the model quietly stops matching its input's precision,
and a `Node.split` no longer round-trips through the dtype it was drawn from.

**Fix:**

```python
split = X.dtype.type(rng.uniform(vmin, vmax))
```

**How you find it:** the tell is sitting three lines below, in the fallback
path, which already gets it right:

```python
split = np.nextafter(vmin, vmax, dtype=X.dtype)     # correct dtype
```

Two branches setting the same variable, only one of them dtype-aware. That
asymmetry is the bug, and it is the same "two implementations of one rule"
smell as `predict_single` versus `predict_batch`.

**Lesson:** NumPy's scalar-returning functions do not preserve input dtype the
way its array operations do. `rng.uniform`, `float(...)`, `sum(...)` and
friends all widen to 64-bit. When a dtype is part of the contract, cast
explicitly — `X.dtype.type(value)` — rather than assuming it propagates.

</details>

---

## The general lessons

1. **The docstring is the spec.** Four of these six are visible as a
   disagreement between a comment and the lines under it.
2. **NaN does not raise; it spreads.** It survives comparisons as `False`,
   survives equality as "not equal", and propagates through every aggregation
   it touches.
3. **`0 * inf` is NaN.** Any `log`, division or reciprocal needs its zero case
   checked.
4. **A documented range is an assertion.** A "score between 0 and 1" that
   returns −0.13 has told you where the bug is.
5. **Redundancy is a debugging tool.** Two implementations of one rule
   localise a bug for free.
6. **An assertion message is the author's guess about why it could fail.**
   Read the condition, not the string.
7. **NumPy scalars widen.** Array ops preserve dtype; scalar-returning
   functions like `rng.uniform` return 64-bit floats regardless.
8. **One change, one run.** Batched fixes destroy the signal you are using.
