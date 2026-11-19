# Prereq: NumPy essentials for this mock

Run every example here yourself — `python3 -i` and paste. The goal is
reflexes, not reading comprehension.

## `np.array`, and floating point

```python
>>> import numpy as np
>>> np.array([[1.0, 2.0], [3.0, 4.0]])
array([[1., 2.],
       [3., 4.]])
```

`np.array` infers a dtype from its contents. Mix an `int` and a `float`
(or a `NaN`, which is a float) in one row and the whole row upcasts to
`float64` — this is what makes `[[float("nan")] * num_days for _ in
station_ids]` safe to feed straight to `np.array`: every cell is already a
Python float, so the array comes out `float64` with no surprises.

Floating point is inexact: `0.1 + 0.2 != 0.3`. Tests in this mock that check
a computed mean use `assertAlmostEqual`, not `assertEqual` — do the same in
your own scratch checks.

## `NaN`: the semantics that drive this whole mock

`float("nan")` (equivalently `np.nan`) is a float that is **not equal to
anything, including itself.**

```python
>>> float("nan") == float("nan")
False
>>> x = np.array([1.0, float("nan"), 3.0])
>>> x == x
array([ True, False,  True])
```

That means you cannot filter it out with `x != x` meaning "keep everything"
— you have to ask NumPy directly:

```python
>>> np.isnan(x)
array([False,  True, False])
>>> x[~np.isnan(x)]          # ~ inverts the boolean mask: "the valid ones"
array([1., 3.])
```

And it propagates through reductions instead of raising:

```python
>>> np.min(x), np.max(x), np.sum(x)
(np.float64(nan), np.float64(nan), np.float64(nan))
```

**The rule for this whole mock: mask with `np.isnan` before you reduce, never
after.** `np.min(x[~np.isnan(x)])` is correct; `np.min(x)` silently isn't.

(NumPy also ships `np.nanmin` / `np.nansum` / etc., which do this masking
for you — this mock deliberately doesn't use them, so that the masking step
is something you write and can misplace.)

## Broadcasting: replacing a pairwise loop with one expression

Two arrays combine element-wise if their shapes are compatible by lining up
from the right and either matching or one side being `1`. The trick used in
level 2: turn a 1-D array into a column with `[:, None]`, and NumPy expands
both sides into a full pairwise grid.

```python
>>> means = np.array([4.0, 2.0, 11.0])
>>> means[:, None]                      # shape (3, 1) - a column
array([[ 4.],
       [ 2.],
       [11.]])
>>> means[None, :]                      # shape (1, 3) - a row
array([[ 4.,  2., 11.]])
>>> means[:, None] - means[None, :]     # shape (3, 3): every pairwise difference
array([[ 0.,  2., -7.],
       [-2.,  0., -9.],
       [ 7.,  9.,  0.]])
```

Row `i` of that result is "station `i` minus every station", so
`np.sum(np.abs(diffs), axis=1)` gives one number per station: its total
distance from everyone else. No loop over pairs.

## `np.bincount`: counting into integer buckets

`np.bincount(x)` needs a 1-D array of **non-negative integers** and returns,
at index `i`, how many entries of `x` equal `i`.

```python
>>> np.bincount(np.array([0, 2, 2, 5]))
array([1, 0, 2, 0, 0, 1])
```

The output is only as long as `max(x) + 1` unless you pass `minlength` — pad
it explicitly if you need trailing empty bins to show up:

```python
>>> np.bincount(np.array([0, 2]), minlength=6)
array([1, 0, 1, 0, 0, 0])
```

If your buckets have an offset (e.g. temperatures starting at some `low`
that isn't 0), subtract it first — `bincount` will raise on negative
values, not clip them.

## `np.nonzero` and `np.where`: turning a condition into indices

Both answer "which positions satisfy this?" On a 1-D array they return the
same thing, wrapped in a one-element tuple (one array of indices per
dimension of the input):

```python
>>> counts = np.array([3, 0, 1, 0])
>>> np.nonzero(counts == 0)
(array([1, 3]),)
>>> np.where(counts == 0)
(array([1, 3]),)
>>> np.nonzero(counts == 0)[0]          # unwrap: the actual index array
array([1, 3])
```

`np.where` has a second form, `np.where(condition, if_true, if_false)`, that
picks element-wise between two arrays — that's a different function in
practice, so check how many arguments you're passing.

## `np.random.RandomState`: reproducible, *isolated* randomness

```python
>>> rng = np.random.RandomState(42)
>>> rng.choice(5, size=2, replace=False)
array([1, 4])
```

The same seed on a fresh `RandomState` always produces the same sequence —
that's what makes an "audit sample" checkable in a test at all. The trap is
reaching for the module-level `np.random.choice(...)` instead: it shares one
global generator across your whole process, so its output depends on
whatever else happened to call `np.random` first. `RandomState(seed)`
creates an independent generator with no such history.
