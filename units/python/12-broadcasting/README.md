# 12 — Broadcasting and vectorisation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/python/12-broadcasting/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/python/broadcasting?utm_source=github&utm_medium=labs)

*~55 minutes · 32 tests*

Every deep-learning framework is NumPy's shape semantics with autograd
bolted on. If you can reason about broadcasting on paper, you can read a
`forward()` you have never seen, and you can debug the shape error that
every training run eventually throws.

This module is where "senior AI engineer" interviews actually start. The
questions are rarely "implement backprop" — they are *"give me pairwise
distances without a loop"*, *"why is my softmax NaN?"*, *"write IoU"*.

Work in `exercises.py`. Check your work with `python -m labkit check 12`, or open the notebook (`lab.ipynb`) in Colab.

Each section below ends with **Test it** — the exact command for just that
piece, run from `units/python/12-broadcasting/`. No need to finish the
whole file before checking your work.

## What you'll build

Nine functions in `exercises.py`, in this order. Keep this table open as you
go — each section below tells you which row it's building:

| Function | You'll implement | Shows up in real code as |
|---|---|---|
| `broadcast_shape` | `broadcast_shape` | The mental model for reading a shape-mismatch traceback — reasoning by hand about why a `(batch, heads, seq, dim)` attention tensor won't align with a `(seq, seq)` mask, before you ever run the op |
| `pairwise_distances` | `pairwise_distances` | k-NN search, k-means cluster assignment, the retrieval step of RAG — scoring one batch of embeddings against another with no Python loop |
| `softmax` | `softmax` | The output layer of every classifier, and the attention-weight normalisation inside every transformer block |
| `one_hot` | `one_hot` | Turning integer class labels into the dense target format a cross-entropy loss expects |
| `masked_mean` | `masked_mean` | Pooling a batch of variable-length token embeddings into one vector per sequence while ignoring padding |
| `iou` | `iou` | Non-max suppression and mAP scoring in every object-detection pipeline |
| `batched_dot` | `batched_dot` | Per-example similarity in a contrastive loss (SimCLR-style), computed without ever materialising the full `(n, n)` similarity matrix |
| `top_k` | `top_k` | Top-k / nucleus sampling during text generation, and pruning candidates in beam search or retrieval |
| `normalise_rows` | `normalise_rows` | L2-normalising embeddings before cosine similarity — standard practice before any embedding-index lookup |

---

## 1. The broadcasting rule, stated once

Align the shapes **from the right**. Pad the shorter one with leading 1s.
For each axis, the sizes must be **equal**, or **one of them must be 1**. The
output takes the larger.

```
(3, 1)  and  (1, 4)   ->  (3, 4)      ok
(5, 3)  and     (3,)  ->  (5, 3)      ok, (3,) is padded to (1, 3)
(3,)    and     (4,)  ->  error       3 != 4 and neither is 1
(2, 3)  and  (3, 2)   ->  error       aligned from the right: 3 vs 2
```

A size-1 axis is not copied — NumPy sets its stride to 0 and re-reads the
same memory. Broadcasting is free in memory; only the *output* is allocated.

```python
>>> broadcast_shape((3, 1), (1, 4))
(3, 4)
>>> broadcast_shape((5, 3), (3,))
(5, 3)
>>> broadcast_shape((3,), (4,))          # incompatible -> None, not an exception
>>> broadcast_shape((3,), (4,)) is None
True
```

**In `exercises.py`:** `broadcast_shape` is this section's only target — the
rule written out as ten lines of Python. Implementing it once is the fastest
way to stop guessing.

**Test it:** `python3 -m unittest test_exercises.TestBroadcastShape -v`

## 2. `None` (a.k.a. `np.newaxis`) is how you steer it

Inserting a length-1 axis is the entire technique behind every "no loops"
answer.

```python
a[:, None, :]      # (n, d) -> (n, 1, d)
b[None, :, :]      # (m, d) -> (1, m, d)
a[:, None, :] - b[None, :, :]     # (n, m, d) — every pair
```

```python
>>> a = np.array([[0.0, 0.0], [3.0, 4.0]])
>>> b = np.array([[0.0, 0.0], [0.0, 5.0]])
>>> pairwise_distances(a, b)
array([[0.        , 5.        ],
       [5.        , 3.16227766]])
```

That is `pairwise_distances`. The same move, applied to a column of labels
against a row of class indices, is `one_hot`:

```python
labels[:, None] == np.arange(num_classes)[None, :]     # (n, C) booleans
```

```python
>>> one_hot(np.array([0, 2, 1]), 3)
array([[1., 0., 0.],
       [0., 0., 1.],
       [0., 1., 0.]], dtype=float32)
```

And applied four times over box coordinates — `boxes_a[:, i][:, None]`
against `boxes_b[:, i][None, :]` for each of x1/y1/x2/y2 — it is `iou`.

**The memory warning.** The intermediate above is `n * m * d` elements. For
10k × 10k × 128 float32 that is 51 GB. Interviewers love this follow-up: the
production form expands `‖a - b‖² = ‖a‖² + ‖b‖² - 2a·b` so you only ever
materialise `(n, m)`. Know that the naive version is correct *and* know when
it stops being usable.

**In `exercises.py`:** `pairwise_distances` and `one_hot` are fully built
from this section's technique alone. `iou` uses the same None-axis trick for
its four coordinate comparisons, but its full implementation also needs the
division guard from §5 — it's covered there, not here.

**Test it:** `python3 -m unittest test_exercises.TestPairwiseDistances test_exercises.TestOneHot -v`

## 3. `keepdims=True` is not cosmetic

A reduction drops the axis it reduced:

```python
>>> x = np.arange(20.0).reshape(4, 5)
>>> x.shape
(4, 5)
>>> x.sum(axis=-1).shape
(4,)
>>> x.sum(axis=-1, keepdims=True).shape
(4, 1)
```

Only the `keepdims` version broadcasts back against `x`. Without it,
`x / x.sum(axis=-1)` either raises or — worse — silently broadcasts along the
*wrong* axis when the array happens to be square. Square test inputs hide
this bug; the tests here deliberately use non-square shapes.

This section doesn't map to a function of its own — it's the detail that
makes two functions built elsewhere actually correct: `softmax` (§4) needs
`keepdims=True` on both the max and the sum, and `normalise_rows` (§5) needs
it on the norm. Get it wrong in either and the *shape* still looks fine most
of the time — the classic silent-square-array bug — which is exactly why
each has a dedicated non-square (or square-but-asymmetric-values) test.

**In `exercises.py`:** no function belongs to this section alone; it's
exercised as part of finishing `softmax` (§4) and `normalise_rows` (§5).

**Test it:** `python3 -m unittest test_exercises -k sums_to_one -k rows_become_unit_length -v`

`test_sums_to_one` uses a non-square `(5, 7)` array specifically so a missing
`keepdims` raises instead of silently passing; `test_rows_become_unit_length`
uses a square `(2, 2)` array where a missing `keepdims` broadcasts the row
norms along columns instead of rows and produces a wrong-but-shape-valid
answer. Both tests also require the rest of `softmax`/`normalise_rows` to be
implemented — neither function is finished by getting `keepdims` right alone.

## 4. Numerical stability: the max-subtraction trick

```python
shifted = x - np.max(x, axis=axis, keepdims=True)
e = np.exp(shifted)
return e / np.sum(e, axis=axis, keepdims=True)
```

`exp(1000)` is `inf`, and `inf / inf` is NaN. Subtracting the row max changes
nothing mathematically — the constant cancels in numerator and denominator —
but caps the largest exponent at `exp(0) == 1`. Every real softmax does this,
including PyTorch's.

```python
>>> x = np.array([1000.0, 1000.0, 1000.0])
>>> softmax(x)
array([0.33333333, 0.33333333, 0.33333333])
>>> np.exp(x) / np.sum(np.exp(x))        # the naive version, same input
array([nan, nan, nan])
```

The same family of tricks worth naming: `logsumexp` instead of
`log(sum(exp(...)))`, `log1p(x)` instead of `log(1 + x)` for small `x`, and
clamping before dividing.

**In `exercises.py`:** `softmax` is this section's target (its `keepdims`
requirement is §3's material — same function, same implementation).

**Test it:** `python3 -m unittest test_exercises.TestSoftmax -v`

## 5. Guarding division without breaking gradients

`masked_mean`, `iou` and `normalise_rows` all divide by something that can be
zero. The pattern used throughout:

```python
np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)
```

```python
>>> masked_mean(np.array([1.0, 2.0]), np.array([False, False]))
array(nan)
>>> normalise_rows(np.array([[0.0, 0.0], [3.0, 4.0]]))
array([[0. , 0. ],
       [0.6, 0.8]])
```

Note that **both branches of `np.where` are evaluated** — it is not a
short-circuit. So you cannot use it to avoid the division; you must make the
division itself safe (`np.maximum(denominator, eps)`) *and* select
afterwards. Getting this wrong produces a correct-looking answer plus a
stream of `RuntimeWarning: invalid value encountered`:

```python
>>> totals, counts = np.array(0.0), np.array(0.0)
>>> totals / counts       # no guard: still ends up nan, but now it warns too
<stdin>:1: RuntimeWarning: invalid value encountered in divide
nan
```

`np.errstate(invalid="ignore", divide="ignore")` suppresses the warning once
you have deliberately handled the case. Reach for it only after the guard,
never instead of it.

**In `exercises.py`:** `masked_mean` and `normalise_rows` are fully built
from this section's guard pattern alone. `iou` needs the same guard for the
case where two boxes' union is 0, on top of the None-axis broadcasting from
§2 — its test class exercises both pieces together and can't isolate one
from the other.

**Test it:** `python3 -m unittest test_exercises.TestMaskedMean test_exercises.TestIoU test_exercises.TestNormaliseRows -v`

## 6. Vectorised does not mean "one line"

`batched_dot` is the standard example:

```python
np.sum(a * b, axis=-1)          # O(n·d) time, O(n·d) memory
np.diag(a @ b.T)                # O(n²·d) time, O(n²) memory — same answer
```

```python
>>> a = np.array([[1.0, 2.0], [3.0, 4.0]])
>>> b = np.array([[5.0, 6.0], [7.0, 8.0]])
>>> batched_dot(a, b)
array([17., 53.])
>>> np.diag(a @ b.T)             # same numbers, but an (n, n) matrix along the way
array([17., 53.])
```

Both are "vectorised". One is quadratic. The interview answer is that you
should compute what you need, not slice it out of something bigger.
`np.einsum("nd,nd->n", a, b)` says the same thing and is worth being fluent
in — einsum notation transfers directly to attention implementations.

**In `exercises.py`:** `batched_dot` is the only target here.

**Test it:** `python3 -m unittest test_exercises.TestBatchedDot -v`

## 7. Partial sorting

`top_k` uses `np.argpartition`, which places the k-th element in its final
position in **O(n)** and leaves the rest unordered. Only those k are then
sorted. Full `argsort` is O(n log n) over the whole array — pointless when
k is 10 and n is a million.

Tie-breaking needs `np.lexsort`, which takes the **primary key last**:

```python
candidates[np.lexsort((candidates, -scores[candidates]))]
#                      ^secondary   ^primary
```

Descending score first, lower index as the tie-break. Reading `lexsort`'s
argument order backwards is a very common bug.

```python
>>> scores = np.array([0.1, 0.9, 0.5, 0.7, 0.9])
>>> top_k(scores, 3)      # two 0.9s tie at indices 1 and 4; lower index wins
array([1, 4, 3])
>>> top_k(scores, 0)
array([], dtype=int64)
```

**In `exercises.py`:** `top_k` is the only target here.

**Test it:** `python3 -m unittest test_exercises.TestTopK -v`

---

## The exercises

| Function | What it teaches |
|---|---|
| `broadcast_shape` | The rule itself, including the failure cases |
| `pairwise_distances` | `None`-axis insertion; the memory blow-up |
| `softmax` | Max-subtraction; `keepdims` |
| `one_hot` | Broadcasting a comparison; dtype control |
| `masked_mean` | Safe division; empty-group semantics |
| `iou` | Detection staple; clipping non-overlaps at 0 |
| `batched_dot` | Vectorised ≠ efficient; einsum |
| `top_k` | `argpartition` vs `argsort`; `lexsort` key order |
| `normalise_rows` | Epsilon guards; all-zero rows |

One edge case in this suite has no test coverage: `iou`'s `union > 0` guard
— the branch that hands back `0.0` instead of dividing when two boxes have
zero combined area — is never exercised, since every box pair in
`test_exercises.py` has positive union. Nothing breaks if you skip the
special-case handling for these tests specifically, but a real zero-union
pair would divide by `np.maximum(0, 1e-12)` instead of hitting the intended
`0.0` branch without it.

## Check your understanding

1. **`(3, 1)` and `(1, 4)` broadcast together to `(3, 4)`. Why does
   `(3,)` and `(4,)` fail instead of also broadcasting?**
   <details><summary>Answer</summary>Shapes align from the right, and
   each axis must either match or be 1. <code>(3,)</code> vs
   <code>(4,)</code> compares 3 against 4 directly — neither matches nor
   is 1, so there's no valid way to broadcast them.</details>

2. **Why does `softmax` subtract the row max before exponentiating,
   when the result is mathematically identical either way?**
   <details><summary>Answer</summary><code>exp</code> of a large number
   overflows to <code>inf</code>, and <code>inf / inf</code> is
   <code>NaN</code>. Subtracting the max caps the largest exponent at
   <code>exp(0) == 1</code>, avoiding the overflow — the constant cancels
   in the numerator and denominator, so the math is unchanged.</details>

3. **Why can't `np.where(cond, a / b, fallback)` be used to avoid a
   division by zero in `b`?**
   <details><summary>Answer</summary><code>np.where</code> is not
   short-circuiting — both <code>a / b</code> and <code>fallback</code>
   are evaluated regardless of <code>cond</code>. The division itself
   still needs to be made safe (e.g. <code>np.maximum(b, eps)</code>)
   before <code>np.where</code> selects between the two.</details>

4. **`iou`'s test suite never triggers its `union > 0` guard. What would
   happen with a real zero-union box pair if that guard were
   removed?**
   <details><summary>Answer</summary>It would divide by
   <code>np.maximum(0, eps)</code> instead of returning the intended
   <code>0.0</code> — not a crash, but a value that only looks
   correct.</details>

5. **What's the actual difference between `argpartition` and `argsort`
   for `top_k`, and why does it matter?**
   <details><summary>Answer</summary><code>argpartition</code> only
   guarantees the top k elements are in the first k positions, unsorted
   among themselves — O(n). <code>argsort</code> fully sorts everything —
   O(n log n). For just "the top k," partition does less work than a
   full sort requires.</details>

6. **Why is `pairwise_distances` a genuine memory concern at scale, not
   just a style question?**
   <details><summary>Answer</summary>Broadcasting two `(n, 1, d)` and
   `(1, m, d)` arrays materializes an `(n, m, d)` intermediate — at
   10k×10k×128 that's tens of gigabytes, even though the final distance
   matrix is only `(n, m)`.</details>

## Interview questions this module answers

- *Compute pairwise distances between two sets of vectors, no loops.*
- *Why is my softmax returning NaN?*
- *What does `keepdims` do and when do I need it?*
- *Implement IoU for two sets of boxes.*
- *These two shapes don't broadcast — why?*
- *Get the top-k scores from a million-element array efficiently.*
- *Your pairwise-distance version OOMs at 100k rows. Now what?*
