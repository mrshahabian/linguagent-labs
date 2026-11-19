# 13 — Views, copies and dtypes

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/python/13-array-memory/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/python/array-memory?utm_source=github&utm_medium=labs)

*~45 minutes · 26 tests*

An `ndarray` is two things: a flat block of bytes, and a small description of
how to walk it (shape, strides, dtype, offset). Almost every NumPy behaviour
that surprises people follows from that split — including the two quietest
bug classes in numerical code: **accidental aliasing** and **silent
overflow**.

Neither raises. Both produce plausible numbers. This module is about
recognising them before they reach a training run.

Work in `exercises.py`. Check your work with `python -m labkit check 13`, or open the notebook (`lab.ipynb`) in Colab.

Each section below ends with **Test it** — the exact command for just that
piece, run from `units/python/13-array-memory/`. No need to finish the whole
file before checking your work.

## What you'll build

Nine functions in `exercises.py`, in this order. Keep this table open as you
go — each section below tells you which row it's building:

| Function | You'll implement | Shows up in real code as |
|---|---|---|
| `is_view_of` | `np.shares_memory` | Debugging aliasing bugs — confirming whether a preprocessing step handed you a fresh buffer or a window into the original batch |
| `safe_scale` | non-mutating `*` | Writing a library/augmentation function that must not surprise its caller by mutating the batch it was passed |
| `scale_in_place` | `*=`, returning `None` | Memory-bound normalisation over a huge batch — scaling in place instead of allocating a second copy |
| `zero_negatives` | copy-then-mask | Clipping/clamping pixel or activation values (the same shape as a ReLU) without touching the input |
| `to_contiguous` | `np.ascontiguousarray` | The exact call behind PyTorch's `.contiguous()` before `.view()`, or before handing an array to a BLAS/C extension that requires contiguous input |
| `sliding_window_view` | `np.lib.stride_tricks.as_strided` | Zero-copy rolling windows — rolling statistics, `im2col`-style convolution unrolling over large time series or sensor arrays |
| `safe_sum` | choosing an accumulator dtype | Reductions (loss, activation sums) in mixed-precision training, where the accumulator's width matters more than the storage width |
| `downcast_safely` | `np.iinfo`/`np.finfo` range checks | Quantizing to `int8` for inference, or converting a float image back to `uint8` before saving — without silently wrapping |
| `memory_bytes` | `prod(shape) * itemsize` | Sizing a batch before you allocate it, to answer "why does this OOM?" before it happens |

---

## 1. Views vs copies

**Basic slicing returns a view** — a new array object pointing at the *same*
buffer:

```python
x[2:5]        x[:, 0]        x.T        x.reshape(4, 3)        x.ravel()
```

**Fancy indexing returns a copy** — the result cannot be expressed as a
stride pattern, so the data must be gathered:

```python
x[[0, 2, 4]]        x[x > 5]        x.flatten()
```

The consequence:

```python
row = image[1]        # view
row *= 2              # image is now modified
```

`np.shares_memory(a, b)` is the definitive check (`np.may_share_memory` is
the cheap conservative one that can return false positives). `arr.base`
tells you what an array is a view of — `None` if it owns its data:

```python
>>> x = np.arange(10)
>>> row = x[2:5]
>>> row
array([2, 3, 4])
>>> row.base is x
True
>>> np.shares_memory(row, x)
True
>>> copy = x[[0, 2, 4]]      # fancy indexing
>>> copy.base is None
True
>>> np.shares_memory(copy, x)
False
```

**Note the asymmetry:** `x.ravel()` returns a view when it can, `x.flatten()`
always copies. Same for `x.reshape(...)` vs `np.array(x, copy=True)`. When
you need a guarantee in either direction, say so explicitly rather than
relying on which one you happened to type.

**In `exercises.py`:** `is_view_of` is this section's only target — it's a
thin wrapper over `np.shares_memory`, but getting comfortable with *when*
NumPy views vs. copies is the actual point.

**Test it:** `python3 -m unittest test_exercises.TestViews -v`

## 2. In-place operators mutate through views

`x = x * 2` rebinds a local name — the old array is untouched. `x *= 2` calls
`__imul__` and writes into the existing buffer instead. If `x` is a view of
something else, that something else changes too, and the caller has no idea.

### Not mutating the caller: `safe_scale`

The rule for library code: **a function that takes an array and returns a
value must not mutate its input.**

```python
def safe_scale(x, factor):
    return x * factor        # new array, caller's x untouched
```

```python
>>> x = np.array([1.0, 2.0])
>>> out = safe_scale(x, 3.0)
>>> out
array([3., 6.])
>>> x
array([1., 2.])
```

The nastier version of the bug: `x` doesn't have to be the top-level array
for this to matter. If it's a *view* of something else, an in-place multiply
corrupts the parent too:

```python
>>> base = np.ones((3, 3))
>>> row = base[1]          # a view
>>> row *= 5.0              # wrong tool here - mutates base through row
>>> base
array([[1., 1., 1.],
       [5., 5., 5.],
       [1., 1., 1.]])
```

`safe_scale` must leave `base` untouched even when handed `row`, not just
when handed a top-level array.

**In `exercises.py`:** `safe_scale` is this subsection's target.

**Test it:** `python3 -m unittest test_exercises -k safe_scale -v`

### Mutating in place, and returning `None`: `scale_in_place`

In-place *is* the right choice when memory-bound — `np.multiply(a, b,
out=a)` avoids a full allocation, which is why every serious NumPy pipeline
is full of `out=`. The convention, following Python's own (`list.sort()`
returns `None`, `sorted()` returns a list): **a function that mutates in
place returns `None`**, so it can never be confused for one that returns a
fresh value.

```python
>>> x = np.array([1.0, 2.0])
>>> scale_in_place(x, 3.0)
>>> x
array([3., 6.])
```

And because it mutates through the buffer, it writes through a view into its
base too — the flip side of `safe_scale`'s guarantee:

```python
>>> base = np.ones((2, 2))
>>> scale_in_place(base[0], 7.0)
>>> base
array([[7., 7.],
       [1., 1.]])
```

**In `exercises.py`:** `scale_in_place` is this subsection's target — the
`*=` counterpart to `safe_scale`'s `*`.

**Test it:** `python3 -m unittest test_exercises -k scale_in_place -v`

### Copy-then-modify: `zero_negatives`

The general pattern for "return a modified version without touching the
input" when the modification isn't a single operator: copy first, then
mutate the copy.

```python
>>> x = np.array([-1.0, 2.0, -3.0])
>>> zero_negatives(x)
array([0., 2., 0.])
>>> x
array([-1.,  2., -3.])
```

**In `exercises.py`:** `zero_negatives` — the same non-mutation contract as
`safe_scale`, but implemented as an explicit `.copy()` plus a boolean-mask
assignment rather than a single arithmetic operator.

**Test it:** `python3 -m unittest test_exercises -k zero_negatives -v`

## 3. Contiguity

C-contiguous means the last axis varies fastest and the elements sit in
memory in the order you read them. A transpose is a view with **reversed
strides**, so it is not C-contiguous:

```python
>>> x = np.arange(12).reshape(3, 4)
>>> x.strides
(32, 8)
>>> x.T.strides
(8, 32)
>>> x.flags["C_CONTIGUOUS"]
True
>>> x.T.flags["C_CONTIGUOUS"]
False
```

This matters because BLAS calls, `.reshape()` in some cases, and most C
extensions require contiguous input — so they copy silently. Doing it
yourself with `np.ascontiguousarray` makes the cost visible in a profile
instead of hiding it inside a library call:

```python
>>> out = to_contiguous(x.T)
>>> out.flags["C_CONTIGUOUS"]
True
>>> np.shares_memory(out, x)
False
```

It returns the input **unchanged** (same object, no copy) when it is already
contiguous, so it is safe to call unconditionally:

```python
>>> to_contiguous(x) is x
True
```

Same idea, same reason, in PyTorch: `.contiguous()` before a `.view()`.

**In `exercises.py`:** `to_contiguous` is this section's target — one call to
`np.ascontiguousarray`. One of its tests (`test_already_contiguous_is_not_copied`)
checks the "no copy" guarantee via `is_view_of`, so §1's `is_view_of` needs
to be implemented too for that particular assertion to pass, even though
`to_contiguous` itself doesn't call it.

**Test it:** `python3 -m unittest test_exercises.TestContiguity -v`

## 4. Strides, and the trick behind sliding windows

You can fabricate a stride pattern directly:

```python
np.lib.stride_tricks.as_strided(
    x, shape=(len(x) - size + 1, size), strides=(stride, stride))
```

This produces every overlapping window of `x` **with zero copying** — the
rows deliberately alias each other:

```python
>>> x = np.arange(5)
>>> w = sliding_window_view(x, 3)
>>> w
array([[0, 1, 2],
       [1, 2, 3],
       [2, 3, 4]])
>>> w.strides
(8, 8)
>>> np.shares_memory(w, x)
True
>>> w[0, 0] = 99      # write through the view...
>>> x                  # ...corrupts the source
array([99,  1,  2,  3,  4])
```

This is how `np.lib.stride_tricks.sliding_window_view` and convolution
`im2col` work, and it is genuinely the right tool for rolling statistics
over large arrays.

It is also unsafe by construction: `as_strided` does no bounds checking. Give
it a shape that walks past the end of the buffer and you read arbitrary
memory, no error. Validate the sizes yourself before calling it — which is
what the exercise's `ValueError` guards are for.

**In `exercises.py`:** `sliding_window_view` is this section's target — build
the `shape`/`strides` tuple and validate `size` before calling `as_strided`.
One of its tests (`test_is_a_view_not_a_copy`) confirms the zero-copy
guarantee via `is_view_of` from §1, so that needs to be implemented too for
that assertion to pass.

**Test it:** `python3 -m unittest test_exercises.TestSlidingWindowView -v`

## 5. Dtypes overflow silently

NumPy's integer types are fixed-width and wrap; Python's `int` is arbitrary
precision. Mixing the two mental models is the bug.

Two more worth carrying, since neither has its own exercise but both show up
constantly in real pipelines:

- **Type promotion widens.** `float32_array + python_float` stays float32 in
  modern NumPy, but `float32_array + np.float64(...)` becomes float64:

  ```python
  >>> a = np.array([1.0], dtype=np.float32)
  >>> (a + 1.0).dtype
  dtype('float32')
  >>> (a + np.float64(1.0)).dtype
  dtype('float64')
  ```

  A single accidental float64 in a float32 pipeline doubles memory and can
  break an assertion far away from where it happened.
- **`==` on floats is not reliable**, and `np.nan != np.nan` is `True`, so
  `x == np.nan` never matches:

  ```python
  >>> np.array([1.0, np.nan]) == np.nan
  array([False, False])
  ```

  Use `np.isnan`, and `np.allclose` / `np.testing.assert_allclose` for
  comparisons.

### Integer accumulation: `safe_sum`

Plain `np.sum(x)` already protects you for narrow integer dtypes: NumPy
automatically widens the accumulator to (at least) the platform integer, so
`np.sum` on an `int8` array quietly does the right thing on its own. The trap
is when something **pins the accumulator back down** to the storage
dtype — explicitly, with `dtype=x.dtype`, or implicitly, by accumulating
manually in a loop with a narrow scalar:

```python
>>> x = np.full(10, 100, dtype=np.int8)
>>> np.sum(x)                    # no dtype given -> auto-widens, correct
1000
>>> np.sum(x, dtype=np.int8)     # accumulator forced to int8 -> wraps
-24
>>> safe_sum(x)
1000
```

For floats there's no automatic widening at all — `np.sum` on a `float32`
array accumulates in `float32` and loses precision as terms pile up. Naming
a wider accumulator explicitly is the fix, the same reasoning behind
reducing float16 tensors with a float32 accumulator on GPUs — the
accumulator's precision matters more than the storage precision:

```python
>>> xf = np.full(1000, 0.1, dtype=np.float32)
>>> np.sum(xf)                        # accumulates in float32
100.00001
>>> safe_sum(xf)                      # accumulates in float64
100.00000149011612
```

**In `exercises.py`:** `safe_sum` — pick `np.int64` for integer input,
`np.float64` for float input, and pass it as `dtype=` to `np.sum`.

**Test it:** `python3 -m unittest test_exercises -k sum -v`

### Downcasting: `downcast_safely`

Downcasting is quieter still — no accumulator involved, just a value that
doesn't fit in the target type, cast anyway:

```python
>>> np.array([300]).astype(np.uint8)      # no warning
array([44], dtype=uint8)
>>> np.array([-1]).astype(np.uint8)
array([255], dtype=uint8)
```

`downcast_safely` checks against `np.iinfo(dtype)` / `np.finfo(dtype)`
first and raises instead:

```python
>>> downcast_safely(np.array([0, 100, 255]), np.uint8)
array([  0, 100, 255], dtype=uint8)
>>> downcast_safely(np.array([300]), np.uint8)
ValueError: values do not fit in uint8
```

In image pipelines — where `uint8` in, float in the middle, `uint8` out is
the normal shape — this is the difference between a clipped image and a
psychedelic one.

**In `exercises.py`:** `downcast_safely` — look up the range with
`np.iinfo`/`np.finfo`, compare `x` against `.min`/`.max`, then `.astype`.

**Test it:** `python3 -m unittest test_exercises -k downcast -v`

## 6. Counting bytes

`memory_bytes(shape, dtype)` is `prod(shape) * dtype.itemsize` — the same
number as `arr.nbytes`, excluding the ~100 bytes of object overhead:

```python
>>> memory_bytes((3, 4), np.float32)
48
>>> np.zeros((3, 4), dtype=np.float32).nbytes
48
```

Being able to do this in your head is a real interview skill: *"a batch of
32 1024×1024 RGB float32 images"* is 32 × 3 × 1024² × 4 ≈ **402 MB**, per
copy, and a naive pipeline holds several copies at once:

```python
>>> memory_bytes((32, 3, 1024, 1024), np.float32) / 1e6
402.653184
```

That is the arithmetic behind "why does this OOM?".

**In `exercises.py`:** `memory_bytes` — multiply `shape` through with a
plain Python loop (or `math.prod`) and multiply by `np.dtype(dtype).itemsize`.

**Test it:** `python3 -m unittest test_exercises.TestMemoryBytes -v`

---

## The exercises

| Function | What it teaches |
|---|---|
| `is_view_of` | `shares_memory`; the basic-vs-fancy rule |
| `safe_scale` | Not mutating the caller's array — or its base |
| `scale_in_place` | In-place semantics; returning `None` |
| `zero_negatives` | Copy-then-modify |
| `to_contiguous` | Transposes are not contiguous; free when already so |
| `sliding_window_view` | `as_strided`; zero-copy windows; validating sizes |
| `safe_sum` | Accumulator dtype; integer wraparound |
| `downcast_safely` | `iinfo`/`finfo` range checks before casting |
| `memory_bytes` | Sizing arrays from shape and dtype |

## Check your understanding

1. **`arr[1:3]` and `arr[[1, 2]]` look similar. Which one is a view and
   which is a copy?**
   <details><summary>Answer</summary>Basic indexing — slices like
   <code>arr[1:3]</code> — returns a view. Fancy indexing — a list or
   array of indices like <code>arr[[1, 2]]</code> — always
   copies.</details>

2. **You mutate a slice of an array with `+=`, and the original array
   also changed. Why?**
   <details><summary>Answer</summary>The slice was a view, not a copy —
   it shares the same underlying memory as the original, so an in-place
   mutation through it is visible in the original too.</details>

3. **Why is a transposed array not contiguous, and when does that
   actually matter?**
   <details><summary>Answer</summary>Transposing swaps the strides
   without moving any data, so walking the transposed array in memory
   order no longer matches its logical row-major order. It matters when
   a downstream operation (like reshaping, or handing the array to
   another library) needs contiguous memory and silently copies to get
   it.</details>

4. **`np.array([300]).astype(np.uint8)` is `44`, with no warning or
   error. Why?**
   <details><summary>Answer</summary><code>uint8</code> can only hold
   0–255. 300 overflows and wraps around
   (<code>300 % 256 == 44</code>) — NumPy performs the cast silently,
   with no bounds check by default.</details>

5. **What does `as_strided` actually do differently from a normal
   reshape or slice?**
   <details><summary>Answer</summary>It builds a view by directly
   specifying strides, letting the same underlying memory be read as
   overlapping windows — something no ordinary slice or reshape can
   express, since those only ever describe non-overlapping
   views.</details>

6. **Why does `downcast_safely` check `iinfo`/`finfo` range *before*
   casting, instead of casting and checking the result?**
   <details><summary>Answer</summary>Casting out-of-range values first
   would already have silently wrapped or corrupted them (as in question
   4) — by the time you inspect the result, the original values are
   gone. Checking the range beforehand catches the problem before any
   data is lost.</details>

## Interview questions this module answers

- *Does slicing a NumPy array copy it?*
- *I modified a slice and the original changed. Why?*
- *What does `.contiguous()` do in PyTorch and when do you need it?*
- *Why did summing this uint8 array give the wrong answer?*
- *How much memory does a batch of N images take?*
- *How would you compute a rolling mean over a 10M-element array?*
- *Why is `x == np.nan` always False?*
