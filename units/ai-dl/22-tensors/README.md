# 22 — Tensors

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/ai-dl/22-tensors/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/ai-dl/tensors?utm_source=github&utm_medium=labs)

*~45 minutes · 48 tests*

A `torch.Tensor` is a NumPy array plus three things: a **device**, an
optional **gradient history**, and a **memory format**. Almost everything
surprising about PyTorch comes from one of those three, and most of the
first-day bugs in a new codebase are device or dtype mismatches.

This module is the bridge. If you did `python/12-broadcasting` and
`python/13-array-memory`, the shape and view rules transfer unchanged — the
new material is devices, autograd-safety, and channels-last.

Work in `exercises.py`. Check your work with `python -m labkit check 22`, or open the notebook (`lab.ipynb`) in Colab.

Each section below ends with **Test it** — the exact command for just that
piece, run from `units/ai-dl/22-tensors/`. No need to finish the whole file
before checking your work.

## What you'll build

10 functions in `exercises.py`, in this order. Keep this table open as you
go — each section below tells you which row it's building:

| Function | You'll implement with | Shows up in real code as |
|---|---|---|
| `as_tensor_like` | `torch.as_tensor(...).to(reference)` | The device/dtype guard every custom `collate_fn`, augmentation, or inference server needs, since `torch.tensor(x)` always lands on CPU in the default dtype regardless of context |
| `shares_storage` | `.untyped_storage().data_ptr()` | Debugging silent aliasing — confirming an augmentation step actually copied pixels instead of mutating a batch tensor another worker or `DataLoader` is still reading |
| `safe_normalise` | non-mutating arithmetic, autograd-safety | A preprocessing/normalisation layer safe to call on a tensor still attached to the autograd graph — the same shape as `BatchNorm`'s internals |
| `flatten_batch` | `.flatten(1)` vs. `.view`/`.reshape` | The `nn.Flatten()` between a conv backbone and a classifier head — must survive a non-contiguous feature map coming out of an earlier `permute` |
| `to_channels_last` | `.contiguous(memory_format=torch.channels_last)` | Enabling channels-last training for a real speed-up on Tensor Cores, without touching any model code that only cares about logical shape |
| `batched_pairwise_distance` | `.unsqueeze` broadcasting, `.clamp_min` before `.sqrt` | Triplet/contrastive loss, k-NN retrieval, or the distance matrix behind a clustering step — and the classic NaN-loss-on-step-one bug |
| `masked_softmax` | `.masked_fill`, `torch.softmax`, `torch.where` | Attention over padded sequences — every transformer encoder that batches variable-length inputs |
| `count_parameters` | `.parameters()`, `.numel()`, `.requires_grad` | Reporting trainable-parameter count after freezing a backbone for fine-tuning, matching what a model-summary tool prints |
| `memory_bytes` | `.element_size()` | Back-of-envelope OOM diagnosis: does this batch even fit on the GPU, before you run it |
| `stable_logsumexp` | `.max()`, `.exp().sum().log()`, `torch.where` | The numerically-stable core of softmax/cross-entropy — interviewers ask for the derivation, not the built-in |

---

## 1. Device and dtype travel together

`torch.tensor(...)` always allocates fresh, on the CPU, in a default dtype —
**no matter what else is around it**:

```python
>>> torch.tensor([1, 2, 3])
tensor([1, 2, 3])
>>> torch.tensor([1, 2, 3]).dtype
torch.int64
```

That is the source of `Expected all tensors to be on the same device, but
found at least two devices, cuda:0 and cpu!` — the single most common error
message in PyTorch. `.to()` accepts a tensor and copies both attributes from
it in one call:

```python
>>> reference = torch.zeros(3, dtype=torch.float64)
>>> torch.as_tensor([1, 2, 3]).to(reference)
tensor([1., 2., 3.], dtype=torch.float64)
```

Also worth knowing:

- **`as_tensor` vs `tensor`.** `torch.tensor(x)` always copies;
  `torch.as_tensor(x)` shares memory with a NumPy array when it can. In a
  data pipeline that difference is real throughput.
- **`.to()` is a no-op when nothing changes**, so calling it defensively
  costs nothing.
- **`torch.zeros_like(x)` / `empty_like(x)`** inherit device, dtype *and*
  layout. Prefer them to naming a device explicitly.

**In `exercises.py`:** `as_tensor_like` is this section's target — turn
`data` into a tensor matching `reference`'s device and dtype in one call.

**Test it:** `python3 -m unittest test_exercises.TestAsTensorLike -v`

## 2. Views and copies — same rules as NumPy

| Returns a view | Returns a copy |
|---|---|
| `x[2:5]`, `x[:, 0]` | `x[[0, 2, 4]]` |
| `x.view(...)`, `x.permute(...)` | `x[x > 5]` |
| `x.transpose(0, 1)`, `x.expand(...)` | `x.clone()`, `x.flatten()` when non-contiguous |
| `x.reshape(...)` *when it can* | `x.reshape(...)` when it cannot |

`x.untyped_storage().data_ptr()` is the definitive check — two tensors that
return the same pointer read the same memory, whatever their shape or
dtype view happens to look like:

```python
>>> x = torch.arange(12).reshape(3, 4)
>>> view = x.view(4, 3)
>>> view.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
True
>>> copy = x[[0, 2]]                    # fancy indexing
>>> copy.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
False
```

`x._base` tells you what a view points at — `None` for a tensor that owns
its storage, and the original tensor for one that doesn't:

```python
>>> leaf = torch.zeros(3, 4)
>>> leaf._base is None
True
>>> leaf[:, 0]._base is leaf
True
```

**`expand` vs `repeat`.** `expand` sets a stride to 0 and costs nothing;
`repeat` allocates. `expand` is what you want for broadcasting a mask; it
is also read-only in practice, because writing into a stride-0 axis writes
to every position at once.

**In `exercises.py`:** `shares_storage` is this section's target — return
whether two tensors read the same underlying memory.

**Test it:** `python3 -m unittest test_exercises.TestSharesStorage -v`

## 3. `view` fails where `reshape` works

```python
>>> x = torch.zeros(2, 3, 4).permute(0, 2, 1)
>>> x.is_contiguous()
False
>>> x.view(2, -1)
RuntimeError: view size is not compatible with input tensor's size and
stride (at least one dimension spans across two contiguous subspaces). Use
.reshape(...) instead.
>>> x.reshape(2, -1).shape          # fine — copies
torch.Size([2, 12])
>>> x.flatten(1).shape              # fine, and says what you mean
torch.Size([2, 12])
```

`view` requires the new shape to be expressible with the *existing*
strides. After any `permute` or `transpose` it usually is not. The error is
loud and its suggested fix (`.contiguous().view(...)`) is exactly what
`reshape` does for you.

Use `flatten(1)` when the intent is "keep the batch dimension" — it cannot
be got wrong the way `view(x.shape[0], -1)` can.

**In `exercises.py`:** `flatten_batch` is this section's target — `(B,
...)` to `(B, N)`, keeping the batch dimension, and working whether or not
the input is contiguous.

**Test it:** `python3 -m unittest test_exercises.TestFlattenBatch -v`

## 4. In-place operations have two costs

```python
x -= x.mean()          # mutates the caller's tensor
```

The first cost is the NumPy one: if `x` is a view, its base changes too.
The second is specific to PyTorch — an in-place op on a tensor that
`requires_grad` and is a graph leaf is refused outright:

```python
>>> x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
>>> x -= x.mean()
RuntimeError: a leaf Variable that requires grad is being used in an
in-place operation.
```

and, more insidiously, autograd may need the *original* value of a
non-leaf tensor for the backward pass. Overwriting it produces a different
error that only fires later, at `.backward()`, far from the line that
caused it:

```python
>>> y = x.exp()
>>> z = y.sum()
>>> y += 1
>>> z.backward()
RuntimeError: one of the variables needed for gradient computation has
been modified by an inplace operation: ... is at version 1; expected
version 0 instead.
```

In-place is still the right choice when you are memory-bound and outside
the graph — `optimiser.step()` is entirely in-place, and `out=` on a large
elementwise op saves a full allocation. Just do it deliberately.

**In `exercises.py`:** `safe_normalise` is this section's target — zero
mean, unit variance, without mutating `x` and without breaking a graph `x`
is part of.

**Test it:** `python3 -m unittest test_exercises.TestSafeNormalise -v`

## 5. Memory format: channels-last

A `(B, C, H, W)` tensor is stored with W varying fastest. **Channels-last**
keeps the same logical shape and re-orders memory so C varies fastest:

```python
>>> x = torch.zeros(2, 3, 4, 5)
>>> y = x.contiguous(memory_format=torch.channels_last)
>>> y.shape
torch.Size([2, 3, 4, 5])
>>> y.stride()
(60, 1, 15, 3)
>>> y.is_contiguous(memory_format=torch.channels_last)
True
```

Convolution kernels and Tensor Cores want channels-last, so
`model.to(memory_format=torch.channels_last)` plus the same on the input is
a real speed-up on modern GPUs for free. The shape does not change, so
nothing downstream needs editing — which is exactly why it is easy to
forget it exists.

This is the same contiguity question as NumPy's, and the same reason
`.contiguous()` before a `.view()` shows up everywhere.

**In `exercises.py`:** `to_channels_last` is this section's target —
`(B, C, H, W)` in, memory laid out as `(B, H, W, C)` out, shape unchanged.

**Test it:** `python3 -m unittest test_exercises.TestChannelsLast -v`

## 6. Broadcasting a pairwise distance, and the sqrt-at-zero NaN

`torch.cdist` is the built-in for "distance between every pair of rows,"
but writing it by hand is the point — it's `python/12-broadcasting`'s
unsqueeze pattern applied to a real op. Given `a: (B, N, D)` and
`b: (B, M, D)`, inserting a size-1 axis in a different spot on each makes
them broadcast against each other elementwise:

```python
>>> a = torch.randn(2, 3, 5)
>>> b = torch.randn(2, 4, 5)
>>> diff = a.unsqueeze(2) - b.unsqueeze(1)   # (2,3,1,5) vs (2,1,4,5)
>>> diff.shape
torch.Size([2, 3, 4, 5])
>>> diff.pow(2).sum(-1).shape                # -> (B, N, M)
torch.Size([2, 3, 4])
```

```python
>>> a = torch.tensor([[[0.0, 0.0]]])
>>> b = torch.tensor([[[3.0, 4.0], [0.0, 0.0]]])
>>> diff = a.unsqueeze(2) - b.unsqueeze(1)
>>> diff.pow(2).sum(-1).clamp_min(1e-12).sqrt()
tensor([[[5.0000e+00, 1.0000e-06]]])
```

Look closely at that second entry: the true distance from `(0, 0)` to
itself is `0`, but the output is `1e-06`, not `0`. That is `clamp_min(eps)`
doing its job. `sqrt` has an **infinite derivative at 0**, so without the
clamp, the squared distance of a point to itself (`0`) feeds `sqrt`
directly, the chain rule computes `0 * inf = NaN`, and the loss goes to NaN
on the first training step — clamping to exactly `0` does not fix this,
because the gradient is only cut off when the input is *strictly below*
the clamp:

```python
>>> a = torch.zeros(1, 2, 3, requires_grad=True)
>>> naive = a.pow(2).sum(-1).sqrt()          # no clamp
>>> naive.sum().backward()
>>> torch.isnan(a.grad).any().item()
True
```

This bites every triplet/contrastive loss ever written.

**In `exercises.py`:** `batched_pairwise_distance` is this section's
target — the unsqueeze-broadcast-then-clamp pattern above, in full.

**Test it:** `python3 -m unittest test_exercises.TestBatchedPairwiseDistance -v`

## 7. Masking before the softmax, never after

```python
weights = torch.softmax(scores, -1)
weights = weights * mask               # WRONG — the row no longer sums to 1
```

The excluded positions must be `-inf` **before** the exponential:

```python
>>> scores = torch.tensor([[1.0, 1.0, 100.0]])
>>> mask = torch.tensor([[True, True, False]])
>>> neg_inf = torch.finfo(scores.dtype).min
>>> masked = scores.masked_fill(~mask, neg_inf)
>>> torch.softmax(masked, dim=-1)
tensor([[0.5000, 0.5000, 0.0000]])
```

Compare that to the wrong order: softmax-then-zero would leave `[0.5, 0.5,
0.0]` scaled from a row that still included the huge `100.0` logit, and the
kept entries would no longer sum to 1.

Two details:

- Use `torch.finfo(dtype).min`, not `float("-inf")`. Literal `-inf`
  produces NaN as soon as a whole row is masked (`softmax` of all `-inf` is
  `0/0`), and it breaks under float16 autocast.
- A **fully masked row** — a padded sequence attending to nothing — still
  needs handling. The reference returns all-zero for it, which is what an
  attention layer wants; leaving it as a uniform distribution over garbage
  is a quiet correctness bug:

```python
>>> masked_softmax(torch.randn(1, 4), torch.zeros(1, 4, dtype=torch.bool))
tensor([[0., 0., 0., 0.]])
```

**In `exercises.py`:** `masked_softmax` is this section's target — mask
before the exponential, and return all-zero (not NaN) for a fully masked
row.

**Test it:** `python3 -m unittest test_exercises.TestMaskedSoftmax -v`

## 8. Numerical stability, again

```python
log Σ exp(x)  =  m + log Σ exp(x - m),   m = max(x)
```

`exp(1000)` is `inf`; `log(inf)` is `inf`. Subtracting the max caps the
largest exponent at 1 and changes nothing mathematically:

```python
>>> x = torch.tensor([[1000.0, 1001.0, 1002.0]])
>>> x.exp().sum(-1).log()             # naive — overflows
tensor([inf])
>>> stable_logsumexp(x)
tensor([1002.4076])
```

This is the same trick as the stable softmax and it is worth being able to
derive on demand — interviewers ask for the derivation, not for
`torch.logsumexp`.

The edge case the reference handles: an all-`-inf` row would give
`-inf + inf = NaN` once the max is subtracted back in, so the shift itself
is clamped to a finite value (`0`) whenever the row's max isn't finite.

**In `exercises.py`:** `stable_logsumexp` is this section's target — the
max-subtraction derivation above, plus the all-`-inf`-row edge case.

**Test it:** `python3 -m unittest test_exercises.TestStableLogSumExp -v`

## 9. Counting parameters and bytes

Two independent pieces of arithmetic that both come up when a model
doesn't fit: how many learnable numbers it has, and how many bytes one
tensor costs.

### Parameters

```python
>>> model = torch.nn.Linear(3, 4)
>>> count_parameters(model)             # 3*4 weights + 4 biases
16
>>> model.weight.requires_grad_(False)
>>> count_parameters(model)             # only the bias is trainable now
4
>>> count_parameters(model, trainable_only=False)
16
```

`p.numel()` counts scalars, not tensors — a `(3, 4)` weight is 12
parameters, not 1. Filtering on `requires_grad` is what makes the number
match a model summary after freezing a backbone. Buffers (`BatchNorm`'s
running mean/var) are not parameters and must not be counted:

```python
>>> bn = torch.nn.BatchNorm1d(4)        # 4 weight + 4 bias, plus 2 buffers
>>> count_parameters(bn)
8
```

**In `exercises.py`:** `count_parameters` is this subsection's target —
sum `numel()` over `module.parameters()`, optionally filtered on
`requires_grad`.

**Test it:** `python3 -m unittest test_exercises.TestCountParameters -v`

### Bytes

```python
32 × 3 × 224 × 224 × 4 bytes ≈ 19 MB
```

```python
>>> memory_bytes((32, 3, 224, 224))
19267584
>>> memory_bytes((1000,), torch.float16)
2000
```

per copy of one image batch — and the *activations* behind it are many
times that, because every layer's output stays alive for the backward
pass. Being able to do this arithmetic in your head is what turns "it
OOMs" into "the activations are the problem, so gradient checkpointing or
a smaller batch".

**In `exercises.py`:** `memory_bytes` is this subsection's target —
`element_count × dtype.element_size()`, with no tensor actually allocated.

**Test it:** `python3 -m unittest test_exercises.TestMemoryBytes -v`

---

## The exercises

| Function | What it teaches |
|---|---|
| `as_tensor_like` | Device and dtype in one call |
| `shares_storage` | Views vs copies; `data_ptr` |
| `safe_normalise` | Not mutating the caller; in-place vs autograd |
| `flatten_batch` | `view` vs `reshape` on non-contiguous input |
| `to_channels_last` | Memory format; shape unchanged, strides moved |
| `batched_pairwise_distance` | `unsqueeze` broadcasting; the sqrt-at-zero NaN |
| `masked_softmax` | Mask before the exponential; fully masked rows |
| `count_parameters` | `numel`, `requires_grad`, buffers are not parameters |
| `memory_bytes` | Sizing a batch |
| `stable_logsumexp` | The max-subtraction derivation |

## Check your understanding

1. **Why does `view` fail on a non-contiguous tensor when `reshape`
   doesn't?**
   <details><summary>Answer</summary><code>view</code> requires the new
   shape to be expressible with the existing strides, without moving any
   data. <code>reshape</code> does the same when possible, but silently
   copies to a contiguous layout first when it isn't.</details>

2. **`to_channels_last` changes a tensor's strides but not its logical
   shape. What's actually different about the memory afterward?**
   <details><summary>Answer</summary>The channel dimension becomes the
   fastest-varying one in memory instead of the spatial dimensions — the
   shape reported by <code>.shape</code> is unchanged, but the physical
   layout (and therefore which operations are cheap) is
   different.</details>

3. **Why does `batched_pairwise_distance` need a small epsilon before
   the square root?**
   <details><summary>Answer</summary><code>sqrt</code>'s gradient is
   infinite at exactly 0 — for two identical points (distance 0), a
   direct <code>sqrt(0)</code> produces a NaN gradient during
   backprop, even though the forward value is correct.</details>

4. **Why must a mask be applied *before* the softmax's exponential, not
   after?**
   <details><summary>Answer</summary>Masking after softmax would zero
   out already-normalized probabilities, breaking the property that
   they sum to 1 — masking before (setting masked positions to
   <code>-inf</code>) makes their contribution to the exponential
   correctly vanish while the remaining values still normalize
   properly.</details>

5. **`count_parameters` needs to know about buffers to work correctly.
   Why aren't buffers just parameters?**
   <details><summary>Answer</summary>Buffers (like a `BatchNorm`
   running mean) are part of a module's state but aren't updated by
   the optimizer via gradients — counting them as parameters would
   overstate the model's actual learnable capacity.</details>

6. **Why does `stable_logsumexp` subtract the max before taking the
   log of the sum of exponentials?**
   <details><summary>Answer</summary>Same reason as softmax's max-
   subtraction trick: exponentiating a large value overflows to `inf`
   before you ever get to take the log. Subtracting the max first keeps
   every exponent bounded, and the constant is added back afterward
   without changing the result.</details>

## Interview questions this module answers

- *What is the difference between `view` and `reshape`?*
- *Why does `x[1] *= 2` change the original tensor?*
- *What does `.contiguous()` do and when do you need it?*
- *Why is my loss NaN from the very first step?*
- *How do you mask padding in attention?*
- *How much memory does this batch take, and where does the rest go?*
- *What does `.to(device)` actually move, and what does it not?*
