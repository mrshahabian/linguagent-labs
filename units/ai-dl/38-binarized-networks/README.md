# 38 — Binarized and quantized neural networks

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/ai-dl/38-binarized-networks/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/ai-dl/binarized-networks?utm_source=github&utm_medium=labs)

*~50 minutes · 18 tests*

Module 30 covers int8 quantization — a softer compression that keeps 256
representable levels per weight. This module goes to the extreme end of
the same spectrum: weights (and sometimes activations) collapsed to a
single bit, `{-1, +1}`. Getting there needs one genuinely new mechanism —
training *through* a function with almost no gradient at all — and the
efficiency argument behind it is different in kind from int8's, not just
degree.

Work in `exercises.py`. Check your work with `python -m labkit check 38`, or open the notebook (`lab.ipynb`) in Colab.

Each section below ends with **Test it** — the exact command for just that
piece, run from `units/ai-dl/38-binarized-networks/`.

## What you'll build

| Name | You'll implement | Shows up in real code as |
|---|---|---|
| `BinarizeSTE` | `forward`, `backward` | The straight-through estimator — training through a non-differentiable function |
| `binarize` | the function | The call site every binarized layer uses |
| `BinaryLinear` | `forward` | BinaryConnect's "real latent weight, binarized forward" pattern |
| `xnor_popcount_dot` | the function | The actual hardware efficiency argument for binarization |
| `memory_footprint_bytes` | the function | The ~32x compression claim, in bytes |

---

## 1. Training through a step function: the straight-through estimator

```python
@staticmethod
def forward(ctx, x):
    ctx.save_for_backward(x)
    return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

@staticmethod
def backward(ctx, grad_output):
    (x,) = ctx.saved_tensors
    pass_through = (x.abs() <= 1).float()
    return grad_output * pass_through
```

`sign(x)` is a step function: its derivative is zero everywhere except
at `x = 0`, where it's undefined. Backpropagating through it honestly
would zero out every gradient before it reaches an earlier layer — there
is no direction in which nudging `x` a tiny amount changes a step
function's output, almost anywhere.

The **straight-through estimator** (STE) is a deliberate lie, told only
in the backward pass: pretend the forward function was the identity, and
let the gradient through unchanged. The one refinement here — clipping
the pass-through to `|x| <= 1` — reflects that `sign` has already
*saturated* well before `|x|` reaches large values; a gradient nudging
`x` from 5 to 6 can't possibly change the binarized output, so passing
one through there would only push `x` needlessly further from zero. This
is the "clipped" STE from BinaryConnect and XNOR-Net, as opposed to an
unclipped version that passes every gradient through regardless of `x`.

```python
>>> x = torch.tensor([-5.0, 0.2, 5.0], requires_grad=True)
>>> binarize(x).sum().backward()
>>> x.grad
tensor([0., 1., 0.])
```

**In `exercises.py`:** `BinarizeSTE.forward`, `BinarizeSTE.backward`,
`binarize`.

**Test it:** `python3 -m unittest test_exercises.TestBinarize -v`

## 2. BinaryConnect: real latent weights, binarized forward

```python
def forward(self, x):
    binary_weight = binarize(self.weight)
    return x @ binary_weight.T
```

`self.weight` stays an ordinary real-valued `float32` parameter — it is
what the optimizer actually updates, and it is never literally `±1`
itself. Every forward call binarizes a **fresh view** of it via the STE;
gradients (small, real-valued nudges) accumulate in this "latent" weight
even though the forward computation only ever multiplies by `+1` or
`-1`. This is BinaryConnect's core move: plain gradient descent cannot
optimize a genuinely discrete weight directly (there is no small nudge
that changes a discrete value, by definition), so instead it optimizes a
continuous *proxy* that happens to get snapped to a discrete value on
every forward pass.

```python
>>> layer = BinaryLinear(8, 4)
>>> layer(torch.randn(1, 8)).shape
torch.Size([1, 4])
>>> layer.weight.abs().max() < 1   # the latent weight, still real-valued
True
```

**In `exercises.py`:** `BinaryLinear.forward`.

**Test it:** `python3 -m unittest test_exercises.TestBinaryLinear -v`

## 3. XNOR + popcount: the actual efficiency argument

```python
a_bits = (a_signs > 0)
b_bits = (b_signs > 0)
agreements = torch.sum(~(a_bits ^ b_bits)).item()
return 2 * agreements - a_signs.numel()
```

For vectors whose entries are always exactly `+1` or `-1`: two entries'
product is `+1` when they **agree** in sign, and `-1` when they
**disagree**. Encoding `+1` as bit `1` and `-1` as bit `0`, XNOR-ing the
two bit patterns produces a `1` exactly where they agree — so
`popcount(xnor(a, b))` counts the agreements directly, and the dot
product falls out as `agreements - disagreements = 2 * agreements - N`.

This is the actual hardware argument behind binarization, and it's
different in *kind* from int8 quantization's savings, not just degree:
on hardware with a popcount instruction, a binary dot product replaces
`N` floating-point multiply-adds with **one XNOR and one popcount**
over `N` bits — no floating-point multiplication happens at all. int8
quantization still does integer multiplication; binarization can skip
multiplication entirely.

```python
>>> a = torch.tensor([1., -1., 1., -1.])
>>> xnor_popcount_dot(a, a)          # identical vectors: full agreement
4
>>> xnor_popcount_dot(a, -a)         # opposite vectors: full disagreement
-4
```

**In `exercises.py`:** `xnor_popcount_dot`.

**Test it:** `python3 -m unittest test_exercises.TestXnorPopcountDot -v`

## 4. The compression claim, in bytes

```python
float32_bytes = num_params * 4
binary_bytes = (num_params + 7) // 8
```

float32 spends 4 bytes per weight; a binarized weight needs exactly one
**bit**, and bits pack 8 to a byte. That's a ~32x reduction — compare
int8 quantization's ~4x (module 30) for the same parameter count. The
two are not mutually exclusive in principle (a model can mix int8
layers with binarized ones), but binarization sits at the far,
more-accuracy-costly end of the same size-vs-precision spectrum
quantization occupies.

```python
>>> memory_footprint_bytes(8_000_000)
(32000000, 1000000)   # float32 vs. binary, same parameter count
```

**In `exercises.py`:** `memory_footprint_bytes`.

**Test it:** `python3 -m unittest test_exercises.TestMemoryFootprintBytes -v`

---

## The exercises

| Function / class | What it teaches |
|---|---|
| `BinarizeSTE` | The straight-through estimator; clipping outside `[-1, 1]` |
| `binarize` | The call site wrapping `BinarizeSTE.apply` |
| `BinaryLinear` | Real latent weights trained through a binarized forward pass |
| `xnor_popcount_dot` | Binary dot products need no floating-point multiplication |
| `memory_footprint_bytes` | The ~32x compression claim, computed exactly |

Every function in this table has a stub in `exercises.py` and test
coverage in `test_exercises.py` — a clean 1:1 mapping.

## Check your understanding

1. **Why can't you just backpropagate through `sign(x)` honestly?**
   <details><summary>Answer</summary><code>sign</code> is a step
   function — its derivative is zero everywhere it's defined (and
   undefined exactly at <code>x = 0</code>). An honest backward pass
   would multiply every upstream gradient by zero almost everywhere,
   which would stop any learning signal from ever reaching the
   parameters that produced <code>x</code>.</details>

2. **What does the straight-through estimator actually do, in one
   sentence?**
   <details><summary>Answer</summary>It pretends, only during the
   backward pass, that the forward function was the identity — letting
   the incoming gradient pass through largely unchanged instead of being
   multiplied by the (mostly zero) true derivative of
   <code>sign</code>.</details>

3. **Why does `BinaryLinear` keep a real-valued `self.weight` instead of
   storing the weight as literally `+1`/`-1` values?**
   <details><summary>Answer</summary>Plain gradient descent needs to
   apply small, continuous nudges to a parameter — there's no small
   nudge that changes a value that's constrained to be exactly
   <code>+1</code> or <code>-1</code>. Keeping a real-valued "latent"
   weight lets ordinary gradient updates accumulate normally; it's only
   binarized fresh on each forward pass to produce the actual
   computation.</details>

4. **Why is XNOR-popcount described as a different KIND of efficiency
   gain from int8 quantization, not just a bigger one?**
   <details><summary>Answer</summary>int8 quantization still performs
   integer multiplication, just with fewer bits per operand than
   float32. XNOR-popcount replaces multiplication entirely — a binary
   dot product is computed with a bitwise XNOR and a population count,
   with no multiply operation involved at all, which is a categorically
   different computation, not merely a narrower one.</details>

5. **A model needs to fit in a much smaller memory budget than int8
   quantization achieves. What's the further step, and what does it
   cost?**
   <details><summary>Answer</summary>Binarizing weights (and optionally
   activations) to a single bit each, roughly an 8x reduction beyond
   int8's own 4x (so ~32x versus float32 total). The cost is a much
   larger drop in model accuracy than int8 quantization typically
   causes, since a weight now has only two possible values instead of
   256.</details>

## Interview questions this module answers

- *How do you train a network whose weights are binary, given that
  gradient descent needs small continuous updates?*
- *What is the straight-through estimator, and where else does the same
  trick show up?*
- *Why is a binarized network's hardware speedup different from int8
  quantization's, not just a bigger version of the same thing?*
- *Compare the memory savings of int8 quantization versus binarization
  for the same model.*
