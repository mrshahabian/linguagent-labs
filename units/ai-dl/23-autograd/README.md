# 23 — Autograd

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/ai-dl/23-autograd/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/ai-dl/autograd?utm_source=github&utm_medium=labs)

*~50 minutes · 46 tests*

Autograd builds a graph as you compute. Every operation on a tensor with
`requires_grad=True` records a node holding the inputs it will need for the
backward pass. `.backward()` walks that graph in reverse applying the chain
rule, then throws it away.

Almost every autograd bug is one of three things: **the graph was never
built**, **the graph was built and thrown away too early**, or **gradients
accumulated when you thought they were replaced**. This module makes each
one visible.

Work in `exercises.py`. Check your work with `python -m labkit check 23`, or open the notebook (`lab.ipynb`) in Colab.

Each section below ends with **Test it** — the exact command for just that
piece, run from `units/ai-dl/23-autograd/`. No need to finish the whole
file before checking your work.

## What you'll build

Nine functions and classes in `exercises.py`, in this order. Keep this
table open as you go — each section below tells you which row it's
building:

| Name | You'll implement | Shows up in real code as |
|---|---|---|
| `grad_of` | `grad_of` | A functional gradient — a saliency map, a metric, or a penalty term computed without disturbing the `.grad` state the optimiser is about to read |
| `accumulate_twice` | `accumulate_twice` | Gradient accumulation — simulating a large batch on hardware that only fits a small one, and the exact mechanism a missing `zero_grad()` corrupts |
| `frozen_forward` | `frozen_forward` | A no-graph forward pass — the memory-saving half of real inference (the other half is `model.eval()`) |
| `detach_vs_clone` | `detach_vs_clone` | Choosing between "take the value" (`loss.item()`, logging) and "take a differentiable copy" (an auxiliary loss branch) |
| `clip_grad_norm` | `clip_grad_norm` | Gradient clipping — the standard defence against exploding gradients in RNNs and Transformers, and the norm you log to catch a diverging run early |
| `Square` | `forward`, `backward` | A hand-written `autograd.Function` — the shape of any custom or fused kernel you wrap for PyTorch, e.g. a custom CUDA op |
| `StraightThroughRound` | `forward`, `backward` | The gradient estimator behind quantisation-aware training and the VQ-VAE codebook |
| `numeric_grad` | `numeric_grad` | Central-difference checking — how you prove a hand-written `backward` is correct before trusting it in training (`torch.autograd.gradcheck`) |
| `gradient_penalty` | `gradient_penalty` | A second-order objective — WGAN-GP's gradient penalty, and the same `create_graph=True` mechanism behind MAML |

---

## 1. `torch.autograd.grad` vs `.backward()`

```python
torch.autograd.grad(output, inputs)     # returns them
output.backward()                        # accumulates into .grad
```

Use the functional form for anything that is not the training step — a
penalty term, a metric, a saliency map — because it does not disturb the
`.grad` state the optimiser is about to read.

```python
>>> x = torch.tensor([1.0, 2.0, 3.0])
>>> grad_of(lambda t: (t ** 3).sum(), x)
tensor([ 3., 12., 27.])
>>> x.grad is None      # untouched - grad_of never assigns into it
True
```

**In `exercises.py`:** `grad_of` is the only function this section asks
for — note it must work even when `x` doesn't already have
`requires_grad=True` (the reference solution detaches and re-enables it),
since a metric is often computed on a tensor nobody thought to make
differentiable.

**Test it:** `python3 -m unittest test_exercises.TestGradOf -v`

## 2. `.grad` accumulates, and the graph is freed after one backward

Two related facts about `.backward()`, both of which explain real bugs, and
both of which the same exercise exercises.

**It accumulates. It is not assigned.**

```python
(x ** 2).sum().backward(retain_graph=True)
(x ** 2).sum().backward()
x.grad          # 4x, not 2x
```

This surprises everyone once, and it is deliberate: it is what makes
**gradient accumulation** work, letting you simulate a batch of 256 on
hardware that fits 32.

```python
for i, batch in enumerate(loader):
    (loss_fn(model(batch)) / accumulation_steps).backward()
    if (i + 1) % accumulation_steps == 0:
        optimiser.step()
        optimiser.zero_grad()
```

It is also why a training loop that forgets `zero_grad()` still runs, still
decreases the loss, and just trains badly — the worst possible failure
mode. `zero_grad(set_to_none=True)` is the default now: it frees the
gradient tensors instead of filling them with zeros, saving memory and
skipping a kernel — the visible difference is that `p.grad` is `None`
rather than a zero tensor.

**The graph is freed as soon as one backward pass has used it.**

```
RuntimeError: Trying to backward through the graph a second time (or
directly access saved tensors after they have already been freed).
```

Holding the saved buffers around is the expensive part of autograd, so
they're released the moment they've been consumed. `retain_graph=True`
keeps them alive for a second pass — reach for it only when you genuinely
need two backward passes through one graph. Nine times out of ten this
error means you accidentally kept a tensor from a previous iteration alive
(the classic: `total_loss += loss` instead of `total_loss += loss.item()`,
which grows the graph across the entire epoch until you run out of
memory).

```python
>>> x = torch.tensor([1.0, 2.0, 3.0])
>>> accumulate_twice(x)
tensor([ 4.,  8., 12.])
```

For `x = [1, 2, 3]`, `d(x**2)/dx = 2x = [2, 4, 6]`; two accumulated
backward passes (the second needs `retain_graph=True` on the first, since
the same graph is walked twice) give `2 * 2x = [4, 8, 12]`.

**In `exercises.py`:** `accumulate_twice` is this section's only target —
both behaviours above (accumulation, and needing `retain_graph=True` for
the second pass) are exercised by the same function.

**Test it:** `python3 -m unittest test_exercises.TestAccumulation -v`

## 3. `no_grad` is not `eval()`

Two different things that both get called "inference mode":

| | What it does |
|---|---|
| `torch.no_grad()` | Stops the graph being recorded. Saves memory. |
| `model.eval()` | Dropout off; BatchNorm uses running stats. |

Real inference needs **both**. Using only `eval()` wastes memory recording
a graph nobody will use. Using only `no_grad()` leaves dropout active and
BatchNorm normalising by the batch — so your predictions depend on what
else happened to be in the batch, which is a genuinely confusing bug to
chase.

`torch.inference_mode()` is a stricter, faster `no_grad` (the resulting
tensors can never be used in autograd at all). Prefer it for serving.

```python
>>> torch.manual_seed(0)
>>> model = torch.nn.Linear(4, 2)
>>> x = torch.randn(3, 4)
>>> out = frozen_forward(model, x)
>>> out.requires_grad, out.grad_fn
(False, None)
>>> torch.is_grad_enabled()   # no_grad() does not leak past the call
True
```

**In `exercises.py`:** `frozen_forward` is the only function here. Note it
must not call `model.eval()` or `model.train()` itself — the tests check
that whatever training mode the model was already in survives the call
unchanged, since `no_grad` and `eval()` are orthogonal.

**Test it:** `python3 -m unittest test_exercises.TestFrozenForward -v`

## 4. `detach` vs `clone`

```python
x.detach()          # shares storage, drops the graph
x.clone()           # copies storage, KEEPS the graph
x.detach().clone()  # separate tensor, no graph — the safe both
```

`detach()` sharing storage matters: writing into the detached tensor
silently corrupts `x`, and the gradient computed from it. When in doubt,
`.detach().clone()`.

The mnemonic: **detach takes the value, clone takes a differentiable
copy.**

For logging, `loss.item()` is the right call — it detaches *and* leaves the
GPU, so nothing keeps the graph alive.

```python
>>> x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True) * 2
>>> detached, cloned = detach_vs_clone(x)
>>> detached.requires_grad, cloned.requires_grad
(False, True)
>>> detached.data_ptr() == x.data_ptr()   # shares storage
True
>>> cloned.data_ptr() == x.data_ptr()     # separate storage
False
```

**In `exercises.py`:** `detach_vs_clone` is the only function here.

**Test it:** `python3 -m unittest test_exercises.TestDetachVsClone -v`

## 5. Gradient clipping is global

```python
total = sqrt(Σ_p ||p.grad||²)
if total > max_norm:
    for p in parameters:
        p.grad *= max_norm / total
```

Every parameter is treated as **one long vector**. All of them are scaled
by the *same* factor, so the update **direction** is preserved and only its
length changes. Clipping each tensor separately would change the direction
— a different, worse algorithm.

Return the norm *before* clipping and log it. A spike there is the earliest
warning that a run is about to diverge, and it shows up several steps
before the loss does.

`clip_grad_norm_` must be called **after** `backward()` and **before**
`step()`. With AMP it must also come after `scaler.unscale_(optimiser)`, or
you are clipping scaled gradients and the threshold means nothing.

```python
>>> p = torch.nn.Parameter(torch.zeros(2))
>>> p.grad = torch.tensor([3.0, 4.0])
>>> clip_grad_norm([p], max_norm=1.0)
tensor(5.)
>>> p.grad          # scaled in place to norm 1.0, direction unchanged
tensor([0.6000, 0.8000])
```

**In `exercises.py`:** `clip_grad_norm` is the only function here.

**Test it:** `python3 -m unittest test_exercises.TestClipGradNorm -v`

## 6. Custom `autograd.Function`

```python
class Square(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x ** 2

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * 2 * x
```

`backward` receives dL/dy and returns dL/dx — one return value per input to
`forward`. Rules:

- **`save_for_backward`, not `ctx.x = x`.** Only the former lets autograd
  detect that a saved tensor was modified in place before backward ran.
- **Check it with `gradcheck`.** `torch.autograd.gradcheck(fn, (x,))` with a
  float64 input compares against central differences. It is the standard
  way to prove a hand-written kernel is right, and it is what an
  interviewer wants to hear when you say you wrote a custom op.

```python
>>> x = torch.tensor([2.0, 3.0], requires_grad=True)
>>> Square.apply(x)
tensor([4., 9.], grad_fn=<SquareBackward>)
>>> Square.apply(x).sum().backward()
>>> x.grad          # d(x^2)/dx = 2x
tensor([4., 6.])
>>> xd = torch.randn(4, dtype=torch.float64, requires_grad=True)
>>> torch.autograd.gradcheck(Square.apply, (xd,))
True
```

**In `exercises.py`:** `Square.forward` and `Square.backward`.

**Test it:** `python3 -m unittest test_exercises.TestSquareFunction -v`

### The straight-through estimator

`round()` has a derivative of zero almost everywhere, so a quantised
network would receive no gradient at all:

```python
>>> x = torch.tensor([0.2, 0.7], requires_grad=True)
>>> torch.round(x).sum().backward()
>>> x.grad
tensor([0., 0.])
```

The straight-through estimator simply lies about this in `backward`:

```python
@staticmethod
def backward(ctx, grad_output):
    return grad_output          # pretend the forward was the identity
```

This is how quantisation-aware training and the VQ-VAE codebook work. Being
able to name it, and say *why* the lie is acceptable (the forward is
"close enough" to the identity that the gradient direction is still
useful), is a good senior-level answer.

```python
>>> x = torch.tensor([0.2, 0.7, -1.4])
>>> StraightThroughRound.apply(x)
tensor([ 0.,  1., -1.])
>>> x2 = torch.tensor([0.2, 0.7], requires_grad=True)
>>> StraightThroughRound.apply(x2).sum().backward()
>>> x2.grad          # the "lie": gradient passes through unchanged, unlike torch.round above
tensor([1., 1.])
```

**In `exercises.py`:** `StraightThroughRound.forward` and
`StraightThroughRound.backward`. Coupled to §6 above only in that it's the
same protocol (`torch.autograd.Function`) applied a second time — nothing
in `Square` needs finishing first.

**Test it:** `python3 -m unittest test_exercises.TestStraightThrough -v`

## 7. Numerical gradient checking

Central difference, in float64:

```python
(f(x + h) - f(x - h)) / (2h)
```

Error O(h²), versus O(h) for the one-sided version — whose error swamps the
thing you are trying to verify. Float32 rounding noise is larger than the
signal at any useful `h`, which is why `gradcheck` insists on doubles.

```python
>>> def f(t):
...     return (t ** 3).sum()
...
>>> x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
>>> numeric_grad(f, x)
tensor([ 3.0000, 12.0000, 27.0000], dtype=torch.float64)
>>> 3 * x ** 2                    # the closed form, for comparison
tensor([ 3., 12., 27.], dtype=torch.float64)
```

**In `exercises.py`:** `numeric_grad` is the only function here.

**Test it:** `python3 -m unittest test_exercises.TestNumericGrad -v`

One test in that class, `test_matches_autograd_on_a_network`, also calls
`grad_of` (§1) to produce the analytic gradient it compares against — so
that single test needs `grad_of` finished too, not just `numeric_grad`. The
other three tests in the class don't.

## 8. `create_graph=True` — differentiating a gradient

```python
gradients = torch.autograd.grad(output.sum(), x, create_graph=True)[0]
penalty = gradients.pow(2).sum()
penalty.backward()          # only works because of create_graph
```

Without it the gradient is a constant and the penalty contributes nothing —
it does not error, it just silently does nothing. With it, the gradient
computation is itself part of the graph and can be differentiated again.

This is the mechanism behind WGAN-GP, MAML and every second-order method.
It roughly doubles memory, which is the trade-off to mention.

```python
>>> layer = torch.nn.Linear(3, 1, bias=False)
>>> with torch.no_grad():
...     layer.weight.copy_(torch.tensor([[1.0, 2.0, 2.0]]))
...
>>> penalty = gradient_penalty(layer, torch.randn(4, 3))
>>> penalty          # y = Wx -> dy/dx = W -> ||W||^2 = 1 + 4 + 4
tensor(9., grad_fn=<MeanBackward0>)
>>> penalty.backward()
>>> layer.weight.grad          # d/dW of ||W||^2 is 2W
tensor([[2., 4., 4.]])
```

**In `exercises.py`:** `gradient_penalty` is the only function here.

**Test it:** `python3 -m unittest test_exercises.TestGradientPenalty -v`

---

## The exercises

| Function | What it teaches |
|---|---|
| `grad_of` | `autograd.grad` vs `.backward()` |
| `accumulate_twice` | `.grad` accumulates; `retain_graph` |
| `frozen_forward` | `no_grad` vs `eval()`; not leaking the context |
| `detach_vs_clone` | Storage sharing and graph retention |
| `clip_grad_norm` | Global norm; preserving direction; the number to log |
| `Square` | `forward`/`backward`, `save_for_backward`, `gradcheck` |
| `StraightThroughRound` | Non-differentiable ops; the estimator |
| `numeric_grad` | Central difference; why float64 |
| `gradient_penalty` | `create_graph=True`; second-order objectives |

## Check your understanding

1. **Two consecutive `loss.backward()` calls without zeroing gradients
   in between — what happens to `.grad`?**
   <details><summary>Answer</summary>Gradients accumulate — the second
   call's gradients are added to whatever's already in `.grad` from the
   first, rather than replacing it. This is why training loops call
   `optimizer.zero_grad()` every step.</details>

2. **What's the actual difference between `no_grad()` and calling
   `model.eval()`?**
   <details><summary>Answer</summary><code>eval()</code> changes layer
   *behavior* (dropout off, BatchNorm uses running stats) but still
   builds the autograd graph. <code>no_grad()</code> stops graph
   construction entirely — they're independent and usually both
   wanted together at inference time.</details>

3. **`detach()` and `clone()` both give you a new tensor. What's the
   difference in what each one keeps and drops?**
   <details><summary>Answer</summary><code>detach()</code> shares the
   same underlying storage but drops the autograd history.
   <code>clone()</code> copies the storage but keeps the autograd
   history (gradients flow back through a clone).</details>

4. **Why does gradient clipping use the *global* norm across all
   parameters, rather than clipping each parameter's gradient
   independently?**
   <details><summary>Answer</summary>Clipping each parameter separately
   would distort the relative direction of the overall gradient step —
   using one global norm and scaling every parameter by the same factor
   preserves the gradient's direction, only shrinking its
   magnitude.</details>

5. **A custom `autograd.Function` needs both `forward` and `backward`.
   What is `save_for_backward` for, and why not just close over the
   input in a closure?**
   <details><summary>Answer</summary>It tells autograd to track the
   saved tensors properly (respecting in-place operation checks and
   memory management) rather than silently holding a reference that
   could go stale or leak — `Function.forward`/`backward` are static
   methods with no closure to capture into anyway.</details>

6. **Why does computing a gradient penalty require `create_graph=True`
   in the first `backward`/`grad` call?**
   <details><summary>Answer</summary>The gradient penalty differentiates
   the gradient itself (a second-order derivative) — that only works if
   the graph used to compute the first gradient is itself
   differentiable, which is exactly what `create_graph=True`
   preserves.</details>

## Interview questions this module answers

- *What does `loss.backward()` actually do?*
- *Why does my training loop need `zero_grad()`?*
- *What is the difference between `no_grad()` and `eval()`?*
- *`detach()` or `clone()` — which do I want here?*
- *Why does gradient clipping use a global norm?*
- *Write a custom autograd function and prove it is correct.*
- *How do you get a gradient through a non-differentiable operation?*
- *Why does my memory usage grow every epoch?*
