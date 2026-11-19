# 29 — Optimizers and learning-rate schedules

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/ai-dl/29-optimizers/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/ai-dl/optimizers?utm_source=github&utm_medium=labs)

*~50 minutes · 20 tests*

`loss.backward()` gives you gradients. What the optimizer does with them —
not just "subtract the gradient" — is the difference between a run that
converges in an hour and one that diverges on step one, or converges to a
worse optimum, or trains fine in float32 and quietly breaks in bf16. This
module rebuilds SGD-with-momentum, Adam, and AdamW from the update rule up,
and the two schedule/clipping mechanics that sit around them in every real
training loop.

Every exercise here is checked against PyTorch's own optimizer classes,
step for step — so the numbers you get are pinned to the real
implementation, not to one author's arithmetic.

Work in `exercises.py`. Check your work with `python -m labkit check 29`, or open the notebook (`lab.ipynb`) in Colab.

Each section below ends with **Test it** — the exact command for just that
piece, run from `units/ai-dl/29-optimizers/`.

## What you'll build

| Function | You'll implement | Shows up in real code as |
|---|---|---|
| `sgd_momentum_step` | the momentum update | The optimizer everyone learns first, and still the right choice for plenty of vision training |
| `adam_step` | bias-corrected first/second moment update | The default optimizer for almost everything else |
| `adamw_step` | Adam plus DECOUPLED weight decay | What `torch.optim.AdamW` actually does differently from `Adam(weight_decay=...)` |
| `cosine_warmup_lr` | linear warmup, cosine decay | The LR schedule under nearly every published training recipe |
| `clip_grad_norm_` | global-norm clipping, in place | The line that stops one bad batch from blowing up a whole run |

---

## 1. Momentum: a running average of gradients, not just the last one

```python
velocity = momentum * velocity + grad
param = param - lr * velocity
```

Plain SGD moves exactly as far as the current gradient says, every step,
independently of what the last ten steps said. Momentum keeps `velocity`
as a decaying running average of past gradients, so a direction that has
been consistently downhill keeps building speed, while noise that
disagrees step to step gets damped rather than followed. This is what
carries a run through flat regions where the instantaneous gradient is
tiny, and reduces the oscillation you get crossing a narrow ravine in the
loss surface.

```python
>>> velocity = torch.zeros(1)
>>> param = torch.tensor([4.0])
>>> param, velocity = sgd_momentum_step(param, torch.tensor([2.0]), velocity, lr=0.5, momentum=0.9)
>>> param
tensor([3.])
```

**In `exercises.py`:** `sgd_momentum_step`.

**Test it:** `python3 -m unittest test_exercises.TestSgdMomentumStep -v`

## 2. Adam: per-parameter step sizes, and why bias correction matters

```python
m = beta1 * m + (1 - beta1) * grad          # mean estimate
v = beta2 * v + (1 - beta2) * grad * grad   # variance estimate
m_hat = m / (1 - beta1 ** t)
v_hat = v / (1 - beta2 ** t)
param = param - lr * m_hat / (v_hat.sqrt() + eps)
```

`m` and `v` are exponential moving averages of the gradient and its
square — a running estimate of the gradient's mean and variance for
*each parameter separately*. Dividing by `sqrt(v)` is what makes Adam's
effective step size roughly comparable across parameters no matter how
differently scaled their raw gradients are, which is the property that
makes it forgiving of bad initial learning-rate guesses compared to
plain SGD.

**Bias correction is not optional.** `m` and `v` both start at zero, so
early on the moving averages are biased toward zero — dividing by
`1 - beta**t` corrects for that. Skip it, and the first several steps are
far too small; `t` is small there, so `beta1**t` and `beta2**t` are still
close to 1, and `1 - beta**t` is close to 0.

```python
>>> m0 = v0 = torch.zeros(1)
>>> new_param, new_m, new_v = adam_step(torch.tensor([1.0]), torch.tensor([1.0]), m0, v0, t=1, lr=0.1)
>>> new_param        # bias-corrected: a nearly full lr-sized step even at t=1
tensor([0.9000])
```

**In `exercises.py`:** `adam_step`.

**Test it:** `python3 -m unittest test_exercises.TestAdamStep -v`

## 3. AdamW: weight decay taken OUT of the adaptive step

```python
param = param - lr * (m_hat / (v_hat.sqrt() + eps) + weight_decay * param)
```

The naive way to add L2 regularisation to Adam folds it into the
gradient before `m` and `v` ever see it: `grad = grad + weight_decay *
param`. That's what `torch.optim.Adam(..., weight_decay=...)` actually
does, and for an adaptive optimizer it's subtly wrong — the decay term
then gets divided by `sqrt(v)` right alongside the real gradient, so
parameters with large gradient variance get decayed *less* than
parameters with small variance. Weight decay was never supposed to care
about gradient variance.

AdamW (Loshchilov & Hutter, 2019) applies the decay directly to the
parameter, entirely outside the `m`/`v` machinery — **decoupled** from
the adaptive step, hence the name. Every parameter is decayed by the
same proportion, full stop.

```python
>>> param = torch.tensor([2.0])
>>> m0 = v0 = torch.zeros(1)
>>> new_param, _, _ = adamw_step(param, torch.tensor([0.0]), m0, v0, t=1, lr=0.1, weight_decay=0.5)
>>> new_param   # zero gradient - only the decay term moves the parameter
tensor([1.9000])
```

With `weight_decay=0`, `adamw_step` and `adam_step` are identical — the
tests pin that down directly.

**In `exercises.py`:** `adamw_step`.

**Test it:** `python3 -m unittest test_exercises.TestAdamwStep -v`

## 4. Warmup, then cosine decay

```python
if step < warmup_steps:
    return base_lr * step / warmup_steps
progress = (step - warmup_steps) / (total_steps - warmup_steps)
cosine = 0.5 * (1 + math.cos(math.pi * progress))
return min_lr + (base_lr - min_lr) * cosine
```

**Warmup** exists because Adam's earliest steps are the least
trustworthy ones — `m` and `v` are still close to their zero
initialisation, so a full-strength step right away can push a randomly
initialised model somewhere the optimizer never cheaply recovers from.
Ramping the LR up linearly over the first `warmup_steps` avoids that.

**Cosine decay** after warmup brings the LR down to `min_lr` smoothly
instead of stepping down abruptly — empirically it tends to outperform a
hard step schedule, and it introduces no extra hyperparameter beyond
"how long is this run." At the exact midpoint between the end of warmup
and the end of training, the cosine term is 0.5 — the LR is exactly
halfway between `base_lr` and `min_lr`.

```python
>>> [round(cosine_warmup_lr(s, total_steps=100, warmup_steps=10, base_lr=1.0), 2) for s in [0, 5, 10, 55, 100]]
[0.0, 0.5, 1.0, 0.5, 0.0]
```

**In `exercises.py`:** `cosine_warmup_lr`.

**Test it:** `python3 -m unittest test_exercises.TestCosineWarmupLr -v`

## 5. Global-norm gradient clipping

```python
total_norm = torch.sqrt(sum((g ** 2).sum() for g in grads))
clip_coef = max_norm / (total_norm + 1e-6)
if clip_coef < 1:
    for g in grads:
        g.mul_(clip_coef)
```

The **global** norm treats every gradient tensor in the model as one
concatenated vector, not as independent tensors clipped separately.
Scaling every tensor by the *same* factor when the combined norm exceeds
`max_norm` preserves the gradient's direction across the whole model;
clipping each parameter's gradient independently by its own norm would
distort that direction, changing what the optimizer effectively
optimizes for.

This is a mutating, in-place operation — matching
`torch.nn.utils.clip_grad_norm_`'s own signature — and it returns the
norm **before** clipping, which is what you log to see whether clipping
is actually triggering on a given batch (if it fires on nearly every
step, that is itself worth investigating, not just silently tolerating).

```python
>>> g = torch.tensor([3.0, 4.0])   # norm = 5
>>> total = clip_grad_norm_([g], max_norm=1.0)
>>> total
tensor(5.)
>>> g.norm()
tensor(1.0000)
```

**In `exercises.py`:** `clip_grad_norm_`.

**Test it:** `python3 -m unittest test_exercises.TestClipGradNorm -v`

---

## The exercises

| Function | What it teaches |
|---|---|
| `sgd_momentum_step` | Momentum as a running average of past gradients |
| `adam_step` | Per-parameter adaptive step size; why bias correction is not optional |
| `adamw_step` | Decoupled weight decay — the actual difference from `Adam(weight_decay=...)` |
| `cosine_warmup_lr` | Why warmup exists; smooth cosine decay to a floor |
| `clip_grad_norm_` | Global norm across all tensors, not per-tensor clipping |

Every function in this table has a stub in `exercises.py` and test
coverage in `test_exercises.py`, verified step-for-step against the
matching `torch.optim` / `torch.nn.utils` implementation.

## Check your understanding

1. **Why does Adam need bias correction in its early steps, but SGD with
   momentum (as implemented here) does not?**
   <details><summary>Answer</summary>Adam divides by <code>sqrt(v)</code>,
   and both <code>m</code> and <code>v</code> start at zero — without
   correction, that division makes the very first steps far too small
   until the moving averages have had time to accumulate real signal.
   Momentum's velocity also starts at zero, but it isn't used as a
   denominator, so an uncorrected early velocity just means the first
   step is closer to a plain gradient step rather than being distorted
   in scale.</details>

2. **What specifically is "decoupled" about AdamW's weight decay?**
   <details><summary>Answer</summary>The decay term
   (<code>weight_decay * param</code>) is subtracted from the parameter
   directly, entirely outside the <code>m_hat / (sqrt(v_hat) +
   eps)</code> adaptive step. In plain Adam with
   <code>weight_decay</code>, the decay is instead added into the
   gradient before <code>m</code> and <code>v</code> are computed, so it
   ends up divided by <code>sqrt(v)</code> along with the real gradient
   — meaning parameters with noisier gradients get decayed less, which
   isn't the intended effect of weight decay.</details>

3. **Why ramp the learning rate up at the start of training instead of
   starting at the target LR immediately?**
   <details><summary>Answer</summary>Early in training (and especially
   early in Adam's own moment estimates) the optimizer's sense of "which
   direction and how far" is least reliable — a large step taken then
   can push a randomly initialized model to a bad region it doesn't
   cheaply recover from. Warmup ramps the step size up only once the
   gradient estimates have had a chance to stabilize.</details>

4. **Why clip by the global norm across all parameters, rather than
   clipping each parameter's gradient independently to the same max
   norm?**
   <details><summary>Answer</summary>Clipping every tensor by the same
   scale factor (derived from the combined norm) preserves the relative
   direction of the gradient across the whole model. Clipping each
   tensor independently to its own max norm would scale different parts
   of the model differently, distorting the update's direction relative
   to what unclipped gradient descent would have done.</details>

5. **If you log `clip_grad_norm_`'s return value and see it firing (the
   norm exceeding `max_norm`) on almost every step, what does that
   suggest?**
   <details><summary>Answer</summary>That the learning rate is
   effectively too high for the loss landscape being trained on, or
   something upstream (a bad batch, an unstable loss term, a data bug)
   is producing consistently oversized gradients — clipping is masking a
   problem rather than occasionally guarding against a rare spike, which
   is what it's meant for.</details>

## Interview questions this module answers

- *Why does AdamW exist — what's actually different from Adam plus L2
  regularisation?*
- *What does bias correction fix in Adam's early steps?*
- *Design a learning-rate schedule for a training run — what are the
  pieces and why?*
- *Your loss spiked to NaN on step one — what's the first thing you'd
  check?*
- *What's the difference between clipping gradients globally vs.
  per-parameter?*
