# 24 — Modules and model surgery

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/ai-dl/24-modules/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/ai-dl/modules?utm_source=github&utm_medium=labs)

*~50 minutes · 52 tests*

`nn.Module` is a registry with a `forward` attached. It tracks three kinds
of state — **parameters**, **buffers**, and **child modules** — and almost
everything it gives you (`.to()`, `.parameters()`, `state_dict()`,
`.train()`) is a recursive walk over that registry.

Understanding the registry is what lets you do *model surgery*: freeze a
backbone, load a checkpoint that does not quite fit, tie two layers
together, or find out why layer 7 stopped producing signal. That is most of
the practical work in an AI role, and it is what code-review rounds probe.

Work in `exercises.py`. Check your work with `python -m labkit check 24`, or open the notebook (`lab.ipynb`) in Colab.

Each section below ends with **Test it** — the exact command for just that
piece, run from `units/ai-dl/24-modules/`. No need to finish the whole
file before checking your work.

## What you'll build

Eight classes and functions in `exercises.py`, in this order. Keep this
table open as you go — each section below tells you which row it's
building:

| Function | You'll implement | Shows up in real code as |
|---|---|---|
| `MLP` | `forward` | The single most common `nn.Module` bug in production: layers held in a plain Python list instead of `nn.ModuleList` train forever from their random initialisation, with no error anywhere |
| `RunningMean` | `forward` | BatchNorm's `running_mean`/`running_var` — state a module owns that gradient descent must never touch, the shape behind every buffer you'll register |
| `freeze` | `freeze` | Transfer learning: fine-tuning only the head of a pretrained backbone, and the two things people forget to also do when they freeze it |
| `init_weights` | `init_weights` | Why a deep network trained from scratch produces exploding or vanishing activations — and NaN loss — within the first few steps if you skip proper scaling |
| `tie_weights` | `tie_weights` | Halving a language model's parameter count by sharing the input embedding and output projection — standard practice, and a genuinely hard bug to spot if done wrong |
| `activation_stats` | `activation_stats` | Diagnosing a model you did not write — is the signal vanishing, exploding, or dead — without editing its `forward()` |
| `load_partial` | `load_partial` | Fine-tuning a pretrained classifier onto a different label set: the checkpoint has a 1000-way head, yours has 10 |
| `module_summary` | `module_summary` | The first command you run against an unfamiliar checkpoint to find out where the parameters actually live |

---

## 1. The registry, and the bug everyone hits once

```python
self.layers = [nn.Linear(4, 4) for _ in range(3)]        # INVISIBLE
self.layers = nn.ModuleList(nn.Linear(4, 4) for _ in range(3))   # registered
```

Assigning a `Module` to an attribute registers it. Putting modules in a
**plain Python list** does not. The consequence: `.parameters()` never
yields them, the optimiser never sees them, `.to(device)` never moves them,
and they never appear in `state_dict()`.

```python
>>> class Bad(nn.Module):
...     def __init__(self):
...         super().__init__()
...         self.layers = [nn.Linear(4, 4) for _ in range(3)]   # plain list
...
>>> list(Bad().parameters())
[]
>>> class Good(nn.Module):
...     def __init__(self):
...         super().__init__()
...         self.layers = nn.ModuleList(nn.Linear(4, 4) for _ in range(3))
...
>>> len(list(Good().parameters()))
6
```

The failure mode is the worst kind — the model runs, the loss goes down
(the registered layers still learn), and `Bad`'s layers simply stay at
their initialisation forever. Nothing errors.

The containers: `nn.ModuleList`, `nn.ModuleDict`, `nn.ParameterList`,
`nn.ParameterDict`, and `nn.Sequential` when the layers just chain.

**In `exercises.py`:** `MLP.forward` is the stub. The registration itself —
`self.layers = nn.ModuleList(...)` — is already written in `__init__`; this
section is about knowing why that line has to look exactly like that, not
about writing it yourself.

**Test it:** `python3 -m unittest test_exercises.TestMLP -v`

## 2. Parameters vs buffers

| | Parameter | Buffer |
|---|---|---|
| Created with | `nn.Parameter(...)` | `register_buffer(...)` |
| In `.parameters()` | yes | no |
| Gets gradients | yes | no |
| In `state_dict()` | yes | yes (unless `persistent=False`) |
| Moves with `.to()` | yes | yes |

A buffer is **state the module owns that gradient descent must not touch**:
BatchNorm's `running_mean` and `running_var`, a causal mask, RoPE tables, a
step counter.

```python
>>> class Counter(nn.Module):
...     def __init__(self):
...         super().__init__()
...         self.register_buffer("step", torch.zeros((), dtype=torch.long))
...
>>> c = Counter()
>>> list(c.parameters())
[]
>>> dict(c.named_buffers())
{'step': tensor(0)}
>>> c.state_dict()
OrderedDict({'step': tensor(0)})
```

`step` shows up in `state_dict()` and `named_buffers()` but never in
`parameters()` — the optimiser will never touch it, exactly as intended.

Two things follow:

- A buffer is saved in the checkpoint, so loading into a model built
  *without* it fails loudly with an unexpected-key error. That is usually
  a version mismatch worth investigating, not something to suppress with
  `strict=False`.
- Never store such state as a plain tensor attribute. It will not move to
  the GPU with the model, and you will get a device-mismatch error the
  first time you run on hardware.

**In `exercises.py`:** `RunningMean.forward`. The buffers themselves —
`register_buffer("mean", ...)` and `register_buffer("count", ...)` — are
already registered in `__init__`; `forward` is what updates `mean` by an
exponential moving average while `self.training` is `True`, and always
subtracts it from the input.

**Test it:** `python3 -m unittest test_exercises.TestRunningMean -v`

## 3. Freezing has two halves

```python
>>> model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
>>> for p in model.parameters():
...     p.requires_grad_(False)
...
>>> any(p.requires_grad for p in model.parameters())
False
```

That is half of it. The other two:

1. **The optimiser.** If it was built over every parameter it will still
   apply weight decay and momentum to the frozen ones — they drift even
   with no gradient. Build the optimiser *after* freezing, or filter:
   `filter(lambda p: p.requires_grad, model.parameters())`.
2. **BatchNorm.** Freezing does not stop running statistics updating. A
   frozen backbone left in `train()` mode keeps drifting its normalisation
   as it sees new data. `module.eval()` is what stops that, and it is the
   half people forget:

```python
>>> bn = nn.BatchNorm1d(3)
>>> for p in bn.parameters():
...     p.requires_grad_(False)
...
>>> bn.train()
BatchNorm1d(3, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
>>> bn.running_mean
tensor([0., 0., 0.])
>>> _ = bn(torch.randn(8, 3) + 5)
>>> bn.running_mean   # drifted, despite every parameter being frozen
tensor([0.5185, 0.5116, 0.5372])
```

**In `exercises.py`:** `freeze(module, prefix=None)` — set
`requires_grad_(False)` on every parameter (or only those whose dotted name
starts with `prefix`), and return how many scalars were frozen. The
optimiser-ordering and `eval()` points above are not exercised by
`freeze` itself — they are what to say out loud in an interview once
`freeze` is called — but the test suite does check `freeze`'s own two
behaviours: the prefix filter, and that a frozen parameter gets no
gradient.

**Test it:** `python3 -m unittest test_exercises.TestFreeze -v`

## 4. Initialisation

With the wrong scale, activation variance either explodes or collapses
layer by layer and a deep network does not train at all.

- **Xavier/Glorot** keeps the variance constant for a symmetric activation
  (tanh, linear): bounds `±sqrt(6 / (fan_in + fan_out))`.
- **Kaiming/He** is the right choice after a ReLU, which throws away half
  the signal, so the variance needs a factor of 2 back:
  `kaiming_normal_(w, nonlinearity="relu")`.
- **Biases to zero.** Almost always.

`module.apply(fn)` walks every submodule, which is the idiomatic way to do
this across a whole model:

```python
>>> _ = torch.manual_seed(0)
>>> model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
>>> def initialise(layer):
...     if isinstance(layer, nn.Linear):
...         nn.init.xavier_uniform_(layer.weight)
...         nn.init.zeros_(layer.bias)
...
>>> _ = model.apply(initialise)
>>> model[0].bias
Parameter containing:
tensor([0., 0., 0., 0.], requires_grad=True)
>>> model[0].weight.abs().max().item() <= (6 / (4 + 4)) ** 0.5
True
```

Modern architectures also lean on **normalisation layers** (LayerNorm,
RMSNorm) to keep the scale in range, and on **residual connections** so the
gradient has an identity path. Init still matters, but it is one of three
mechanisms rather than the only one — worth saying if you are asked.

**In `exercises.py`:** `init_weights(module, gain=1.0)` — apply
`xavier_uniform_` (scaled by `gain`) to every `nn.Linear` weight via
`module.apply`, zero every bias that exists, and leave non-`Linear` layers
(BatchNorm, ReLU, ...) untouched.

**Test it:** `python3 -m unittest test_exercises.TestInitWeights -v`

## 5. Weight tying

```python
output_layer.weight = embedding.weight        # shares
output_layer.weight.copy_(embedding.weight)   # equal once, then diverges
```

Assignment shares the same `Parameter` object; `copy_` makes the values
equal at that instant and they immediately drift apart. The two look
identical right after initialisation, which makes this a genuinely hard bug
to spot.

```python
>>> embedding = nn.Embedding(5, 3)
>>> head = nn.Linear(3, 5, bias=False)
>>> head.weight is embedding.weight
False
>>> head.weight = embedding.weight
>>> head.weight is embedding.weight
True
>>> with torch.no_grad():
...     embedding.weight.add_(1.0)
...
>>> torch.equal(head.weight, embedding.weight)   # moved together
True
```

Standard in language models: the input embedding `(V, D)` and the output
projection `(V, D)` are learning the same thing, so tying them halves that
parameter count — for a 50k vocabulary at 768 dims that is 38M parameters —
and usually *improves* perplexity.

Because they are one tensor, gradients from both uses accumulate into it,
which is exactly what you want.

**In `exercises.py`:** `tie_weights(embedding, output_layer)` — assign (not
copy) `embedding.weight` onto `output_layer.weight`, raising `ValueError`
first if the shapes don't match, and return `output_layer`.

**Test it:** `python3 -m unittest test_exercises.TestTieWeights -v`

## 6. Hooks: observing a model you did not write

```python
>>> _ = torch.manual_seed(0)
>>> layer = nn.Linear(4, 4)
>>> outputs = {}
>>> handle = layer.register_forward_hook(
...     lambda mod, inputs, output: outputs.setdefault("std", output.std().item()))
>>> _ = layer(torch.randn(8, 4))
>>> outputs["std"]
0.6875953674316406
>>> handle.remove()
>>> len(layer._forward_hooks)
0
```

Hooks let you inspect intermediate activations without editing `forward`.
Diagnosing a dead network starts here:

- **Std shrinking layer by layer** → the signal is vanishing; check init
  and normalisation.
- **Std growing without bound** → the run is about to produce NaN; check
  the learning rate and clipping.
- **Std exactly zero** → dead ReLUs, or a layer that never got wired in.

`register_full_backward_hook` does the same for gradients.

**Every handle must be removed.** A leaked hook holds references (so memory
never frees) and runs on every subsequent forward pass. Put the removal in
a `finally` — the reference implementation does, and a test checks it holds
even when the forward raises.

**In `exercises.py`:** `activation_stats(model, x)` — register a forward
hook on every leaf submodule (one with no children of its own), run `x`
through the model once under `torch.no_grad()`, collect `{name: output std}`
from each hook, and remove every handle in a `finally` so it happens even
if the forward call raises.

**Test it:** `python3 -m unittest test_exercises.TestActivationStats -v`

## 7. Loading a checkpoint that does not quite fit

The everyday transfer-learning problem: weights trained with 1000 output
classes going into a model with 10.

```python
model.load_state_dict(sd)                 # raises on any mismatch
model.load_state_dict(sd, strict=False)   # tolerates missing/unexpected KEYS
                                          # but still raises on a SHAPE mismatch
```

So the filtering has to be explicit: keep entries whose **name and shape**
both match, load those, and report what was skipped. Reporting matters —
silently skipping half a checkpoint gives you a randomly-initialised model
that trains from scratch while you believe you are fine-tuning.

```python
>>> model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 10))
>>> source = {"0.weight": torch.ones(8, 4), "0.bias": torch.ones(8),
...           "1.weight": torch.ones(1000, 8), "1.bias": torch.ones(1000)}
>>> skipped = load_partial(model, source)
>>> skipped   # the mismatched 1000-class head, not the matching backbone
['1.bias', '1.weight']
>>> torch.equal(model[0].weight, torch.ones(8, 4))   # backbone loaded
True
```

`load_state_dict` also returns a `namedtuple` of `missing_keys` and
`unexpected_keys`. Read it.

**In `exercises.py`:** `load_partial(model, state_dict)` — keep only the
entries whose name is in `model.state_dict()` *and* whose shape matches,
load those with `model.load_state_dict(usable, strict=False)`, and return
the sorted list of names that were skipped.

**Test it:** `python3 -m unittest test_exercises.TestLoadPartial -v`

## 8. Reading a model

```python
model.named_children()    # one level deep
model.named_modules()     # recursive — summing over it double-counts
model.named_parameters()  # every parameter, with dotted names
```

```python
>>> nested = nn.Sequential(
...     nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4)), nn.Linear(4, 2))
>>> [name for name, _ in nested.named_children()]
['0', '1']
>>> [name for name, _ in nested.named_modules()]
['', '0', '0.0', '0.1', '1']
```

Two children (`'0'`, `'1'`), but five entries once you recurse — the root
module itself (named `''`), plus both grandchildren inside `'0'`. Summing a
per-module parameter count over `named_modules()` here would count the
inner `Linear`s once directly and again as part of their parent — that's
the double-count.

`named_children` plus a parameter count per child is the practical "where
do the parameters live" summary, and it is the first thing to run when
someone hands you an unfamiliar checkpoint:

```python
>>> model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
>>> module_summary(model)
[('0', 'Linear', 40), ('1', 'ReLU', 0), ('2', 'Linear', 18)]
```

**In `exercises.py`:** `module_summary(model)` — return
`[(name, class name, parameter count)]` for each entry in
`model.named_children()`, one level deep, not recursed.

**Test it:** `python3 -m unittest test_exercises.TestModuleSummary -v`

---

## The exercises

| Function | What it teaches |
|---|---|
| `MLP` | `ModuleList` vs a bare list; registration |
| `RunningMean` | Buffers vs parameters; train/eval behaviour |
| `freeze` | `requires_grad_`; what freezing does *not* do |
| `init_weights` | Xavier/Kaiming; `module.apply` |
| `tie_weights` | Sharing vs copying a `Parameter` |
| `activation_stats` | Forward hooks; always removing the handle |
| `load_partial` | Name-and-shape filtering; reporting what was skipped |
| `module_summary` | `named_children` vs `named_modules` |

## Check your understanding

1. **Why does storing submodules in a plain Python `list` break
   parameter registration, when `ModuleList` doesn't?**
   <details><summary>Answer</summary><code>nn.Module.__setattr__</code>
   specifically recognizes <code>Module</code>/<code>ModuleList</code>
   assignments and registers them — a plain list is invisible to that
   mechanism, so `.parameters()` and `.to(device)` silently skip
   whatever's inside it.</details>

2. **What's the actual difference between a buffer and a parameter?**
   <details><summary>Answer</summary>Both are tracked as part of a
   module's state (saved in `state_dict`, moved with `.to(device)`), but
   only parameters are updated by the optimizer via gradients. A buffer
   like a `BatchNorm` running mean is state that changes some other way.</details>

3. **`freeze` sets `requires_grad_(False)`. What does that *not* do,
   that people sometimes assume it does?**
   <details><summary>Answer</summary>It doesn't remove the parameters
   from an optimizer that was already constructed with them — a frozen
   parameter can still be "updated" by an optimizer that references it
   unless it's also excluded from the optimizer's parameter group.</details>

4. **Why must a forward hook's handle always be removed, even if the
   hook only needs to run once?**
   <details><summary>Answer</summary>A hook stays registered until
   explicitly removed via <code>handle.remove()</code> — leaving it
   attached means it keeps firing (and potentially holding references,
   causing a memory leak) on every subsequent forward pass, long after
   its job was done.</details>

5. **`load_partial` needs both name *and* shape filtering. Why isn't
   matching by name alone enough?**
   <details><summary>Answer</summary>Two checkpoints can have a
   parameter with the same name but a different shape (e.g. after
   changing a layer's output size) — loading it by name alone would
   raise a shape-mismatch error instead of skipping it cleanly.</details>

6. **What's the difference between `named_children()` and
   `named_modules()`?**
   <details><summary>Answer</summary><code>named_children()</code> only
   yields the immediate submodules, one level down.
   <code>named_modules()</code> recurses through the entire module tree,
   yielding every submodule at every depth.</details>

## Interview questions this module answers

- *I added layers and they never train. Why?*
- *What is the difference between a parameter and a buffer?*
- *How do you freeze a backbone — and what else has to change?*
- *Why does initialisation matter, and which scheme after a ReLU?*
- *What is weight tying and why do language models do it?*
- *How would you inspect activations in a model you did not write?*
- *This checkpoint has 1000 classes and my model has 10. Load it.*
