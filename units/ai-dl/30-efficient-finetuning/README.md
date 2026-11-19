# 30 — Efficient fine-tuning and inference

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/ai-dl/30-efficient-finetuning/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/ai-dl/efficient-finetuning?utm_source=github&utm_medium=labs)

*~55 minutes · 20 tests*

Full fine-tuning and full-precision, full-memory inference are the
expensive defaults. This module is the mechanics behind the usual answers
to "we can't afford that": LoRA instead of updating every weight,
quantization instead of storing every weight in float32, gradient
checkpointing instead of holding every activation in memory, and dynamic
batching instead of a fixed-size request queue.

Work in `exercises.py`. Check your work with `python -m labkit check 30`, or open the notebook (`lab.ipynb`) in Colab.

Each section below ends with **Test it** — the exact command for just that
piece, run from `units/ai-dl/30-efficient-finetuning/`.

## What you'll build

| Name | You'll implement | Shows up in real code as |
|---|---|---|
| `LoRALinear` | `forward`, `merge` | Fine-tuning a large model on a fraction of its parameters |
| `quantize_int8`, `dequantize_int8` | both | Shrinking a model's memory footprint for deployment |
| `Checkpoint`, `checkpoint` | `forward`, `backward` | Training a model that would otherwise OOM on activations |
| `dynamic_batches` | `dynamic_batches` | The grouping logic behind a model-serving request queue |

---

## 1. LoRA: freeze the model, train a low-rank delta

```python
self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

def forward(self, x):
    base_out = self.base(x)
    lora_out = (x @ self.lora_A.T) @ self.lora_B.T
    return base_out + self.scaling * lora_out
```

Full fine-tuning trains every entry of a `(out_features, in_features)`
weight matrix. LoRA freezes that matrix completely and instead trains two
small matrices, `A` (`rank x in_features`) and `B` (`out_features x
rank`), whose product approximates the weight *delta* a full fine-tune
would have learned. For `rank` much smaller than `in_features` and
`out_features`, that is a large cut in both trainable parameters and
optimizer state — Adam alone keeps two extra tensors per trainable
parameter, so this savings compounds.

**`lora_B` starts at zero, `lora_A` does not.** At the start of training
the LoRA term is exactly zero regardless of what's in `A`, so the model
starts out behaving identically to the frozen base — training begins
from the pretrained behaviour, not from a random perturbation of it.

```python
>>> lora = LoRALinear(nn.Linear(8, 4), rank=2, alpha=4.0)
>>> torch.allclose(lora(x), lora.base(x))   # true before any training
True
```

**In `exercises.py`:** `LoRALinear.forward`.

**Test it:** `python3 -m unittest test_exercises.TestLoRALinear -v`

## 2. Merging: zero-overhead deployment

```python
delta = self.scaling * (self.lora_B @ self.lora_A)
merged.weight.copy_(self.base.weight + delta)
```

Because the LoRA update is *linear* in `x`, it can be folded directly
into the base weight — `B @ A` is the same shape as the base weight
matrix, so `base_weight + scaling * B @ A` is a single matrix that
produces the identical output, with none of the extra matmuls or extra
parameters at inference time. This is the standard "merge the adapter"
step before deployment: train with LoRA (cheap), merge once, then serve
a plain `nn.Linear` indistinguishable in cost from a fully fine-tuned
model. A non-linear adapter could not be merged this way — merging is a
property specific to LoRA's linear form.

```python
>>> merged = lora.merge()
>>> torch.allclose(merged(x), lora(x), atol=1e-5)
True
```

**In `exercises.py`:** `LoRALinear.merge`.

**Test it:** `python3 -m unittest test_exercises.TestLoRALinear.test_merge_matches_unmerged_forward -v`

## 3. int8 quantization: fewer bits, bounded error

```python
scale = tensor.abs().max() / 127
q = torch.clamp(torch.round(tensor / scale), -127, 127).to(torch.int8)
```

**Symmetric** quantization: zero always maps to zero, so only a `scale`
is needed, not a separate zero-point — this implementation covers the
symmetric case, which is what weight quantization typically uses
(activation quantization more often needs the asymmetric, zero-point
variant, since activations after a ReLU are not centred on zero). `scale`
is chosen so the tensor's largest-magnitude value maps to `±127` — int8
holds `-128..127`, and using `127` on both sides keeps the mapping
symmetric. Every other value rounds to the nearest of 255 representable
levels, which is exactly where the reconstruction error comes from: a
value between two levels cannot be represented exactly, and the maximum
possible rounding error is bounded by half the `scale` — or one full
`scale` in the worst case for the round in this implementation.

```python
>>> q, scale = quantize_int8(torch.tensor([-10.0, 3.0, 10.0]))
>>> q
tensor([-127,   38,  127], dtype=torch.int8)
>>> dequantize_int8(q, scale)
tensor([-10.0000,   2.9921,  10.0000])
```

**In `exercises.py`:** `quantize_int8`, `dequantize_int8`.

**Test it:** `python3 -m unittest test_exercises.TestInt8Quantization -v`

## 4. Gradient checkpointing: trade compute for memory

```python
class Checkpoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fn, x):
        ctx.fn = fn
        ctx.save_for_backward(x)
        with torch.no_grad():
            return fn(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            y = ctx.fn(x)
        grad_x, = torch.autograd.grad(y, x, grad_output)
        return None, grad_x
```

Normal autograd keeps every intermediate activation inside `fn` alive
from forward to backward, because the backward pass needs them to
compute gradients. `Checkpoint.forward` runs `fn` under `no_grad` and
saves only the *input* — none of `fn`'s internal activations are kept.
`backward` pays for that by **recomputing `fn(x)` from scratch**, this
time with gradient tracking on, purely to regenerate those activations
long enough to differentiate through them.

The trade is explicit: `fn` runs **twice** per training step instead of
once — more compute — in exchange for not holding its internal
activations in memory between forward and backward — less memory. That
is the entire content of `torch.utils.checkpoint`, and it is why
checkpointing a model typically slows down each step by something like
20-30% while letting you fit a meaningfully larger batch size or model
in the same GPU memory.

```python
>>> calls = 0
>>> def fn(x):
...     global calls; calls += 1
...     return (x * 2).sin().sum()
>>> y = checkpoint(fn, x)   # calls == 1: only the forward pass so far
>>> y.backward()            # calls == 2: backward recomputed fn once more
```

**In `exercises.py`:** `Checkpoint.forward`, `Checkpoint.backward`.

**Test it:** `python3 -m unittest test_exercises.TestCheckpoint -v`

## 5. Dynamic batching: size cap, or wait cap, whichever first

```python
def dynamic_batches(arrival_times, max_batch_size, max_wait):
    batches, current, batch_start = [], [], None
    for index, arrival in enumerate(arrival_times):
        if not current:
            batch_start = arrival
        current.append(index)
        if len(current) >= max_batch_size or (arrival - batch_start) >= max_wait:
            batches.append(current)
            current, batch_start = [], None
    if current:
        batches.append(current)
    return batches
```

A **static** batch always waits for exactly `N` requests before running
— fine under heavy, steady traffic, but under light traffic the `N`th
request might not show up for seconds, and everyone already queued pays
that latency for nothing. **Dynamic** batching adds a second exit
condition: flush once `max_wait` time has passed since the batch's first
request, even if it never fills. That bounds worst-case added latency at
both high and low traffic, at the cost of sometimes running a
smaller-than-optimal batch through the model.

One simplification worth naming explicitly, because it's a real gap
between this exercise and a production system: both conditions here are
only checked **when a new request arrives** — this function has no
clock ticking on its own between arrivals. A batch sitting one request
short of `max_batch_size`, with no further arrivals, will not flush
until either another request finally shows up (however late) or the
input ends. A real batcher needs an actual background timer independent
of arrivals to guarantee the latency bound on a lone pending request;
this models the grouping decision the timer would trigger, not the
timer itself.

```python
>>> dynamic_batches([0.0, 0.05, 0.1, 0.15], max_batch_size=2, max_wait=100.0)
[[0, 1], [2, 3]]
```

**In `exercises.py`:** `dynamic_batches`.

**Test it:** `python3 -m unittest test_exercises.TestDynamicBatches -v`

---

## The exercises

| Function / class | What it teaches |
|---|---|
| `LoRALinear` | Low-rank fine-tuning; zero-initialised `B` preserves pretrained behaviour at step 0 |
| `LoRALinear.merge` | Folding a linear adapter into the base weight for free inference |
| `quantize_int8`, `dequantize_int8` | Symmetric per-tensor quantization and its bounded rounding error |
| `Checkpoint`, `checkpoint` | Recompute-in-backward: the mechanism under `torch.utils.checkpoint` |
| `dynamic_batches` | Static vs. dynamic batching; a size cap and a wait cap together |

Every function in this table has a stub in `exercises.py` and test
coverage in `test_exercises.py` — a clean 1:1 mapping.

## Check your understanding

1. **Why does `lora_B` start at zero while `lora_A` starts at a small
   random value, rather than both starting at zero or both random?**
   <details><summary>Answer</summary>The LoRA output is
   <code>scaling * (x @ A.T) @ B.T</code> — if <code>B</code> is zero,
   that whole term is zero regardless of what's in <code>A</code>, so
   the model starts training from exactly the pretrained model's
   behaviour. If both were zero, the gradient with respect to
   <code>A</code> would also be zero at step one (since it flows through
   <code>B</code>), so nothing could ever start learning.</details>

2. **Why can a LoRA adapter be merged into the base weight for free, but
   a non-linear adapter (say, one with an activation function inside
   it) could not be?**
   <details><summary>Answer</summary>LoRA's update,
   <code>scaling * B @ A</code>, is a plain matrix the same shape as the
   base weight — adding two matrices together is still just one matrix,
   so the merged result behaves identically to running base and adapter
   separately and summing. A non-linear adapter's contribution can't be
   collapsed into the base weight matrix this way, because matrix
   addition doesn't commute with a non-linear function in between.</details>

3. **What does gradient checkpointing actually trade, and what's the
   concrete cost?**
   <details><summary>Answer</summary>It trades activation memory for
   recomputation time: the wrapped function's internal activations are
   never held between forward and backward, only the input is saved, and
   backward recomputes the forward pass to regenerate them. The concrete
   cost is that the wrapped function runs twice instead of once per
   training step, which usually costs something like 20-30% more time
   per step in exchange for a meaningfully smaller memory footprint.</details>

4. **In `quantize_int8`, why is the scale computed from the tensor's
   maximum absolute value rather than some fixed constant?**
   <details><summary>Answer</summary>Using the tensor's own max absolute
   value makes the mapping use the FULL available int8 range
   (<code>-127</code> to <code>127</code>) regardless of how that
   particular tensor happens to be scaled — a fixed scale would either
   clip a tensor with larger values or waste most of int8's resolution
   on a tensor with smaller ones.</details>

5. **Why does `dynamic_batches` fail to guarantee a strict latency bound
   for a single pending request, even though it has a `max_wait`
   parameter?**
   <details><summary>Answer</summary>The wait condition is only checked
   when a new request arrives — there's no independent clock driving a
   flush. A lone pending request with no further arrivals sits
   unflushed indefinitely; only another arrival (of any lateness) or the
   end of input triggers the check that would flush it. A real system
   needs a background timer that fires on its own, not tied to
   arrivals.</details>

## Interview questions this module answers

- *How does LoRA reduce the cost of fine-tuning, and what's the trade-off?*
- *Why can you merge a LoRA adapter into the base model but not every
  kind of adapter?*
- *What does int8 quantization actually do to a weight tensor, and where
  does the error come from?*
- *Your model doesn't fit in GPU memory during training — what are your
  options, and what does each cost?*
- *Design the batching layer for a model-serving endpoint — static or
  dynamic, and why?*
