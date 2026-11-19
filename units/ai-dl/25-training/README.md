# 25 — Data loading and the training loop

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/ai-dl/25-training/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/ai-dl/training?utm_source=github&utm_medium=labs)

*~55 minutes · 56 tests*

The training loop is four lines and everyone knows them. What separates a
working run from a broken one is everything around them: whether the data
pipeline is actually random, whether eval mode is set, whether the
checkpoint can be resumed, and whether you checked the model *can* learn
before spending eight hours proving it cannot.

This module is the part of the job that never appears in a paper and always
appears in an interview, because it is what you spend most of your time on.

Work in `exercises.py`. Check your work with `python -m labkit check 25`, or open the notebook (`lab.ipynb`) in Colab.

Each section below ends with **Test it** — the exact command for just that
piece, run from `units/ai-dl/25-training/`. No need to finish the whole
file before checking your work.

## What you'll build

One class and eleven functions in `exercises.py` — twelve pieces in eleven
rows below, since `save_checkpoint`/`load_checkpoint` share a row — in this
order. Keep this table open as you go — each section below tells you which
row it's building:

| Function | You'll implement | Shows up in real code as |
|---|---|---|
| `ArrayDataset` | `__len__`, `__getitem__` | Any custom dataset wrapping in-memory tensors, a Parquet shard, or a set of file paths — the shape every map-style `Dataset` takes |
| `pad_collate` | `pad_collate` | Batching variable-length sequences — tokenised text, per-image detections, point clouds — anywhere `torch.stack` cannot apply |
| `seed_everything` | `seed_everything` | The seeding call at the top of every training script, the thing that makes a bug report reproducible and a paper's numbers regenerable |
| `worker_init_fn` | `worker_init_fn` | The fix for a famous class of published-code bug: `num_workers>1` silently quartering your augmentation diversity because every forked worker replays the same NumPy draws |
| `train_one_epoch` | `train_one_epoch` | The training loop itself — the four-line core that every framework (Lightning, HF `Trainer`, Accelerate) wraps in more configuration |
| `evaluate` | `evaluate` | The validation pass run after every epoch — the one place `eval()`/`no_grad()` both have to be right, or your metric is wrong or your memory blows up |
| `cosine_warmup_lr` | `cosine_warmup_lr` | The default schedule for training a transformer from scratch or fine-tuning one, in recipes from BERT to LLaMA |
| `save_checkpoint` / `load_checkpoint` | `save_checkpoint`, `load_checkpoint` | Resuming a multi-day run after a pre-emption, or handing a checkpoint to a teammate who has one GPU where you had eight |
| `overfit_one_batch` | `overfit_one_batch` | The thirty-second sanity check before spending a training budget on a model that might have a wiring bug |
| `class_weights` | `class_weights` | Training on a 99%-negative fraud/defect/anomaly dataset without the model degenerating to predicting the majority class |
| `build_loader` | `build_loader` | The `DataLoader` you actually construct in a training script — the point where seeding, shuffling and `drop_last` all have to agree |

---

## 1. `Dataset` returns ONE example

```python
def __getitem__(self, index):
    return self.features[index], self.labels[index]     # one, not a batch
```

The `DataLoader` does the batching. Returning a batch here gives you an
extra leading dimension that only shows up as a confusing shape error
inside the loss.

```python
>>> features = torch.arange(12.).reshape(4, 3)
>>> labels = torch.tensor([0, 1, 0, 1])
>>> ds = ArrayDataset(features, labels)
>>> len(ds)
4
>>> ds[2]
(tensor([6., 7., 8.]), tensor(0))
>>> item, label = ds[2]
>>> item.shape, label.shape        # one row, one scalar - no batch dimension
(torch.Size([3]), torch.Size([]))
```

Two styles:

- **Map-style** (`__len__` + `__getitem__`): random access, shuffling,
  sampling. What you want almost always.
- **Iterable-style** (`__iter__`): streams that have no length — a Kafka
  topic, a sharded remote dataset. You must handle worker sharding
  yourself or every worker yields the same data.

**In `exercises.py`:** `ArrayDataset.__len__` and `ArrayDataset.__getitem__`
are this section's targets.

**Test it:** `python3 -m unittest test_exercises.TestArrayDataset -v`

## 2. `collate_fn` — the moment your data is not rectangular

The default collate calls `torch.stack`, which requires identical shapes.
The moment your samples are sentences, detections, or point clouds, you
write this yourself:

```python
lengths = torch.tensor([len(s) for s in sequences])
padded = torch.full((len(sequences), lengths.max()), pad_value)
for row, sequence in enumerate(sequences):
    padded[row, :len(sequence)] = sequence
return padded, lengths
```

```python
>>> padded, lengths = pad_collate([[1, 2], [1, 2, 3], [1]])
>>> padded
tensor([[1, 2, 0],
        [1, 2, 3],
        [1, 0, 0]])
>>> lengths
tensor([2, 3, 1])
```

**Return the lengths.** Padding without lengths means the model cannot mask
the padded positions, and it silently learns from them. Everything
downstream — attention masks, `pack_padded_sequence`, pooled means — needs
that vector.

A related throughput trick worth naming: bucketing samples of similar
length into the same batch, so you pad less.

**In `exercises.py`:** `pad_collate` is this section's target — pad every
sequence in the batch to the longest one, and return `(padded, lengths)`.

**Test it:** `python3 -m unittest test_exercises.TestPadCollate -v`

## 3. The worker-seed bug

`DataLoader(num_workers=4)` forks, and `fork()` copies the parent's RNG
state. So every worker generates the **same** "random" augmentations.

With 4 workers you get each augmentation four times per epoch, your
effective augmentation diversity is quartered, and nothing errors — the
loss still goes down, just more slowly than it should. This was a real bug
in a lot of published code.

```python
def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
```

PyTorch seeds its **own** generator per worker correctly. NumPy and
`random` are the ones that need this — and NumPy is what most augmentation
libraries use.

```python
>>> torch.manual_seed(0); worker_init_fn(0)
>>> np.random.rand(3)              # worker 0's first draw
array([0.5488135 , 0.71518937, 0.60276338])
>>> torch.manual_seed(0); worker_init_fn(1)
>>> np.random.rand(3)              # worker 1: same torch seed, different draw
array([4.17022005e-01, 7.20324493e-01, 1.14374817e-04])
```

Without `worker_init_fn`, both workers would draw the identical sequence —
the same augmentation, replayed once per worker, every epoch.

**In `exercises.py`:** `worker_init_fn` is this section's target — derive a
per-worker seed from `torch.initial_seed()` and seed NumPy and `random`
with it.

**Test it:** `python3 -m unittest test_exercises.TestSeeding -k worker_init -v`

This isolates the one `TestSeeding` method about `worker_init_fn` from the
four others in that class, which belong to §4 below.

## 4. Reproducibility has layers

```python
random.seed(s); np.random.seed(s); torch.manual_seed(s)
torch.cuda.manual_seed_all(s)
```

```python
>>> seed_everything(123)
123
>>> torch.randn(3)
tensor([-0.1115,  0.1204, -0.3696])
>>> seed_everything(123)
123
>>> torch.randn(3)                 # same seed -> same draw
tensor([-0.1115,  0.1204, -0.3696])
```

Three generators; missing any one makes a run unreproducible. Then, beyond
seeding:

- **DataLoader shuffling** needs an explicitly seeded `generator=`.
- **Workers** need `worker_init_fn` (§3).
- **cuDNN** picks algorithms non-deterministically. `torch.use_
  deterministic_algorithms(True)` forces it, at a real speed cost.
- **Atomics on GPU** make some reductions non-deterministic regardless.

Bit-exact reproducibility is expensive. The honest interview answer is that
you seed everything, pin your versions, log the config, and accept that
exact GPU reproducibility costs performance you usually would rather spend
elsewhere.

**In `exercises.py`:** `seed_everything` is this section's target — seed
`random`, `numpy` and `torch` (CPU and CUDA) from one integer, and return
it.

**Test it:** `python3 -m unittest test_exercises.TestSeeding -k same_seed -k different_seeds_differ -v`

This runs the four `TestSeeding` methods about `seed_everything` and
excludes the `worker_init_fn` one covered in §3.

## 5. The loop

```python
optimiser.zero_grad()
loss = loss_fn(model(x), y)
loss.backward()
clip_grad_norm_(model.parameters(), 1.0)     # after backward, before step
optimiser.step()
```

The order is not negotiable. `zero_grad` first because `.grad` accumulates
(module 02). Clipping between `backward` and `step`, because that is the
only window where gradients exist and have not yet been applied.

**Averaging.** `loss.item() * len(batch)`, summed, then divided by the
dataset size. Averaging the per-batch means over-weights a short final
batch — which is exactly where an unexplained train/val discrepancy comes
from.

**`loss.item()`, not `loss`.** Accumulating the tensor keeps the entire
epoch's graph alive and you run out of memory somewhere in epoch 2.

```python
>>> torch.manual_seed(0)
>>> features, labels = torch.randn(16, 4), torch.randint(0, 3, (16,))
>>> loader = torch.utils.data.DataLoader(ArrayDataset(features, labels), batch_size=4)
>>> model = nn.Linear(4, 3)
>>> optimiser = torch.optim.SGD(model.parameters(), lr=0.1)
>>> loss_fn = nn.CrossEntropyLoss()
>>> for epoch in range(3):
...     print(train_one_epoch(model, loader, optimiser, loss_fn))
...
1.4143387079238892
1.2438783645629883
1.1218021214008331
```

**In `exercises.py`:** `train_one_epoch` is this section's target — the
four lines above, plus per-sample-weighted averaging of the loss.

**Test it:** `python3 -m unittest test_exercises.TestTrainOneEpoch -v`

## 6. Evaluation needs both switches

```python
@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    ...
```

`eval()` for dropout and BatchNorm, `no_grad()` for memory. Module 02 makes
the case; here it is in practice:

```python
>>> model = nn.Sequential(nn.Dropout(p=1.0))    # drops every unit
>>> loader = torch.utils.data.DataLoader(
...     ArrayDataset(torch.ones(4, 2), torch.ones(4, 2)), batch_size=4)
>>> model.train()                                # left in TRAIN mode on purpose
Sequential(
  (0): Dropout(p=1.0, inplace=False)
)
>>> evaluate(model, loader, nn.MSELoss())        # evaluate() must switch to eval() itself
0.0
>>> model.training                               # ...and restore train mode after
True
```

If `evaluate` forgot to call `model.eval()`, dropout would still zero every
activation during the forward pass and the loss would come out as `1.0`
(prediction `0` vs. target `1`), not `0.0`.

One more detail worth copying: **restore the previous mode on the way
out**. A function that leaves the model in `eval()` and gets called from
inside a training loop silently disables dropout for the rest of training.

**In `exercises.py`:** `evaluate` is this section's target — mean loss over
the loader, in eval mode, without building a graph, restoring whatever mode
the model was in beforehand.

**Test it:** `python3 -m unittest test_exercises.TestEvaluate -v`

## 7. Learning-rate schedules

```
warm-up:  lr * (step + 1) / warmup_steps
decay:    min_lr + (base - min_lr) * 0.5 * (1 + cos(pi * progress))
```

**Warm-up** exists because Adam's second-moment estimate is garbage for the
first few hundred steps — the denominator is based on almost no data, so a
full-size update early on can wreck the initialisation. Large-batch
training and transformers need it; small CNNs with SGD often do not.

**Cosine** rather than step decay because it spends more time at small
learning rates near the end, where the fine detail is learned. It also has
no hyperparameters to tune beyond the length, which is why it became the
default.

```python
>>> [cosine_warmup_lr(s, 1.0, 10, 100) for s in [0, 3, 6, 9]]   # linear ramp
[0.1, 0.4, 0.7, 1.0]
>>> cosine_warmup_lr(50, 1.0, 0, 100)         # no warm-up, halfway through decay
0.5
>>> cosine_warmup_lr(100, 1.0, 10, 100, min_lr=0.01)   # decay finishes at min_lr
0.01
```

Call `scheduler.step()` once per **step** or once per **epoch** — matching
however you defined `total_steps`. Mixing the two is a common bug that
makes the schedule finish 100× too early.

**In `exercises.py`:** `cosine_warmup_lr` is this section's target — the
linear warm-up followed by the cosine decay formula above.

**Test it:** `python3 -m unittest test_exercises.TestCosineWarmup -v`

## 8. Checkpoints must be resumable

```python
torch.save({"model": model.state_dict(),
            "optimiser": optimiser.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch}, path)
```

The **optimiser state** is what people leave out. Adam carries two moment
estimates per parameter; resuming without them restarts the optimiser cold
and produces a visible loss spike that looks like a data problem.

```python
torch.load(path, map_location="cpu", weights_only=True)
```

- **`map_location="cpu"`** so a checkpoint saved from GPU 3 loads on a
  machine that has one GPU.
- **`weights_only=True`** because an unrestricted `torch.load` unpickles
  arbitrary objects — a downloaded checkpoint can execute code. It became
  the default in torch 2.6. This is a good security answer to have ready,
  not just an API detail.

```python
>>> torch.manual_seed(0)
>>> model = nn.Linear(4, 2)
>>> optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
>>> model(torch.randn(4, 4)).sum().backward()
>>> optimiser.step()
>>> save_checkpoint("/tmp/ckpt.pt", model, optimiser, epoch=3, best_metric=0.92)
>>> fresh_model = nn.Linear(4, 2)
>>> fresh_optimiser = torch.optim.Adam(fresh_model.parameters(), lr=0.01)
>>> payload = load_checkpoint("/tmp/ckpt.pt", fresh_model, fresh_optimiser)
>>> sorted(payload.keys())
['best_metric', 'epoch', 'model', 'optimiser']
>>> payload["epoch"], payload["best_metric"]
(3, 0.92)
>>> torch.allclose(fresh_model.weight, model.weight)  # weights round-tripped
True
```

**In `exercises.py`:** `save_checkpoint` and `load_checkpoint` are this
section's targets — save model/optimiser state plus `epoch` and any
`**extra` keys, and restore them in place via `weights_only=True`.

**Test it:** `python3 -m unittest test_exercises.TestCheckpoints -v`

## 9. Overfit one batch — the first thing to run

Take 8 samples and train until the loss is ~0.

```python
>>> torch.manual_seed(0)
>>> features, labels = torch.randn(8, 4), torch.randint(0, 3, (8,))
>>> model = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 3))
>>> history = overfit_one_batch(model, features, labels, nn.CrossEntropyLoss())
>>> history[0]
1.0627373456954956
>>> history[-1]
2.011653350564302e-06
```

A model that **cannot** memorise one batch has a bug: a detached graph, a
frozen layer, a label misalignment, a learning rate of zero, a loss
applied to the wrong axis. No amount of data fixes any of those, and this
check finds them in thirty seconds instead of eight hours.

A model that *can* overfit one batch might still be wrong — but the wiring
is connected, and you have eliminated the entire class of "it never had a
chance" bugs. It is the single highest-value habit in this module.

**In `exercises.py`:** `overfit_one_batch` is this section's target — an
Adam loop over `steps` iterations on the fixed batch, returning the loss
history.

**Test it:** `python3 -m unittest test_exercises.TestOverfitOneBatch -v`

## 10. Class imbalance

```
weight_c = N / (C * count_c)
```

```python
>>> class_weights(torch.tensor([0, 0, 0, 0, 1]), 2)   # N=5, C=2, counts=[4, 1]
tensor([0.6250, 2.5000])
>>> class_weights(torch.tensor([0, 0, 1]), 4)          # classes 2, 3 never occur
tensor([0.3750, 0.7500, 0.0000, 0.0000])
```

sklearn's "balanced" convention, and it plugs straight into
`nn.CrossEntropyLoss(weight=...)`. Without it, a 99%-negative dataset
trains a model that predicts the majority class and reports 99% accuracy.

The alternatives worth naming: resampling (oversample the minority /
undersample the majority), focal loss (down-weights easy examples, which is
how dense detectors handle a 1000:1 background ratio), and — most
importantly — **changing the metric**. Accuracy on imbalanced data is
meaningless; use precision/recall, F1, or AUC-PR.

Absent classes get weight 0 rather than infinity, or one missing class in a
shard takes the whole loss to NaN.

**In `exercises.py`:** `class_weights` is this section's target — inverse
frequency weights over `num_classes`, with absent classes mapped to `0`
instead of dividing by zero.

**Test it:** `python3 -m unittest test_exercises.TestClassWeights -v`

## 11. Reproducible batching: `build_loader`

`seed_everything` (§4) and `worker_init_fn` (§3) only help if the
`DataLoader` you actually build wires them in. `build_loader` is that
assembly point:

```python
generator = torch.Generator().manual_seed(seed)
return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                  generator=generator if shuffle else None,
                  worker_init_fn=worker_init_fn, **kwargs)
```

```python
>>> torch.manual_seed(0)
>>> dataset = ArrayDataset(torch.randn(8, 4), torch.arange(8))
>>> order_a = [b[1] for b in build_loader(dataset, batch_size=4, seed=7)]
>>> order_b = [b[1] for b in build_loader(dataset, batch_size=4, seed=7)]
>>> order_a
[tensor([1, 2, 3, 5]), tensor([7, 4, 6, 0])]
>>> order_b                                    # same seed -> same batch order
[tensor([1, 2, 3, 5]), tensor([7, 4, 6, 0])]
>>> len(list(build_loader(dataset, batch_size=3, drop_last=True)))  # 8 // 3
2
```

Passing an explicitly seeded `generator=` — not just calling
`torch.manual_seed` beforehand — is what makes the shuffle order
reproducible independent of anything else that has drawn from the global
RNG first. `drop_last` matters more than it looks: a final batch of size 1
makes BatchNorm raise, because it cannot compute a variance from one
sample.

**In `exercises.py`:** `build_loader` is this section's target — construct
the `DataLoader` with a seeded `generator` (only when `shuffle=True`) and
`worker_init_fn` from §3 wired in, forwarding `**kwargs` such as
`drop_last`.

**Test it:** `python3 -m unittest test_exercises.TestBuildLoader -v`

---

## The exercises

| Function | What it teaches |
|---|---|
| `ArrayDataset` | One example, not a batch |
| `pad_collate` | Custom collate; returning lengths |
| `worker_init_fn` | The forked-RNG augmentation bug |
| `seed_everything` | Three generators, and what seeding does not cover |
| `train_one_epoch` | The four lines; per-sample averaging |
| `evaluate` | `eval()` + `no_grad`; restoring the mode |
| `cosine_warmup_lr` | Why warm-up, why cosine |
| `save_checkpoint` / `load_checkpoint` | Resumable state; `weights_only` |
| `overfit_one_batch` | The sanity check to run first |
| `class_weights` | Imbalance; avoiding an infinite weight |
| `build_loader` | Seeded shuffling; `drop_last` and BatchNorm |

## Check your understanding

1. **Why does a `Dataset`'s `__getitem__` return one example, when a
   training step needs a whole batch?**
   <details><summary>Answer</summary>Batching, shuffling and collation
   are the `DataLoader`'s job, not the `Dataset`'s — separating them
   lets the same dataset be reused with different batch sizes, samplers,
   or collate functions without changing it.</details>

2. **What's the actual bug that `worker_init_fn` fixes?**
   <details><summary>Answer</summary>Each DataLoader worker process is
   forked with the *same* RNG state, so without explicit per-worker
   seeding, random augmentations (like a random crop) produce identical
   "random" results across workers instead of genuinely varying per
   sample.</details>

3. **Why does `evaluate` need to restore the model's training mode
   afterward, not just call `model.eval()` at the start?**
   <details><summary>Answer</summary>If the caller had the model in
   training mode before calling `evaluate`, leaving it in `eval()` mode
   afterward would silently disable dropout and freeze BatchNorm
   statistics for the *next* training step, unless the caller remembers
   to switch it back themselves.</details>

4. **Why is `overfit_one_batch` — deliberately memorizing a tiny
   dataset — a useful sanity check, given that it's the opposite of
   what you actually want a model to do?**
   <details><summary>Answer</summary>If a model can't even drive the
   loss to near-zero on a handful of examples it's allowed to memorize,
   something is broken in the pipeline (a bug in the loss, the data, or
   the optimizer) — it isolates infrastructure bugs from the much
   harder question of whether the model generalizes.</details>

5. **Why does `class_weights` need to guard against an infinite
   weight?**
   <details><summary>Answer</summary>A weight computed as the inverse of
   a class's frequency blows up to infinity for a class with zero
   examples in the current data — an unguarded computation would produce
   `inf` or `NaN` weights that poison the loss.</details>

6. **Why does `build_loader` need its own seeded shuffling, separate
   from `seed_everything`?**
   <details><summary>Answer</summary><code>seed_everything</code> seeds
   the global RNGs at startup, but a `DataLoader`'s shuffling uses its
   own generator — without seeding that specifically, epoch-to-epoch (or
   run-to-run) shuffle order isn't actually reproducible even with the
   global seeds set.</details>

## Interview questions this module answers

- *Walk me through a training loop.*
- *Why does my model train more slowly with more DataLoader workers?*
- *How do you make a run reproducible — and what still is not?*
- *What goes in a checkpoint?*
- *Your model is not learning. What do you check first?*
- *Why do transformers need learning-rate warm-up?*
- *Your dataset is 99% one class. What do you do?*
- *Why is the validation loss computed differently from the training loss?*
