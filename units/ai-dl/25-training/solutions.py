"""Reference answers for 25 - data loading and the training loop."""

import math
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class ArrayDataset(Dataset):
    """The minimum a map-style Dataset needs: __len__ and __getitem__.

    __getitem__ returns ONE example, not a batch - the DataLoader does the
    batching. Returning a batch here is a common first mistake and produces
    an extra leading dimension that only shows up in the loss.
    """

    def __init__(self, features, labels):
        if len(features) != len(labels):
            raise ValueError("features and labels must be the same length")
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


def pad_collate(batch, pad_value=0):
    """Collate variable-length sequences into (padded, lengths).

    The default collate_fn calls torch.stack, which requires identical
    shapes - so the moment your samples are sentences, or detections, or
    point clouds, you must write this yourself.

    Returning the true lengths alongside the padding is what lets the model
    mask the padded positions out. Padding without lengths is how padding
    silently becomes part of the input.
    """
    sequences = [torch.as_tensor(item) for item in batch]
    lengths = torch.tensor([len(s) for s in sequences])
    longest = int(lengths.max()) if len(sequences) else 0
    padded = torch.full((len(sequences), longest), pad_value,
                        dtype=sequences[0].dtype if sequences else torch.long)
    for row, sequence in enumerate(sequences):
        padded[row, :len(sequence)] = sequence
    return padded, lengths


def seed_everything(seed):
    """Seed Python, NumPy and torch so a run is reproducible.

    Three separate generators, and missing one is enough to make a run
    unreproducible - most often NumPy, because it is what augmentation
    pipelines use.

    Note what this does NOT buy you: DataLoader worker processes get their
    own seeds (see `worker_init_fn`), and cuDNN picks algorithms
    non-deterministically unless you also set
    `torch.use_deterministic_algorithms(True)`.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def worker_init_fn(worker_id):
    """Give each DataLoader worker a distinct seed.

    The bug this fixes is famous: fork() copies the parent's RNG state, so
    every worker generates the SAME "random" augmentations. With 4 workers
    you get each augmentation four times per epoch and never notice, because
    the loss still goes down - just more slowly than it should.

    torch seeds its own workers correctly; numpy and random are the ones
    that need this.
    """
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def train_one_epoch(model, loader, optimiser, loss_fn):
    """Run one epoch, returning the mean loss per SAMPLE.

    The four lines, in the only order that works:

        optimiser.zero_grad()    # .grad accumulates; without this the
        loss.backward()          # gradients of every earlier batch are
        clip / inspect           # still in there
        optimiser.step()

    `loss.item() * len(batch)` then dividing by the dataset size is not
    pedantry: averaging the per-batch means over-weights a short final
    batch, which is exactly where a "why does my val loss look different"
    discrepancy comes from.
    """
    model.train()
    total, seen = 0.0, 0
    for features, labels in loader:
        optimiser.zero_grad()
        predictions = model(features)
        loss = loss_fn(predictions, labels)
        loss.backward()
        optimiser.step()
        total += loss.item() * len(features)
        seen += len(features)
    return total / max(seen, 1)


@torch.no_grad()
def evaluate(model, loader, loss_fn):
    """Mean loss over the loader, in eval mode, without building a graph.

    Both halves matter and they do different things:
      - `model.eval()` switches dropout off and makes batch-norm use its
        running statistics instead of the batch's.
      - `no_grad` stops the graph being recorded, which is the memory win.

    Restoring the previous mode on the way out means this is safe to call
    from inside a training loop.
    """
    was_training = model.training
    model.eval()
    try:
        total, seen = 0.0, 0
        for features, labels in loader:
            loss = loss_fn(model(features), labels)
            total += loss.item() * len(features)
            seen += len(features)
        return total / max(seen, 1)
    finally:
        model.train(was_training)


def cosine_warmup_lr(step, base_lr, warmup_steps, total_steps, min_lr=0.0):
    """Learning rate at `step`: linear warm-up, then cosine decay.

    The default schedule for transformers. Warm-up exists because Adam's
    second-moment estimate is garbage for the first few hundred steps, so a
    full-size update early on can wreck the initialisation.

    Cosine rather than step decay because it spends more time at small
    learning rates near the end, where the fine-tuning happens.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = ((step - warmup_steps) / max(total_steps - warmup_steps, 1)
                if total_steps > warmup_steps else 1.0)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def save_checkpoint(path, model, optimiser, epoch, **extra):
    """Save everything needed to RESUME, not just to run inference.

    The optimiser state is the part people forget. Adam carries two moment
    estimates per parameter; resuming without them restarts the optimiser
    cold and produces a visible loss spike. The scheduler and the epoch
    number are the same argument.
    """
    torch.save({"model": model.state_dict(),
                "optimiser": optimiser.state_dict(),
                "epoch": epoch,
                **extra}, path)
    return path


def load_checkpoint(path, model, optimiser=None):
    """Restore a checkpoint in place; return the payload.

    `weights_only=True` is the safe default: an unrestricted torch.load
    unpickles arbitrary objects, which means a downloaded checkpoint can
    execute code. It became the default in torch 2.6, and is worth knowing
    as a security answer, not just an API detail.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"])
    if optimiser is not None and "optimiser" in payload:
        optimiser.load_state_dict(payload["optimiser"])
    return payload


def overfit_one_batch(model, features, labels, loss_fn, steps=200, lr=0.1):
    """Drive the loss on a single batch towards zero; return the loss history.

    The first thing to run on any new model. A model that CANNOT overfit
    one batch has a bug - a detached graph, a frozen layer, a label
    mismatch, a learning rate of zero - and no amount of data will fix it.
    A model that can might still be wrong, but at least the wiring is
    connected.
    """
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    history = []
    for _ in range(steps):
        optimiser.zero_grad()
        loss = loss_fn(model(features), labels)
        loss.backward()
        optimiser.step()
        history.append(loss.item())
    return history


def class_weights(labels, num_classes):
    """Inverse-frequency weights for an imbalanced classification problem.

    Weight_c = N / (C * count_c), normalised so the mean weight is 1 - the
    same convention as sklearn's "balanced". Feeding these to
    `nn.CrossEntropyLoss(weight=...)` stops a 99%-negative dataset training
    a model that predicts the majority class and reports 99% accuracy.

    Absent classes get weight 0 rather than inf.
    """
    labels = torch.as_tensor(labels)
    counts = torch.bincount(labels, minlength=num_classes).float()
    weights = torch.where(counts > 0, len(labels) / (num_classes * counts),
                          torch.zeros_like(counts))
    return weights


def build_loader(dataset, batch_size, shuffle=True, seed=0, **kwargs):
    """A DataLoader whose shuffling is reproducible.

    Passing an explicitly seeded generator is what makes two runs see the
    same batch order. `drop_last` matters more than it looks: a final batch
    of size 1 makes BatchNorm raise, because it cannot compute a variance
    from one sample.
    """
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      generator=generator if shuffle else None,
                      worker_init_fn=worker_init_fn, **kwargs)
