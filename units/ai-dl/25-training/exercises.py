"""25 - data loading and the training loop

Implement each function so the tests in test_exercises.py pass. Read
README.md first; every exercise below is covered there.

    ../run 25            test your work
    ../run 25 --check    watch the reference solutions pass

The docstrings are the specification. Standard library only.
"""

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
        raise NotImplementedError("your code here")

    def __getitem__(self, index):
        raise NotImplementedError("your code here")


def pad_collate(batch, pad_value=0):
    """Collate variable-length sequences into (padded, lengths).

    The default collate_fn calls torch.stack, which requires identical
    shapes - so the moment your samples are sentences, or detections, or
    point clouds, you must write this yourself.

    Returning the true lengths alongside the padding is what lets the model
    mask the padded positions out. Padding without lengths is how padding
    silently becomes part of the input.
    """
    raise NotImplementedError("your code here")


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
    raise NotImplementedError("your code here")


def worker_init_fn(worker_id):
    """Give each DataLoader worker a distinct seed.

    The bug this fixes is famous: fork() copies the parent's RNG state, so
    every worker generates the SAME "random" augmentations. With 4 workers
    you get each augmentation four times per epoch and never notice, because
    the loss still goes down - just more slowly than it should.

    torch seeds its own workers correctly; numpy and random are the ones
    that need this.
    """
    raise NotImplementedError("your code here")


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
    raise NotImplementedError("your code here")


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
    raise NotImplementedError("your code here")


def cosine_warmup_lr(step, base_lr, warmup_steps, total_steps, min_lr=0.0):
    """Learning rate at `step`: linear warm-up, then cosine decay.

    The default schedule for transformers. Warm-up exists because Adam's
    second-moment estimate is garbage for the first few hundred steps, so a
    full-size update early on can wreck the initialisation.

    Cosine rather than step decay because it spends more time at small
    learning rates near the end, where the fine-tuning happens.
    """
    raise NotImplementedError("your code here")


def save_checkpoint(path, model, optimiser, epoch, **extra):
    """Save everything needed to RESUME, not just to run inference.

    The optimiser state is the part people forget. Adam carries two moment
    estimates per parameter; resuming without them restarts the optimiser
    cold and produces a visible loss spike. The scheduler and the epoch
    number are the same argument.
    """
    raise NotImplementedError("your code here")


def load_checkpoint(path, model, optimiser=None):
    """Restore a checkpoint in place; return the payload.

    `weights_only=True` is the safe default: an unrestricted torch.load
    unpickles arbitrary objects, which means a downloaded checkpoint can
    execute code. It became the default in torch 2.6, and is worth knowing
    as a security answer, not just an API detail.
    """
    raise NotImplementedError("your code here")


def overfit_one_batch(model, features, labels, loss_fn, steps=200, lr=0.1):
    """Drive the loss on a single batch towards zero; return the loss history.

    The first thing to run on any new model. A model that CANNOT overfit
    one batch has a bug - a detached graph, a frozen layer, a label
    mismatch, a learning rate of zero - and no amount of data will fix it.
    A model that can might still be wrong, but at least the wiring is
    connected.
    """
    raise NotImplementedError("your code here")


def class_weights(labels, num_classes):
    """Inverse-frequency weights for an imbalanced classification problem.

    Weight_c = N / (C * count_c), normalised so the mean weight is 1 - the
    same convention as sklearn's "balanced". Feeding these to
    `nn.CrossEntropyLoss(weight=...)` stops a 99%-negative dataset training
    a model that predicts the majority class and reports 99% accuracy.

    Absent classes get weight 0 rather than inf.
    """
    raise NotImplementedError("your code here")


def build_loader(dataset, batch_size, shuffle=True, seed=0, **kwargs):
    """A DataLoader whose shuffling is reproducible.

    Passing an explicitly seeded generator is what makes two runs see the
    same batch order. `drop_last` matters more than it looks: a final batch
    of size 1 makes BatchNorm raise, because it cannot compute a variance
    from one sample.
    """
    raise NotImplementedError("your code here")
