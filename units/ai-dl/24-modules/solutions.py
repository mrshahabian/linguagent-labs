"""Reference answers for 24 - nn.Module and model surgery."""

import torch
from torch import nn


class MLP(nn.Module):
    """A plain multi-layer perceptron with configurable hidden sizes.

    Note `nn.ModuleList`, not a Python list: a bare list is invisible to
    `.parameters()`, so the optimiser never sees those layers and they
    silently never train. Same for `nn.ParameterList` and `nn.ModuleDict`.
    This is the single most common nn.Module bug.
    """

    def __init__(self, in_features, hidden, out_features, activation=nn.ReLU):
        super().__init__()
        sizes = [in_features, *hidden]
        self.layers = nn.ModuleList(
            nn.Linear(a, b) for a, b in zip(sizes, sizes[1:]))
        self.activation = activation()
        self.head = nn.Linear(sizes[-1], out_features)

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.head(x)


class RunningMean(nn.Module):
    """Tracks an exponential moving average of the inputs it has seen.

    The average is a BUFFER, not a parameter: it is state the module owns
    but gradient descent must not touch. Buffers appear in `state_dict`
    (so they are saved and restored), move with `.to(device)`, and are
    excluded from `.parameters()`.

    That is exactly how BatchNorm's running_mean/running_var work - and
    why loading a checkpoint into a model built without them fails loudly.
    """

    def __init__(self, size, momentum=0.1):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                batch_mean = x.detach().mean(0)
                self.mean.mul_(1 - self.momentum).add_(
                    batch_mean * self.momentum)
                self.count += 1
        return x - self.mean


def freeze(module, prefix=None):
    """Freeze parameters, optionally only those whose name starts with prefix.

    Returns the number of parameters frozen. Two halves people forget:

    1. `requires_grad_(False)` alone is not enough if the optimiser was
       built over every parameter - it will still apply weight decay and
       momentum to them. Build the optimiser AFTER freezing, or filter.
    2. Freezing does not disable BatchNorm's running statistics. A frozen
       backbone left in train() mode still drifts. `module.eval()` is the
       other half.
    """
    frozen = 0
    for name, parameter in module.named_parameters():
        if prefix is None or name.startswith(prefix):
            parameter.requires_grad_(False)
            frozen += parameter.numel()
    return frozen


def init_weights(module, gain=1.0):
    """Apply Xavier-uniform init to every Linear weight; zero the biases.

    Why init matters: with a bad scale the activation variance either
    explodes or collapses layer by layer, and a deep net stops training
    entirely. Xavier keeps the variance constant for a symmetric
    activation; Kaiming (`kaiming_normal_`, nonlinearity="relu") is the
    right choice after a ReLU, because ReLU throws away half the signal.

    `module.apply(fn)` walks every submodule, which is the idiomatic way to
    do this over a whole model.
    """
    def initialise(layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    module.apply(initialise)
    return module


def tie_weights(embedding, output_layer):
    """Share one weight matrix between an embedding and an output projection.

    Standard in language models: the input embedding (V, D) and the output
    projection (V, D) learn the same thing, so tying them halves the
    parameter count and usually improves perplexity.

    Assignment - not `copy_` - is what shares them. `copy_` makes them
    equal once and they immediately diverge, which looks identical at
    initialisation and is a genuinely hard bug to spot.
    """
    if embedding.weight.shape != output_layer.weight.shape:
        raise ValueError(
            f"cannot tie {tuple(embedding.weight.shape)} to "
            f"{tuple(output_layer.weight.shape)}")
    output_layer.weight = embedding.weight
    return output_layer


def activation_stats(model, x):
    """Run x through the model, returning {layer_name: output std}.

    Forward hooks let you observe a model you did not write, without
    editing its forward(). Diagnosing a dead network usually starts here:
    if the std collapses towards 0 layer by layer the signal is vanishing;
    if it grows without bound the run is about to produce NaN.

    Every handle must be removed. Hooks left registered leak memory and
    silently slow down every subsequent forward pass.
    """
    stats = {}
    handles = []

    def make_hook(name):
        def hook(_module, _inputs, output):
            stats[name] = output.detach().float().std().item()
        return hook

    try:
        for name, layer in model.named_modules():
            if name and not list(layer.children()):   # leaves only
                handles.append(layer.register_forward_hook(make_hook(name)))
        with torch.no_grad():
            model(x)
    finally:
        for handle in handles:
            handle.remove()
    return stats


def load_partial(model, state_dict):
    """Load only the entries whose name AND shape match. Return what was skipped.

    The everyday transfer-learning problem: a checkpoint trained with 1000
    output classes going into a model with 10. `load_state_dict` refuses
    outright, and `strict=False` catches missing names but still raises on
    a shape mismatch - so the filtering has to be explicit.
    """
    own = model.state_dict()
    usable = {name: tensor for name, tensor in state_dict.items()
              if name in own and own[name].shape == tensor.shape}
    skipped = sorted(set(state_dict) - set(usable))
    model.load_state_dict(usable, strict=False)
    return skipped


def module_summary(model):
    """[(name, class name, parameter count)] for each direct child.

    The thing you actually want when someone hands you a checkpoint: where
    the parameters live. `named_children` is one level deep;
    `named_modules` recurses (and would double-count if you summed it).
    """
    return [(name, type(child).__name__,
             sum(p.numel() for p in child.parameters()))
            for name, child in model.named_children()]
