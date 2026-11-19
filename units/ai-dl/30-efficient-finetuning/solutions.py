"""Reference answers for 30 - efficient fine-tuning and inference."""

import torch
from torch import nn


class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable low-rank update.

    Full fine-tuning trains every entry of a (out_features, in_features)
    weight matrix. LoRA instead freezes that matrix entirely and trains
    two small matrices A (rank x in_features) and B (out_features x rank)
    whose product B @ A approximates the weight DELTA a full fine-tune
    would have learned. For rank << min(in_features, out_features), that
    is a large reduction in trainable parameters and optimizer state -
    the entire reason LoRA exists.

    B is initialised to zero (A is not) so that at the start of training
    the LoRA path contributes exactly zero and the model behaves
    identically to the frozen base model - training begins from the
    pretrained behaviour, not from a random perturbation of it.
    """

    def __init__(self, base_linear, rank, alpha=1.0):
        super().__init__()
        self.base = base_linear
        for param in self.base.parameters():
            param.requires_grad_(False)
        out_features, in_features = self.base.weight.shape
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x):
        base_out = self.base(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        return base_out + self.scaling * lora_out

    def merge(self):
        """Fold the LoRA update into a plain nn.Linear with no LoRA overhead.

        merged_weight = base_weight + scaling * B @ A - the same shape as
        the base weight, so the merged layer costs nothing extra at
        inference: no extra matmuls, no extra parameters to load. This is
        the standard "merge adapters for deployment" step, and it is only
        possible because the LoRA update is linear in x - a non-linear
        adapter could not be folded into a single matrix this way.
        """
        merged = nn.Linear(self.base.in_features, self.base.out_features,
                            bias=self.base.bias is not None)
        with torch.no_grad():
            delta = self.scaling * (self.lora_B @ self.lora_A)
            merged.weight.copy_(self.base.weight + delta)
            if self.base.bias is not None:
                merged.bias.copy_(self.base.bias)
        return merged


def quantize_int8(tensor):
    """Symmetric per-tensor int8 quantization. Returns (q_tensor, scale).

    Symmetric means zero maps to zero - no separate zero_point is needed,
    only a scale. scale is chosen so the tensor's largest-magnitude value
    maps to +-127 (int8's range is -128..127; 127 is used on both sides
    to keep the mapping symmetric around zero). Everything in between is
    rounded to the nearest representable level, which is where
    quantization error comes from - a value between two levels cannot be
    represented exactly.
    """
    max_abs = tensor.abs().max()
    scale = max_abs / 127 if max_abs > 0 else torch.tensor(1.0)
    q = torch.clamp(torch.round(tensor / scale), -127, 127).to(torch.int8)
    return q, scale


def dequantize_int8(q_tensor, scale):
    """Reverse quantize_int8: back to float, with rounding error baked in."""
    return q_tensor.float() * scale


class Checkpoint(torch.autograd.Function):
    """The mechanism behind torch.utils.checkpoint, written out directly.

    forward() runs under no_grad and returns only the output - the
    intermediate activations inside `fn` are never saved, only the
    input. backward() recomputes fn(x) WITH grad tracking enabled just
    to get those activations back, then runs autograd through that
    fresh copy to get the gradient. fn(x) therefore runs twice per
    training step instead of once: more compute, in exchange for not
    holding fn's internal activations in memory between forward and
    backward. That trade is the entire point of gradient checkpointing -
    it turns activation memory into recomputation time, which is
    normally the cheaper resource to spend when a model barely fits.
    """

    @staticmethod
    def forward(ctx, fn, x):
        ctx.fn = fn
        ctx.save_for_backward(x)
        with torch.no_grad():
            return fn(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            y = ctx.fn(x)
        (grad_x,) = torch.autograd.grad(y, x, grad_output)
        return None, grad_x


def checkpoint(fn, x):
    """Run fn(x) without saving its internal activations for backward.

    A thin wrapper so callers don't need to know Checkpoint.apply's
    calling convention (autograd.Function.apply does not accept keyword
    arguments, and the first argument position is reserved).
    """
    return Checkpoint.apply(fn, x)


def dynamic_batches(arrival_times, max_batch_size, max_wait):
    """Group requests into batches, respecting a size cap AND a wait cap.

    A batch starts forming the moment the first pending request arrives.
    It flushes the instant EITHER max_batch_size requests have joined it,
    OR max_wait time has passed since that first arrival - whichever
    happens first. This is "dynamic batching": unlike a fixed-size static
    batch that always waits for exactly N requests (adding latency for
    the Nth request under light load), a dynamic batch also flushes on a
    timeout, so light traffic does not wait indefinitely for a batch that
    will never fill.

    One simplification worth naming: both conditions are only checked
    WHEN A NEW REQUEST ARRIVES, since this function has no clock of its
    own ticking between arrivals. A batch that is one request short of
    max_batch_size and receives no further requests will not flush until
    either another request finally arrives (however late) or the input
    ends. A production batcher needs an actual background timer to
    guarantee a latency bound on a lone pending request - this models
    the grouping decision, not the timer that would drive it live.

    Returns a list of batches, each a list of the original indices of
    the requests it contains, in arrival order.
    """
    batches = []
    current = []
    batch_start = None
    for index, arrival in enumerate(arrival_times):
        if not current:
            batch_start = arrival
        current.append(index)
        if len(current) >= max_batch_size or (arrival - batch_start) >= max_wait:
            batches.append(current)
            current = []
            batch_start = None
    if current:
        batches.append(current)
    return batches
