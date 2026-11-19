"""Reference answers for 22 - tensors and where they differ from NumPy."""

import torch


def as_tensor_like(data, reference):
    """Turn `data` into a tensor on the same device and dtype as `reference`.

    `torch.tensor(...)` always allocates on the CPU in the default dtype, so
    the very next op raises "Expected all tensors to be on the same device".
    `.to(reference)` copies device AND dtype in one call - the shortest
    correct answer, and the one to give in an interview.
    """
    return torch.as_tensor(data).to(reference)


def shares_storage(a, b):
    """True if the two tensors read the same memory.

    Same rule as NumPy: basic slicing, `view`, `permute`, `transpose` and
    `expand` return VIEWS; `reshape` returns a view when it can and a copy
    when it cannot; indexing with a list or a bool mask always copies.
    """
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


def safe_normalise(x, eps=1e-6):
    """Return x scaled to zero mean and unit variance, WITHOUT touching x.

    The trap is `x -= x.mean()`. That is an in-place op: it mutates the
    caller's tensor, and if x needs a gradient it also corrupts the autograd
    graph ("a leaf Variable that requires grad is being used in an in-place
    operation").
    """
    return (x - x.mean()) / (x.std() + eps)


def flatten_batch(x):
    """(B, ...) -> (B, N), keeping the batch dimension.

    `x.view(B, -1)` fails on a non-contiguous tensor - a transposed or
    permuted feature map, which is exactly what you get after `permute`.
    `reshape` copies when it has to, so it always works. `flatten(1)` says
    what you mean.
    """
    return x.flatten(1)


def to_channels_last(x):
    """(B, C, H, W) -> a tensor whose memory is laid out as (B, H, W, C).

    Channels-last is what convolution kernels and Tensor Cores want; the
    logical SHAPE is unchanged, only the strides move. This is the tensor
    version of NumPy's contiguity question, and the reason models are
    trained with `model.to(memory_format=torch.channels_last)`.
    """
    return x.contiguous(memory_format=torch.channels_last)


def batched_pairwise_distance(a, b, eps=1e-12):
    """Euclidean distance between every pair of rows, per batch element.

    a: (B, N, D), b: (B, M, D) -> (B, N, M).

    `torch.cdist` is the built-in and is what you should reach for in real
    code. The point of writing it out is the unsqueeze pattern:
    (B, N, 1, D) against (B, 1, M, D) broadcasts to (B, N, M, D).

    `clamp_min(eps)` before the sqrt is not cosmetic. sqrt has an INFINITE
    derivative at 0, and the squared distance of a point to itself is
    exactly 0 - so the chain rule computes 0 * inf = NaN and the loss goes
    to NaN on the first step. Note that clamping to 0 does not help: the
    gradient only gets cut off when the input is strictly below the clamp.
    This bites every triplet/contrastive loss ever written.
    """
    diff = a.unsqueeze(2) - b.unsqueeze(1)
    return diff.pow(2).sum(-1).clamp_min(eps).sqrt()


def masked_softmax(scores, mask, dim=-1):
    """Softmax over `dim`, ignoring positions where mask is False.

    Zeroing the probabilities after a plain softmax is wrong: the remaining
    entries no longer sum to 1. Masking must happen BEFORE the exponential,
    by setting the excluded logits to -inf.

    Rows with nothing unmasked would be 0/0; they are returned as all-zero
    rather than NaN, which is what an attention layer wants for a padded row.
    """
    neg_inf = torch.finfo(scores.dtype).min
    masked = scores.masked_fill(~mask, neg_inf)
    weights = torch.softmax(masked, dim=dim)
    # A fully masked row softmaxes to a uniform distribution over -inf; kill it.
    any_valid = mask.any(dim=dim, keepdim=True)
    return torch.where(any_valid, weights, torch.zeros_like(weights))


def count_parameters(module, trainable_only=True):
    """Number of scalar parameters in a module.

    `p.numel()` counts elements, not tensors. Filtering on `requires_grad` is
    what makes the number match "trainable parameters" in a model summary
    after you have frozen a backbone.
    """
    return sum(p.numel() for p in module.parameters()
               if p.requires_grad or not trainable_only)


def memory_bytes(shape, dtype=torch.float32):
    """Bytes a tensor of this shape and dtype occupies.

    The arithmetic behind "why did this OOM". A (32, 3, 224, 224) float32
    batch is 19 MB; the activations behind it are many times that, because
    every layer's output is kept alive for the backward pass.
    """
    count = 1
    for size in shape:
        count *= size
    return count * torch.empty((), dtype=dtype).element_size()


def stable_logsumexp(x, dim=-1):
    """log(sum(exp(x))) without overflowing.

    Subtracting the max is the same trick as a stable softmax: exp of a
    large logit is inf, and log(inf) is inf. Written out here because
    interviewers ask for the derivation, not the built-in:

        log Σ exp(x) = m + log Σ exp(x - m),  m = max(x)
    """
    maximum = x.max(dim=dim, keepdim=True).values
    # An all-(-inf) row would give -inf + inf = NaN; clamp the shift to finite.
    maximum = torch.where(torch.isfinite(maximum), maximum,
                          torch.zeros_like(maximum))
    shifted = (x - maximum).exp().sum(dim=dim, keepdim=True).log()
    return (maximum + shifted).squeeze(dim)
