"""Reference answers for 27 - convolution, detection and video."""

import torch
import torch.nn.functional as F


def conv_output_size(size, kernel, stride=1, padding=0, dilation=1):
    """Output length of a convolution along one axis.

        out = floor((size + 2*padding - dilation*(kernel-1) - 1) / stride) + 1

    Worth memorising, because you will be asked to compute a feature-map
    size on a whiteboard. Two corollaries to have ready:
      - "same" padding for an odd kernel with stride 1 is (kernel-1)//2.
      - Dilation enlarges the kernel's REACH without adding parameters:
        an effective kernel of dilation*(kernel-1)+1.

    Raises ValueError when the kernel does not fit, rather than returning
    a negative size that fails much later inside a matmul.
    """
    effective = dilation * (kernel - 1) + 1
    out = (size + 2 * padding - effective) // stride + 1
    if out <= 0:
        raise ValueError(
            f"kernel {kernel} (effective {effective}) does not fit in "
            f"{size} with padding {padding}")
    return out


def receptive_field(layers):
    """Receptive field of a stack of [(kernel, stride), ...] layers.

        r_0 = 1;  r_i = r_{i-1} + (k_i - 1) * jump_{i-1}
        jump_i = jump_{i-1} * s_i

    The jump (the cumulative stride) is the part people forget: a 3x3 after
    two stride-2 layers sees 4 input pixels per step, not 1.

    Why it matters: if your receptive field is smaller than the object you
    are detecting, no amount of training fixes it - the network physically
    cannot see the whole thing. This is the argument for dilated
    convolutions in segmentation.
    """
    field, jump = 1, 1
    for kernel, stride in layers:
        field += (kernel - 1) * jump
        jump *= stride
    return field


def conv2d_via_unfold(x, weight, bias=None, stride=1, padding=0):
    """2-D convolution written as one matrix multiply (im2col).

    x: (B, C, H, W), weight: (O, C, kh, kw) -> (B, O, H', W').

    `unfold` extracts every sliding patch into a column, turning the
    convolution into (O, C*kh*kw) @ (C*kh*kw, L). This is how convolution
    is ACTUALLY implemented on hardware: GEMM is the one kernel every
    accelerator is fast at, so everything gets reshaped into one.

    The cost is memory - patches overlap, so the column matrix is kh*kw
    times larger than the input. That trade-off is the whole reason
    implicit-GEMM and FlashAttention-style fused kernels exist.
    """
    batch, _, height, width = x.shape
    out_channels, in_channels, kh, kw = weight.shape
    out_h = conv_output_size(height, kh, stride, padding)
    out_w = conv_output_size(width, kw, stride, padding)

    columns = F.unfold(x, (kh, kw), stride=stride, padding=padding)
    # (B, C*kh*kw, L) with the weight flattened to (O, C*kh*kw).
    out = weight.reshape(out_channels, -1) @ columns
    out = out.reshape(batch, out_channels, out_h, out_w)
    if bias is not None:
        out = out + bias.reshape(1, -1, 1, 1)
    return out


def box_iou(boxes_a, boxes_b):
    """IoU between every pair of boxes in (x1, y1, x2, y2) format.

    a: (N, 4), b: (M, 4) -> (N, M).

    The detection warm-up question. `clamp(min=0)` on the overlap is what
    handles boxes that do not intersect - without it a negative width times
    a negative height gives a positive "intersection", which is a genuinely
    hard bug to see in a metric.
    """
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    top_left = torch.max(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = torch.min(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    overlap = (bottom_right - top_left).clamp(min=0)
    intersection = overlap[..., 0] * overlap[..., 1]

    union = area_a[:, None] + area_b[None, :] - intersection
    return intersection / union.clamp(min=1e-9)


def nms(boxes, scores, iou_threshold=0.5):
    """Non-maximum suppression. Returns kept indices, highest score first.

    Take the best box, drop everything overlapping it by more than the
    threshold, repeat. Every detector ends with this, and it is asked in
    interviews because the naive version is easy and the vectorised one
    shows you can think in tensors.

    Note the threshold is exclusive: IoU exactly equal to the threshold is
    kept. Conventions differ, so say which one you are using.
    """
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        best = order[0]
        keep.append(best)
        if order.numel() == 1:
            break
        ious = box_iou(boxes[best][None], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_threshold]
    return torch.stack(keep)


def patchify(images, patch_size):
    """(B, C, H, W) -> (B, num_patches, C*patch*patch), row-major.

    The first layer of a Vision Transformer: cut the image into
    non-overlapping patches and flatten each one into a token. A 224x224
    image with 16x16 patches gives 196 tokens - which, at O(L^2) attention,
    is why ViT input resolution is such a constrained choice.

    unfold with stride == kernel is exactly "non-overlapping patches".
    """
    batch, channels, height, width = images.shape
    if height % patch_size or width % patch_size:
        raise ValueError(
            f"{height}x{width} is not divisible by patch size {patch_size}")
    patches = F.unfold(images, patch_size, stride=patch_size)
    # (B, C*p*p, L) -> (B, L, C*p*p): tokens first, like a transformer wants.
    return patches.transpose(1, 2)


def depthwise_separable_params(in_channels, out_channels, kernel_size):
    """(standard, separable) parameter counts, excluding bias.

    Standard:  out * in * k * k
    Separable: in * k * k  (depthwise)  +  in * out  (pointwise 1x1)

    The saving is roughly 1/out + 1/k^2 - about 9x for a 3x3 with many
    channels. This one arithmetic fact is MobileNet, and it is the standard
    "how would you make this model run on a phone" answer.
    """
    standard = out_channels * in_channels * kernel_size * kernel_size
    separable = (in_channels * kernel_size * kernel_size
                 + in_channels * out_channels)
    return standard, separable


def normalise_batch(images, mean, std):
    """Per-channel normalisation of a (B, C, H, W) batch.

    The trap is the shape: mean and std are (C,), and subtracting them
    directly broadcasts against the LAST axis - the width - silently
    normalising the wrong dimension on a square image and crashing on a
    non-square one. They must be reshaped to (1, C, 1, 1).

    This is the single most common bug in a hand-written preprocessing
    pipeline, and the reason the tests here use a non-square image.
    """
    mean = torch.as_tensor(mean, dtype=images.dtype,
                           device=images.device).reshape(1, -1, 1, 1)
    std = torch.as_tensor(std, dtype=images.dtype,
                          device=images.device).reshape(1, -1, 1, 1)
    return (images - mean) / std


def sample_frames(num_frames, num_samples):
    """Indices of `num_samples` frames spread evenly over a clip.

    Video models take a fixed number of frames; real clips have arbitrary
    lengths. Uniform sampling over the whole clip beats taking the first N,
    which throws away everything after the first second.

    Short clips are handled by repeating frames rather than raising, since
    a data pipeline that crashes on one short video is worse than one that
    pads. The midpoint offset (+0.5) centres each sample in its segment
    instead of biasing every one towards the start.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    positions = (torch.arange(num_samples, dtype=torch.float32) + 0.5)
    indices = (positions * num_frames / num_samples).long()
    return indices.clamp(max=num_frames - 1)


def frame_difference(clip, stride=1):
    """Temporal difference of a (T, C, H, W) clip -> (T - stride, C, H, W).

    The cheapest possible motion feature, and a genuinely useful baseline:
    a two-stream network's optical-flow branch is doing a more expensive
    version of this. Worth being able to say that a large part of what a
    video model learns from "motion" is available from frame differences at
    almost no cost.
    """
    if stride < 1:
        raise ValueError("stride must be at least 1")
    if clip.shape[0] <= stride:
        raise ValueError(
            f"clip of {clip.shape[0]} frames is too short for stride {stride}")
    return clip[stride:] - clip[:-stride]
