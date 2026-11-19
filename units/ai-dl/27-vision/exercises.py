"""27 - convolution, detection and video

Implement each function so the tests in test_exercises.py pass. Read
README.md first; every exercise below is covered there.

    ../run 27            test your work
    ../run 27 --check    watch the reference solutions pass

The docstrings are the specification. Standard library only.
"""

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
    raise NotImplementedError("your code here")


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
    raise NotImplementedError("your code here")


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
    raise NotImplementedError("your code here")


def box_iou(boxes_a, boxes_b):
    """IoU between every pair of boxes in (x1, y1, x2, y2) format.

    a: (N, 4), b: (M, 4) -> (N, M).

    The detection warm-up question. `clamp(min=0)` on the overlap is what
    handles boxes that do not intersect - without it a negative width times
    a negative height gives a positive "intersection", which is a genuinely
    hard bug to see in a metric.
    """
    raise NotImplementedError("your code here")


def nms(boxes, scores, iou_threshold=0.5):
    """Non-maximum suppression. Returns kept indices, highest score first.

    Take the best box, drop everything overlapping it by more than the
    threshold, repeat. Every detector ends with this, and it is asked in
    interviews because the naive version is easy and the vectorised one
    shows you can think in tensors.

    Note the threshold is exclusive: IoU exactly equal to the threshold is
    kept. Conventions differ, so say which one you are using.
    """
    raise NotImplementedError("your code here")


def patchify(images, patch_size):
    """(B, C, H, W) -> (B, num_patches, C*patch*patch), row-major.

    The first layer of a Vision Transformer: cut the image into
    non-overlapping patches and flatten each one into a token. A 224x224
    image with 16x16 patches gives 196 tokens - which, at O(L^2) attention,
    is why ViT input resolution is such a constrained choice.

    unfold with stride == kernel is exactly "non-overlapping patches".
    """
    raise NotImplementedError("your code here")


def depthwise_separable_params(in_channels, out_channels, kernel_size):
    """(standard, separable) parameter counts, excluding bias.

    Standard:  out * in * k * k
    Separable: in * k * k  (depthwise)  +  in * out  (pointwise 1x1)

    The saving is roughly 1/out + 1/k^2 - about 9x for a 3x3 with many
    channels. This one arithmetic fact is MobileNet, and it is the standard
    "how would you make this model run on a phone" answer.
    """
    raise NotImplementedError("your code here")


def normalise_batch(images, mean, std):
    """Per-channel normalisation of a (B, C, H, W) batch.

    The trap is the shape: mean and std are (C,), and subtracting them
    directly broadcasts against the LAST axis - the width - silently
    normalising the wrong dimension on a square image and crashing on a
    non-square one. They must be reshaped to (1, C, 1, 1).

    This is the single most common bug in a hand-written preprocessing
    pipeline, and the reason the tests here use a non-square image.
    """
    raise NotImplementedError("your code here")


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
    raise NotImplementedError("your code here")


def frame_difference(clip, stride=1):
    """Temporal difference of a (T, C, H, W) clip -> (T - stride, C, H, W).

    The cheapest possible motion feature, and a genuinely useful baseline:
    a two-stream network's optical-flow branch is doing a more expensive
    version of this. Worth being able to say that a large part of what a
    video model learns from "motion" is available from frame differences at
    almost no cost.
    """
    raise NotImplementedError("your code here")
