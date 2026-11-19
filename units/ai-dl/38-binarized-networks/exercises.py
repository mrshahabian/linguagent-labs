"""38 - binarized and quantized neural networks

Implement each function so the tests in test_exercises.py pass. Read
README.md first; every exercise below is covered there.

    ../run 38            test your work
    ../run 38 --check    watch the reference solutions pass

The docstrings are the specification. Standard library only.
"""

import torch
from torch import nn


class BinarizeSTE(torch.autograd.Function):
    """Binarize to {-1, +1} in the forward pass; straight-through in the backward pass.

    sign(x) has a derivative of zero almost everywhere (it's a step
    function) and is undefined at x=0 - backpropagating through it
    honestly would kill every gradient before it reaches an earlier
    layer. The STRAIGHT-THROUGH ESTIMATOR sidesteps this by pretending,
    only for the backward pass, that the forward function was the
    identity - the gradient passes through UNCHANGED, as if sign()
    weren't there at all.

    The one refinement this adds over a pure pass-through: gradient is
    zeroed wherever |x| > 1. Because forward saturates to +-1 well
    before that, a large positive x that increases further can't
    possibly change the binarized output - passing a nonzero gradient
    through there would push x even further from zero for no benefit.
    This is the "clipped" STE from BinaryConnect/XNOR-Net, as opposed to
    a fully unclipped STE that ignores this.
    """

    @staticmethod
    def forward(ctx, x):
        raise NotImplementedError("your code here")

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("your code here")


def binarize(x):
    """Apply BinarizeSTE - the function form callers actually use."""
    raise NotImplementedError("your code here")


class BinaryLinear(nn.Module):
    """A linear layer whose WEIGHTS are binarized on every forward pass.

    self.weight stays real-valued (float32) and is what the optimizer
    actually updates - it is a "latent" weight, never used directly for
    the matmul. Every forward call binarizes a FRESH view of it via STE;
    the small, real-valued gradient updates accumulate in the latent
    weight even though the forward computation only ever sees +-1s. This
    is BinaryConnect's core trick: train real-valued weights that happen
    to get binarized every step, rather than trying to optimize
    literally-discrete weights directly (which plain gradient descent
    cannot do at all - there is no small nudge that changes a discrete
    value).
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)

    def forward(self, x):
        raise NotImplementedError("your code here")


def xnor_popcount_dot(a_signs, b_signs):
    """Dot product of two +-1 vectors, computed via XNOR + popcount, not multiply-add.

    For vectors whose entries are ALWAYS +1 or -1: two entries agree
    (product +1) exactly when their bits match, and disagree (product
    -1) exactly when their bits differ. Encoding +1 as bit 1 and -1 as
    bit 0, XNOR-ing the two bit patterns gives a 1 wherever they agree -
    so popcount(xnor(a_bits, b_bits)) counts the agreements directly.
    The dot product is then agreements - disagreements = agreements -
    (N - agreements) = 2*agreements - N.

    This is BinaryConnect/XNOR-Net's actual efficiency argument: on
    hardware with a popcount instruction, this replaces N floating-point
    multiply-adds with one XNOR and one popcount over N bits - no
    floating-point multiplication happens at all.
    """
    raise NotImplementedError("your code here")


def memory_footprint_bytes(num_params):
    """Bytes to store num_params weights as float32 vs. as single bits.

    Returns (float32_bytes, binary_bytes). float32 uses 4 bytes per
    parameter; binarized weights need only 1 BIT each, packed 8 to a
    byte - roughly a 32x reduction, compared to int8 quantization's
    4x (module 30). The two techniques are not mutually exclusive in
    principle, but binarization is the more aggressive, more
    accuracy-costly end of the same spectrum quantization sits on.
    """
    raise NotImplementedError("your code here")
