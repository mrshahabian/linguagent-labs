"""29 - optimizers and learning-rate schedules

Implement each function so the tests in test_exercises.py pass. Read
README.md first; every exercise below is covered there.

    ../run 29            test your work
    ../run 29 --check    watch the reference solutions pass

The docstrings are the specification. Standard library only.
"""

import math
import torch


def sgd_momentum_step(param, grad, velocity, lr, momentum):
    """One step of SGD with classic (heavy-ball) momentum.

    velocity accumulates a decaying running average of past gradients, so
    the update keeps moving in a direction that has been consistently
    downhill even where the CURRENT gradient is small or noisy - this is
    what carries a run through flat regions and damps oscillation across
    narrow ravines, both of which plain SGD handles badly.

    Returns (new_param, new_velocity); does not mutate its inputs, so a
    caller decides when to commit the update.
    """
    raise NotImplementedError("your code here")


def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """One step of Adam (no weight decay). Returns (new_param, new_m, new_v).

    m and v are exponential moving averages of the gradient and its
    square - a per-parameter estimate of the gradient's mean and
    variance. Dividing by sqrt(v) is what makes Adam's effective step
    size roughly uniform across parameters regardless of how differently
    scaled their raw gradients are.

    Bias correction (dividing by 1 - beta**t) matters most in the first
    few steps: m and v start at zero, so the raw moving averages are
    biased toward zero until enough steps have accumulated to wash that
    initialization out. Skipping it makes early steps far too small.
    """
    raise NotImplementedError("your code here")


def adamw_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999,
                eps=1e-8, weight_decay=0.01):
    """One step of AdamW: Adam with DECOUPLED weight decay.

    The naive way to add L2 regularisation to Adam is to fold it into the
    gradient before the moment estimates see it: `grad = grad +
    weight_decay * param`. That is what plain `torch.optim.Adam(...,
    weight_decay=...)` does, and it is subtly wrong for an adaptive
    optimizer - the decay term then gets divided by sqrt(v) along with
    the real gradient, so parameters with large gradient variance are
    decayed LESS, which was never the intent of weight decay.

    AdamW instead applies the decay directly to the parameter, entirely
    outside the adaptive (m, v) machinery:

        param <- param - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * param)

    so every parameter is decayed by the same proportion regardless of
    its gradient statistics - decoupled from the adaptive step.
    """
    raise NotImplementedError("your code here")


def cosine_warmup_lr(step, total_steps, warmup_steps, base_lr, min_lr=0.0):
    """The learning rate at `step`: linear warmup, then cosine decay to min_lr.

    Warmup exists because Adam's early steps are the least trustworthy
    ones - the moment estimates m and v are still close to their
    zero initialisation, so a large step right away can push a randomly
    initialised model somewhere the optimizer never recovers from cheaply.
    Ramping the LR up linearly over the first `warmup_steps` steps instead
    of starting at full strength avoids that.

    Cosine decay after warmup falls smoothly to min_lr rather than
    stepping down abruptly - shown empirically to out-perform a hard
    step schedule across most training runs, and it has no extra
    hyperparameter beyond "when does training end."
    """
    raise NotImplementedError("your code here")


def clip_grad_norm_(grads, max_norm):
    """Clip a list of gradient tensors IN PLACE by their combined global norm.

    The global norm treats every tensor in `grads` as one big flattened
    vector - it is the norm of the CONCATENATION, not the max or sum of
    each tensor's own norm. Scaling every tensor by the same factor
    `max_norm / total_norm` (only when the total exceeds max_norm)
    preserves the gradient's direction across all parameters jointly;
    clipping each tensor independently would not.

    Returns the total norm BEFORE clipping, exactly like
    torch.nn.utils.clip_grad_norm_ - useful for logging when clipping is
    actually triggering.
    """
    raise NotImplementedError("your code here")
