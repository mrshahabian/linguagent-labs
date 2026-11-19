"""Reference answers for 29 - optimizers and learning-rate schedules."""

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
    new_velocity = momentum * velocity + grad
    new_param = param - lr * new_velocity
    return new_param, new_velocity


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
    new_m = beta1 * m + (1 - beta1) * grad
    new_v = beta2 * v + (1 - beta2) * grad * grad
    m_hat = new_m / (1 - beta1 ** t)
    v_hat = new_v / (1 - beta2 ** t)
    new_param = param - lr * m_hat / (v_hat.sqrt() + eps)
    return new_param, new_m, new_v


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
    new_m = beta1 * m + (1 - beta1) * grad
    new_v = beta2 * v + (1 - beta2) * grad * grad
    m_hat = new_m / (1 - beta1 ** t)
    v_hat = new_v / (1 - beta2 ** t)
    new_param = param - lr * (m_hat / (v_hat.sqrt() + eps) + weight_decay * param)
    return new_param, new_m, new_v


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
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


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
    total_norm = torch.sqrt(sum((g ** 2).sum() for g in grads))
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for g in grads:
            g.mul_(clip_coef)
    return total_norm
