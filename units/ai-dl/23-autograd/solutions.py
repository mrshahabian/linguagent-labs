"""Reference answers for 23 - autograd."""

import torch


def grad_of(f, x):
    """Return df/dx at x, without leaving anything on x.grad.

    `torch.autograd.grad` computes gradients functionally: it returns them
    instead of accumulating into `.grad`. That is what you want inside a
    metric, a penalty term, or anything that must not disturb the training
    step happening around it.
    """
    x = x.detach().requires_grad_(True)
    return torch.autograd.grad(f(x).sum(), x)[0]


def accumulate_twice(x):
    """Backward through x**2 twice and return x.grad.

    The point: `.grad` ACCUMULATES, it is not overwritten. Two backward
    passes give 2*2x, not 2x. That is a feature - it is how gradient
    accumulation over micro-batches works - and it is why every training
    loop must call `zero_grad()`.

    The second backward needs `retain_graph=True`, because the graph is
    freed as soon as it has been traversed once.
    """
    x = x.detach().requires_grad_(True)
    (x ** 2).sum().backward(retain_graph=True)
    (x ** 2).sum().backward()
    return x.grad


def frozen_forward(model, x):
    """Run the model without building a graph, and without changing it.

    `torch.no_grad()` stops the graph being recorded at all: no activations
    are kept alive, so it is the memory win at inference time. It is NOT
    the same as `model.eval()`, which switches dropout and batch-norm to
    inference behaviour but still records gradients. Real inference needs
    both, and confusing the two is a very common interview miss.
    """
    with torch.no_grad():
        return model(x)


def detach_vs_clone(x):
    """Return (detached, cloned) for a tensor that requires grad.

    - `detach()` shares storage, drops the graph. Writing to it silently
      corrupts x, and x's gradient with it.
    - `clone()` copies storage, KEEPS the graph, so gradients still flow
      back to x through the copy.

    "Take the value, not the gradient" is `x.detach()`; "take a separate
    tensor I can still differentiate" is `x.clone()`. `detach().clone()`
    is the safe both-of-those.
    """
    return x.detach(), x.clone()


def clip_grad_norm(parameters, max_norm):
    """Scale gradients in place so their global L2 norm is at most max_norm.

    Returns the norm BEFORE clipping - which is the number worth logging,
    because a spike there is the earliest warning of a diverging run.

    The norm is GLOBAL: every parameter is treated as one long vector, and
    all of them are scaled by the same factor, so the update direction is
    preserved and only its length changes. Clipping each tensor separately
    changes the direction, which is a different (worse) algorithm.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    total = torch.sqrt(sum(g.pow(2).sum() for g in grads))
    if total > max_norm:
        scale = max_norm / (total + 1e-6)
        for g in grads:
            g.mul_(scale)
    return total


class Square(torch.autograd.Function):
    """x**2 with a hand-written backward.

    The shape of every custom kernel: `forward` computes the value and
    stashes what backward needs; `backward` receives dL/dy and returns
    dL/dx. Here dy/dx = 2x, so dL/dx = grad_output * 2x.

    `save_for_backward` (rather than `self.x = x`) is what lets autograd
    detect the tensor being modified in place before backward runs.
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x ** 2

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * 2 * x


class StraightThroughRound(torch.autograd.Function):
    """Round in the forward pass, pretend it was the identity in backward.

    round() has a derivative of 0 almost everywhere, so a quantised network
    would receive no gradient at all. The straight-through estimator lies:
    it passes the incoming gradient through unchanged. This is the trick
    behind quantisation-aware training and the VQ-VAE codebook.
    """

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def numeric_grad(f, x, h=1e-4):
    """Central-difference gradient of a scalar function, for gradient checks.

    Central difference has error O(h^2); the one-sided (f(x+h) - f(x)) / h
    is only O(h), and its error swamps the thing you are trying to verify.
    Use float64: at float32 the rounding noise is larger than the signal,
    which is why `torch.autograd.gradcheck` insists on doubles.
    """
    x = x.detach().to(torch.float64)
    grad = torch.zeros_like(x)
    for index in range(x.numel()):
        step = torch.zeros_like(x)
        step.view(-1)[index] = h
        grad.view(-1)[index] = (f(x + step) - f(x - step)) / (2 * h)
    return grad


def gradient_penalty(discriminator, x):
    """Mean squared L2 norm of d(output)/d(input) - a gradient penalty.

    The move worth learning is `create_graph=True`: it makes the gradient
    itself part of the graph, so the penalty can be differentiated again
    when you call backward on it. Without it the penalty is a constant and
    contributes nothing. This is WGAN-GP, and the same mechanism behind any
    second-order or meta-learning objective.
    """
    x = x.detach().requires_grad_(True)
    output = discriminator(x)
    gradients = torch.autograd.grad(
        output.sum(), x, create_graph=True)[0]
    return gradients.flatten(1).pow(2).sum(1).mean()
