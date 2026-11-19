"""Reference answers for 12 - broadcasting and vectorisation."""

import numpy as np


def broadcast_shape(a, b):
    """Return the shape a and b broadcast to, or None if incompatible.

    The rule: align the shapes from the RIGHT, pad the shorter with 1s, and
    for each axis the sizes must be equal or one of them must be 1. The
    result takes the larger of the two.

    broadcast_shape((3, 1), (1, 4)) -> (3, 4)
    broadcast_shape((3,), (4,))     -> None
    """
    width = max(len(a), len(b))
    left = (1,) * (width - len(a)) + tuple(a)
    right = (1,) * (width - len(b)) + tuple(b)
    out = []
    for x, y in zip(left, right):
        if x != y and x != 1 and y != 1:
            return None
        out.append(max(x, y))
    return tuple(out)


def pairwise_distances(a, b):
    """Euclidean distance between every row of a and every row of b.

    a: (n, d), b: (m, d) -> (n, m). No Python loops.

    The trick is inserting a length-1 axis so the two arrays broadcast
    against each other: (n, 1, d) against (m, d) gives (n, m, d).
    """
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def softmax(x, axis=-1):
    """Numerically stable softmax along `axis`.

    Subtracting the max before exponentiating changes nothing
    mathematically - exp(a - c) / sum(exp(a - c)) equals exp(a) / sum(exp(a))
    - but it stops exp() overflowing to inf for large inputs.

    keepdims=True is what lets the result broadcast back against x.
    """
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / np.sum(exponentiated, axis=axis, keepdims=True)


def one_hot(labels, num_classes):
    """(n,) integer labels -> (n, num_classes) float32 one-hot matrix.

    Comparing a column vector of labels against a row of class indices
    broadcasts to the full matrix in one step.
    """
    return (labels[:, None] == np.arange(num_classes)[None, :]).astype(np.float32)


def masked_mean(x, mask, axis=None):
    """Mean of x over positions where mask is True.

    Positions with no unmasked values produce NaN rather than raising.
    """
    counts = np.sum(mask, axis=axis)
    totals = np.sum(np.where(mask, x, 0.0), axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)


def iou(boxes_a, boxes_b):
    """Intersection-over-union between every pair of boxes.

    Boxes are (x1, y1, x2, y2) with x1 < x2 and y1 < y2.
    a: (n, 4), b: (m, 4) -> (n, m).

    The staple of every detection pipeline, and a common interview warm-up
    for vision roles. Clipping the overlap at 0 is what handles boxes that
    do not intersect at all.
    """
    ax1, ay1, ax2, ay2 = (boxes_a[:, i][:, None] for i in range(4))
    bx1, by1, bx2, by2 = (boxes_b[:, i][None, :] for i in range(4))

    inter_w = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    inter_h = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    intersection = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection

    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, intersection / np.maximum(union, 1e-12), 0.0)


def batched_dot(a, b):
    """Row-wise dot product of two (n, d) arrays -> (n,).

    Equivalent to np.einsum("nd,nd->n", a, b). Do not build the (n, n)
    matrix and take its diagonal - that is O(n^2) memory for an O(n) answer.
    """
    return np.sum(a * b, axis=-1)


def top_k(scores, k):
    """Indices of the k largest values, highest first, ties by lower index.

    argpartition finds the k largest in O(n) without sorting the rest, then
    only those k are sorted.
    """
    k = min(k, scores.shape[-1])
    if k == 0:
        return np.empty(0, dtype=int)
    candidates = np.argpartition(-scores, k - 1)[:k]
    # lexsort takes the PRIMARY key last: sort by descending score, and break
    # ties towards the lower index.
    return candidates[np.lexsort((candidates, -scores[candidates]))]


def normalise_rows(x, eps=1e-12):
    """Scale each row to unit L2 norm; all-zero rows stay all-zero."""
    norms = np.sqrt(np.sum(x ** 2, axis=-1, keepdims=True))
    return x / np.maximum(norms, eps)
