"""12 - broadcasting and vectorisation

Implement each function so the tests in test_exercises.py pass. Read
README.md first; every exercise below is covered there.

    ../run 12            test your work
    ../run 12 --check    watch the reference solutions pass

The docstrings are the specification. Standard library only.
"""

import numpy as np


def broadcast_shape(a, b):
    """Return the shape a and b broadcast to, or None if incompatible.

    The rule: align the shapes from the RIGHT, pad the shorter with 1s, and
    for each axis the sizes must be equal or one of them must be 1. The
    result takes the larger of the two.

    broadcast_shape((3, 1), (1, 4)) -> (3, 4)
    broadcast_shape((3,), (4,))     -> None
    """
    raise NotImplementedError("your code here")


def pairwise_distances(a, b):
    """Euclidean distance between every row of a and every row of b.

    a: (n, d), b: (m, d) -> (n, m). No Python loops.

    The trick is inserting a length-1 axis so the two arrays broadcast
    against each other: (n, 1, d) against (m, d) gives (n, m, d).
    """
    raise NotImplementedError("your code here")


def softmax(x, axis=-1):
    """Numerically stable softmax along `axis`.

    Subtracting the max before exponentiating changes nothing
    mathematically - exp(a - c) / sum(exp(a - c)) equals exp(a) / sum(exp(a))
    - but it stops exp() overflowing to inf for large inputs.

    keepdims=True is what lets the result broadcast back against x.
    """
    raise NotImplementedError("your code here")


def one_hot(labels, num_classes):
    """(n,) integer labels -> (n, num_classes) float32 one-hot matrix.

    Comparing a column vector of labels against a row of class indices
    broadcasts to the full matrix in one step.
    """
    raise NotImplementedError("your code here")


def masked_mean(x, mask, axis=None):
    """Mean of x over positions where mask is True.

    Positions with no unmasked values produce NaN rather than raising.
    """
    raise NotImplementedError("your code here")


def iou(boxes_a, boxes_b):
    """Intersection-over-union between every pair of boxes.

    Boxes are (x1, y1, x2, y2) with x1 < x2 and y1 < y2.
    a: (n, 4), b: (m, 4) -> (n, m).

    The staple of every detection pipeline, and a common interview warm-up
    for vision roles. Clipping the overlap at 0 is what handles boxes that
    do not intersect at all.
    """
    raise NotImplementedError("your code here")


def batched_dot(a, b):
    """Row-wise dot product of two (n, d) arrays -> (n,).

    Equivalent to np.einsum("nd,nd->n", a, b). Do not build the (n, n)
    matrix and take its diagonal - that is O(n^2) memory for an O(n) answer.
    """
    raise NotImplementedError("your code here")


def top_k(scores, k):
    """Indices of the k largest values, highest first, ties by lower index.

    argpartition finds the k largest in O(n) without sorting the rest, then
    only those k are sorted.
    """
    raise NotImplementedError("your code here")


def normalise_rows(x, eps=1e-12):
    """Scale each row to unit L2 norm; all-zero rows stay all-zero."""
    raise NotImplementedError("your code here")
