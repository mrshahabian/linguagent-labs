"""Extremely Randomized Trees - the correct implementation.

This is the reference. Do not read it until you have finished debugging
trees.py, or you will spend the exercise recognising rather than diagnosing.

Extremely Randomized Trees (Geurts, Ernst & Wehenkel, 2006) is a supervised
learning algorithm:
  - the input is a fixed length vector of continuous features
  - the output is a class label in the range [0, n_classes)

What makes it "extremely" randomised: an ordinary decision tree searches every
candidate threshold on a feature for the one that splits the labels best. This
one draws a *single* threshold per candidate feature, uniformly at random
between that feature's minimum and maximum within the node, and keeps only the
best of those few random candidates.

That sounds much worse and mostly isn't, for two reasons. Each tree is far
cheaper to build, so you can afford many more of them; and the extra randomness
decorrelates the trees, which is exactly what makes an ensemble average well.
The variance you add per tree is more than paid back by the variance the
ensemble removes.
"""

from __future__ import annotations

import numpy as np
from numpy import ndarray
from numpy import random
from typing import List, NamedTuple, Optional, Union


class Leaf(NamedTuple):
    """Leaf (terminal) node of a decision tree. Holds a class label."""

    label: int


class Node(NamedTuple):
    """Internal node of a decision tree.

    It's a binary tree, so internal nodes have exactly two children.
    """

    feature: int
    split: np.floating
    left: "Tree"
    right: "Tree"
    nans_go_left: bool


Tree = Union[Node, Leaf]


# ------------------------------------------------------------------ display

def tree_to_string(tree: Tree, max_depth: int = 10) -> str:
    """Format a string representation of a tree using numpy's default formatting."""
    lines: List[str] = []

    def build_lines(node: Tree, indent: str = "", prefix: str = "", depth: int = 0):
        if depth > max_depth:
            lines.append(f"{prefix}...")
            return
        if isinstance(node, Leaf):
            lines.append(f"{prefix}Leaf({node.label})")
            return
        lines.append(f"{prefix}Node(x[{node.feature}] < {node.split})")
        build_lines(node.left, indent + "|  ", indent + "|--", depth + 1)
        build_lines(node.right, indent + "   ", indent + "L--", depth + 1)

    build_lines(tree)
    return "\n".join(lines)


def print_tree(tree: Tree, max_depth: int = 10) -> None:
    """Pretty-print a tree"""
    print(tree_to_string(tree, max_depth=max_depth))


# --------------------------------------------------------------- prediction

def predict_single(x: ndarray, tree: Tree) -> int:
    """Return the predicted label for the input example x.

    If tree is a Leaf, we're done and can just return its class label
    as our prediction.
    Otherwise:
      - if the value of the chosen feature is strictly less than the
        split value, or if the value is NaN and nans_go_left is True,
        then we visit the left child.
      - otherwise, we visit the right child.

    Arguments:
      x: input example with shape (num_features,)
      tree: either a Node with two children, or a Leaf holding a class label.
    """
    while not isinstance(tree, Leaf):
        val = x[tree.feature]
        # `val < split` is already False for NaN, so the nans_go_left branch
        # has to be gated on np.isnan - otherwise it fires for every row.
        if np.isnan(val):
            go_left = tree.nans_go_left
        else:
            go_left = val < tree.split
        tree = tree.left if go_left else tree.right
    return int(tree.label)


def predict_batch(X: ndarray, tree: Tree) -> ndarray:
    """Predict labels for a batch of examples with a single tree.

    Equivalent to calling predict_single on each row of X, but vectorized:
    instead of routing samples one at a time, all samples are pushed through
    the tree together by tracking which samples reach each node.

    The routing rule matches predict_single:
      - a sample visits the left child when its feature value is strictly
        less than the split value, or when the value is NaN and
        nans_go_left is True
      - otherwise it visits the right child.

    Arguments:
      X: input examples with shape (num_examples, num_features)
      tree: either a Node with two children, or a Leaf holding a class label.
    Return: predicted labels with shape (num_examples,), integer dtype
    """
    predictions = np.empty(len(X), dtype=int)
    stack = [(tree, np.arange(len(X)))]
    while stack:
        node, indices = stack.pop()
        if isinstance(node, Leaf):
            predictions[indices] = node.label
        else:
            values = X[indices, node.feature]
            mask = values < node.split
            mask[np.isnan(values)] = node.nans_go_left
            left_indices = indices[mask]
            right_indices = indices[~mask]
            if len(left_indices) > 0:
                stack.append((node.left, left_indices))
            if len(right_indices) > 0:
                stack.append((node.right, right_indices))
    return predictions


# ------------------------------------------------------------------ scoring

def majority(Y: ndarray) -> int:
    """Return the class label that appears the most frequently in Y.

    In case of a tie, return the smallest class label.
    """
    if len(Y) == 0:
        return 0
    return int(np.argmax(np.bincount(Y)))


def find_non_constant_features(X: ndarray) -> ndarray:
    """Return the indices of columns of X that are not constant.

    A column whose values are all identical cannot separate anything, and
    neither can a column that is entirely NaN.

    X: shape (num_examples, num_features), dtype float
    Return: indices of the non-constant columns, integer dtype
    """
    all_nan = np.all(np.isnan(X), axis=0)
    # nanmin/nanmax of an all-NaN column is NaN (and warns), and NaN != NaN
    # is True - so those columns have to be excluded explicitly.
    with np.errstate(all="ignore"):
        feat_min = np.nanmin(np.where(all_nan, 0.0, X), axis=0)
        feat_max = np.nanmax(np.where(all_nan, 0.0, X), axis=0)
    return np.where((feat_min != feat_max) & ~all_nan)[0]


def entropy(Y: ndarray) -> float:
    """Return the entropy of the label array in bits.

    Given probabilities p of each label present in Y, entropy is
    defined as:

      -sum_i (p_i * log2(p_i))

    Y: shape (num_examples,), integer dtype
    """
    if len(Y) == 0:
        return float("NaN")
    counts = np.bincount(Y)
    probs = counts / len(Y)
    # Only labels *present* in Y contribute. A label with p = 0 would give
    # 0 * log2(0) = 0 * -inf = NaN, and poison the whole sum.
    probs = probs[probs > 0]
    bits = -np.sum(probs * np.log2(probs))
    return float(bits)


def score_split(Y: ndarray, is_left: ndarray) -> float:
    """Score the split between 0 and 1, where 1 is the best possible.

    The score is the normalized difference between entropy before the split,
    and the weighted entropies after the split, where each subset's entropy
    is weighted by the fraction of the examples that fall in that subset.

    Arguments:
      - Y: shape (num_examples,) - integer dtype
      - is_left: shape (num_examples,), bool dtype
        if is_left[i] is True, then Y[i] belongs to the left part of the split
    """
    assert Y.shape == is_left.shape, "is_left should be a mask into Y"
    ent_before = entropy(Y)
    assert ent_before > 0, "score_split should not be called with uniform labels"
    ent_left = entropy(Y[is_left])
    ent_right = entropy(Y[~is_left])

    n_left = int(np.count_nonzero(is_left))
    frac_left = n_left / len(Y)
    weighted_after = frac_left * ent_left + (1.0 - frac_left) * ent_right

    info_gain = ent_before - weighted_after
    return float(info_gain / ent_before)


# ----------------------------------------------------------------- training

class ScoredSplit(NamedTuple):
    """A potential split of the training data; used in build_tree."""

    feature: int
    split: np.floating
    is_left: ndarray
    score: float
    nans_go_left: bool


def build_tree(
    X: ndarray,
    Y: ndarray,
    n_min: int,
    k: int,
    rng: random.RandomState,
    max_depth: Optional[int] = None,
    depth: int = 0,
) -> Tree:
    """Fit a decision tree to the provided training data.

    The training process is as follows:

    Return a Leaf containing the most common class label if:
      - the depth limit max_depth has been reached
      - or there are fewer than n_min training examples
      - or all the labels are identical
      - or there are no non-constant features left to split on
        (for example, when all the examples are identical)

    Otherwise, perform the following:
      - sample non-constant columns without replacement as candidates for
        splitting
      - for each considered split column
        - choose a random split value between the min and max value
        - partition the data into a "left" subset containing inputs where the
          feature is less than the split value, and a "right" subset
          containing the other inputs.
        - if any NaN values exist, check both a partition with all NaN on the
          left, and with all NaN on the right.
        - if either subset is empty, use any split value such that no subset
          is empty

    Arguments:
      - X: input data with shape (num_examples, num_features), dtype float
      - Y: gold labels with shape (num_examples,), dtype int
      - n_min: stop building and generate a Leaf when there are fewer than
        this many input examples remaining
      - k: maximum number of split columns to consider; if fewer exist then
        use all available
      - rng: used to randomly choose splits
      - max_depth: maximum depth of any node in the tree, or None for no
        limit. The root is at depth 0, so a tree built with max_depth=d
        contains Nodes only at depths strictly less than d, and Leaves at
        depths no greater than d.
      - depth: the depth of the node currently being built; leave as 0 when
        building a full tree.
    """
    assert len(X) > 0, "Caller should not pass empty X."
    assert len(X) == len(Y), "num_examples should match labels"
    assert k > 0, "must try at least one split"
    assert n_min > 0, "n_min must be positive"

    # Nodes live at depths strictly less than max_depth, so once we reach
    # max_depth this has to be a Leaf.
    if max_depth is not None and depth >= max_depth:
        return Leaf(majority(Y))
    if len(X) < n_min or (Y[0] == Y).all():
        return Leaf(majority(Y))

    features = find_non_constant_features(X)
    if len(features) == 0:
        return Leaf(majority(Y))

    split_features = rng.choice(features, min(k, len(features)), replace=False)
    scored: List[ScoredSplit] = []
    for feature in split_features:
        values = X[:, feature]
        nan_options = (False, True) if np.isnan(values).any() else (False,)
        for nans_go_left in nan_options:
            vmin, vmax = np.nanmin(values), np.nanmax(values)
            # rng.uniform always returns a float64, whatever vmin/vmax are, so
            # the draw has to be cast back to X's dtype - the nextafter
            # fallback below already does this.
            split = X.dtype.type(rng.uniform(vmin, vmax))
            is_left = values < split
            is_left[np.isnan(values)] = nans_go_left
            if is_left.all() or (~is_left).all():
                # Fall back to the smallest threshold that still separates:
                # only the minimum value goes left.
                split = np.nextafter(vmin, vmax, dtype=X.dtype)
                is_left = values < split
                is_left[np.isnan(values)] = nans_go_left
            score = score_split(Y, is_left)
            scored.append(ScoredSplit(feature, split, is_left, score, nans_go_left))

    feature, split, is_left, score, nans_go_left = max(scored, key=lambda s: s.score)
    left = build_tree(X[is_left], Y[is_left], n_min, k, rng, max_depth, depth + 1)
    right = build_tree(X[~is_left], Y[~is_left], n_min, k, rng, max_depth, depth + 1)
    return Node(feature, split, left, right, nans_go_left)


class Ensemble:
    """A forest of decision trees.

    rng: used to randomly choose splits
    m: number of decision trees in the forest
    n_min: generate a Leaf when there are fewer than this many input examples
      remaining
    k: maximum number of split columns to consider when building
    max_depth: maximum depth of any node in each tree, or None for no limit
    """

    rng: random.RandomState
    m: int
    n_min: int
    k: int
    max_depth: Optional[int]
    trees: List[Tree]

    def __init__(
        self,
        rng: random.RandomState,
        m: int,
        n_min: int = 5,
        k: int = 3,
        max_depth: Optional[int] = None,
    ):
        assert m > 0, "ensemble must contain at least 1 tree"
        self.rng = rng
        self.m = m
        self.n_min = n_min
        self.k = k
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X: ndarray, Y: ndarray) -> None:
        """Fit each individual tree in the forest.

        The same data is used for each tree, but each tree's splits will
        differ due to the RNG.

        X: features with shape (num_examples, num_features), dtype float
        Y: labels with shape (num_examples,), dtype int
        """
        # A comprehension, not [expr] * m: the latter evaluates build_tree
        # once and stores m references to the same tree.
        self.trees = [
            build_tree(X, Y, self.n_min, self.k, self.rng, self.max_depth)
            for _ in range(self.m)
        ]

    def predictions(self, X: ndarray) -> ndarray:
        """Run each tree on each data point.

        X: shape (num_examples, num_features)
        Return: shape (num_trees, num_examples)
        """
        y_preds = np.empty((self.m, len(X)), dtype=int)
        for t, tree in enumerate(self.trees):
            y_preds[t] = predict_batch(X, tree)
        return y_preds

    def predict(self, X: ndarray) -> ndarray:
        """Majority vote across the forest, one label per row of X."""
        votes = self.predictions(X)
        n_classes = int(votes.max()) + 1
        counts = np.apply_along_axis(
            lambda column: np.bincount(column, minlength=n_classes),
            axis=0, arr=votes,
        )
        return np.argmax(counts, axis=0)


if __name__ == "__main__":
    print("This module does nothing when run - start with test_trees.py")
