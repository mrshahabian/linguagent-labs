"""Unit tests for trees.py.

You are encouraged to read these tests, but do not modify them - treat them as
correct. If you want to experiment with a case, copy it into a new script and
make your changes there.

Run with:   python3 tests/test_trees.py
       or:  ./run
"""

import importlib
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# `./run --check` points this at the reference implementation instead.
t = importlib.import_module(os.environ.get("TREES_TARGET", "trees"))


def blobs(n=400, n_features=4, gap=2.0, seed=0):
    """Two well-separated Gaussian blobs: any working learner gets ~100%."""
    rng = np.random.RandomState(seed)
    half = n // 2
    X = np.vstack([rng.randn(half, n_features) + gap,
                   rng.randn(half, n_features) - gap])
    Y = np.array([0] * half + [1] * half)
    order = rng.permutation(len(Y))
    return X[order], Y[order]


def with_nans(X, fraction=0.15, seed=1):
    rng = np.random.RandomState(seed)
    out = X.copy()
    out[rng.rand(*out.shape) < fraction] = np.nan
    return out


def depth_of(tree):
    if isinstance(tree, t.Leaf):
        return 0
    return 1 + max(depth_of(tree.left), depth_of(tree.right))


def accuracy(ensemble, X, Y):
    votes = ensemble.predictions(X)
    n_classes = int(Y.max()) + 1
    counts = np.apply_along_axis(
        lambda col: np.bincount(col, minlength=n_classes), axis=0, arr=votes)
    return float(np.mean(np.argmax(counts, axis=0) == Y))


# ------------------------------------------------------------------ entropy

class TestEntropy(unittest.TestCase):

    def test_uniform_labels_have_zero_entropy(self):
        """A pure set carries no information: entropy is 0 bits."""
        self.assertEqual(t.entropy(np.array([0, 0, 0, 0])), 0.0)

    def test_uniform_labels_of_a_nonzero_class(self):
        """Classes below the maximum that are absent must not poison the sum."""
        self.assertEqual(t.entropy(np.array([1, 1, 1, 1, 1])), 0.0)
        self.assertEqual(t.entropy(np.array([3, 3])), 0.0)

    def test_fifty_fifty_is_one_bit(self):
        self.assertAlmostEqual(t.entropy(np.array([0, 1])), 1.0)
        self.assertAlmostEqual(t.entropy(np.array([0, 0, 1, 1])), 1.0)

    def test_four_classes_evenly_split_is_two_bits(self):
        self.assertAlmostEqual(t.entropy(np.array([0, 1, 2, 3])), 2.0)

    def test_skewed_distribution(self):
        self.assertAlmostEqual(t.entropy(np.array([0, 0, 0, 1])),
                               0.8112781244591328)

    def test_gaps_between_labels_are_ignored(self):
        """Labels 0 and 5 present, 1-4 absent: still a one-bit split."""
        self.assertAlmostEqual(t.entropy(np.array([0, 5])), 1.0)


# --------------------------------------------------------------- score_split

class TestScoreSplit(unittest.TestCase):

    def test_perfect_split_scores_one(self):
        """A split which perfectly separates the labels should score 1."""
        Y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
        is_left = np.array([True] * 4 + [False] * 5)
        self.assertEqual(
            t.score_split(Y, is_left), 1.0,
            "perfect split should score 1: left has all 0s, right has all 1s")

    def test_useless_split_scores_zero(self):
        """A split which divides the labels 50/50 should score zero."""
        Y = np.array([0, 1, 0, 1])
        is_left = np.array([True, True, False, False])
        self.assertEqual(t.score_split(Y, is_left), 0.0)

    def test_subsets_are_weighted_by_their_size(self):
        """Each subset's entropy is weighted by the fraction of examples in it."""
        Y = np.array([0, 0, 0, 1])
        is_left = np.array([True, False, False, False])
        # left is [0] (1 of 4, entropy 0); right is [0, 0, 1] (3 of 4, ~0.918)
        self.assertAlmostEqual(
            t.score_split(Y, is_left), 0.15106563978903303,
            msg="left contributes 1/4 of its entropy, right 3/4 of its")

    def test_score_is_between_zero_and_one(self):
        rng = np.random.RandomState(3)
        for _ in range(200):
            Y = rng.randint(0, 3, 12)
            if (Y[0] == Y).all():
                continue
            is_left = rng.rand(12) < 0.5
            score = t.score_split(Y, is_left)
            self.assertGreaterEqual(score, -1e-12, f"score {score} is negative")
            self.assertLessEqual(score, 1 + 1e-12, f"score {score} exceeds 1")

    def test_better_split_scores_higher(self):
        Y = np.array([0, 0, 0, 1, 1, 1])
        clean = np.array([True, True, True, False, False, False])
        messy = np.array([True, True, False, True, False, False])
        self.assertGreater(t.score_split(Y, clean), t.score_split(Y, messy))


# --------------------------------------------------------- constant features

class TestNonConstantFeatures(unittest.TestCase):

    def test_constant_column_is_excluded(self):
        X = np.array([[1.0, 3.0], [2.0, 3.0], [5.0, 3.0]])
        np.testing.assert_array_equal(t.find_non_constant_features(X),
                                      np.array([0]))

    def test_all_nan_column_is_constant(self):
        """A feature that is all NaN is considered constant (so shouldn't be
        split on)."""
        X = np.array([[np.nan, 3, 4], [np.nan, 3, 5], [np.nan, 3, 6]])
        np.testing.assert_array_equal(t.find_non_constant_features(X),
                                      np.array([2]))

    def test_partly_nan_column_can_still_vary(self):
        X = np.array([[np.nan, 1.0], [2.0, 1.0], [7.0, 1.0]])
        np.testing.assert_array_equal(t.find_non_constant_features(X),
                                      np.array([0]))

    def test_partly_nan_but_otherwise_constant_is_constant(self):
        X = np.array([[np.nan, 1.0], [4.0, 2.0], [4.0, 3.0]])
        np.testing.assert_array_equal(t.find_non_constant_features(X),
                                      np.array([1]))


# ---------------------------------------------------------------- prediction

class TestPrediction(unittest.TestCase):

    def test_leaf_returns_its_label(self):
        self.assertEqual(t.predict_single(np.array([0.0]), t.Leaf(3)), 3)

    def test_strictly_less_than_goes_left(self):
        tree = t.Node(0, 1.0, t.Leaf(0), t.Leaf(1), nans_go_left=False)
        self.assertEqual(t.predict_single(np.array([0.5]), tree), 0)
        self.assertEqual(t.predict_single(np.array([1.0]), tree), 1)
        self.assertEqual(t.predict_single(np.array([1.5]), tree), 1)

    def test_nan_follows_nans_go_left(self):
        left = t.Node(0, 1.0, t.Leaf(7), t.Leaf(8), nans_go_left=True)
        right = t.Node(0, 1.0, t.Leaf(7), t.Leaf(8), nans_go_left=False)
        self.assertEqual(t.predict_single(np.array([np.nan]), left), 7)
        self.assertEqual(t.predict_single(np.array([np.nan]), right), 8)

    def test_nans_go_left_does_not_affect_present_values(self):
        """nans_go_left decides where *missing* values go. A value that is
        present must still be compared against the split."""
        tree = t.Node(0, 1.0, t.Leaf(0), t.Leaf(1), nans_go_left=True)
        self.assertEqual(
            t.predict_single(np.array([5.0]), tree), 1,
            "5.0 is not less than 1.0 and is not NaN, so it goes right")
        self.assertEqual(t.predict_single(np.array([0.5]), tree), 0)

    def test_batch_matches_single_on_clean_data(self):
        X, Y = blobs(seed=3)
        model = t.Ensemble(np.random.RandomState(3), m=4, max_depth=6)
        model.fit(X, Y)
        for tree in model.trees:
            single = np.array([t.predict_single(row, tree) for row in X])
            np.testing.assert_array_equal(t.predict_batch(X, tree), single)

    def test_batch_matches_single_with_missing_values(self):
        X, Y = blobs(seed=4)
        Xn = with_nans(X, seed=4)
        model = t.Ensemble(np.random.RandomState(4), m=4, max_depth=6)
        model.fit(Xn, Y)
        for tree in model.trees:
            single = np.array([t.predict_single(row, tree) for row in Xn])
            np.testing.assert_array_equal(t.predict_batch(Xn, tree), single)


# ------------------------------------------------------------ tree building

class TestBuildTree(unittest.TestCase):

    def test_pure_labels_produce_a_leaf(self):
        X, _ = blobs(n=40, seed=8)
        tree = t.build_tree(X, np.zeros(len(X), dtype=int), 2, 3,
                            np.random.RandomState(8))
        self.assertIsInstance(tree, t.Leaf)
        self.assertEqual(tree.label, 0)

    def test_respects_max_depth(self):
        """The root is at depth 0, so max_depth=d allows Nodes only at depths
        strictly less than d - a tree of depth at most d."""
        X, Y = blobs(seed=7)
        for max_depth in (1, 2, 4):
            tree = t.build_tree(X, Y, 2, 3, np.random.RandomState(7),
                                max_depth=max_depth)
            self.assertLessEqual(
                depth_of(tree), max_depth,
                f"tree is deeper than max_depth={max_depth}")

    def test_max_depth_zero_is_a_leaf(self):
        X, Y = blobs(n=40, seed=9)
        tree = t.build_tree(X, Y, 2, 3, np.random.RandomState(9), max_depth=0)
        self.assertIsInstance(tree, t.Leaf)

    def test_training_terminates_with_missing_values(self):
        X, Y = blobs(n=200, seed=10)
        Xn = with_nans(X, fraction=0.3, seed=10)
        tree = t.build_tree(Xn, Y, 2, 3, np.random.RandomState(10), max_depth=8)
        self.assertIsInstance(tree, (t.Node, t.Leaf))

    def test_labels_with_gaps(self):
        """Labels need not be contiguous. A class that is absent below the
        maximum must not break scoring.

        This is the shape that makes score_split's `ent_before > 0` assertion
        fire with its "uniform labels" message even though the labels are five
        distinct classes - because `nan > 0` is False too.
        """
        X = np.arange(8, dtype=float).reshape(8, 1)
        Y = np.array([3, 0, 1, 4, 1, 4, 1, 3])   # class 2 never appears
        tree = t.build_tree(X, Y, 2, 1, np.random.RandomState(5))
        self.assertIsInstance(tree, (t.Node, t.Leaf))

    def test_split_keeps_the_dtype_of_X(self):
        """Split values must be the same dtype as X, not silently widened."""
        for dtype in (np.float16, np.float32, np.float64):
            X, Y = blobs(n=60, seed=21)
            X = X.astype(dtype)
            tree = t.build_tree(X, Y, 2, 3, np.random.RandomState(21), max_depth=4)
            nodes = [tree]
            seen = 0
            while nodes:
                node = nodes.pop()
                if isinstance(node, t.Leaf):
                    continue
                seen += 1
                self.assertIsInstance(
                    node.split, dtype,
                    f"split value {node.split!r} is {type(node.split).__name__}, "
                    f"but X is {np.dtype(dtype)}")
                nodes += [node.left, node.right]
            self.assertGreater(seen, 0, "expected at least one internal node")

    def test_fuzz_parameters(self):
        """Any values of n_min and k should complete training without crashing."""
        rng = np.random.RandomState(234)
        X = rng.randn(20, 3).astype(np.float32)
        Y = rng.randint(0, 10, (20,))
        for n_min in range(1, 5):
            for k in range(1, 4):
                t.build_tree(X, Y, n_min, k, np.random.RandomState(1))

    def test_fuzz_with_missing_values(self):
        rng = np.random.RandomState(77)
        X = rng.randn(30, 4)
        X[rng.rand(*X.shape) < 0.25] = np.nan
        Y = rng.randint(0, 3, (30,))
        for k in range(1, 5):
            t.build_tree(X, Y, 2, k, np.random.RandomState(2))


# ----------------------------------------------------------------- ensemble

class TestEnsemble(unittest.TestCase):

    def test_predictions_shape(self):
        X, Y = blobs(n=100, seed=20)
        model = t.Ensemble(np.random.RandomState(20), m=7, max_depth=5)
        model.fit(X, Y)
        self.assertEqual(model.predictions(X).shape, (7, len(X)))

    def test_trees_are_not_all_identical(self):
        """The same data is used for each tree, but each tree's splits differ
        due to the RNG."""
        X, Y = blobs(seed=11)
        model = t.Ensemble(np.random.RandomState(11), m=8, max_depth=6)
        model.fit(X, Y)
        rendered = {t.tree_to_string(tree) for tree in model.trees}
        self.assertGreater(
            len(rendered), 1,
            "every tree in the forest came out identical - the ensemble is "
            "no better than a single tree")

    def test_accuracy_on_separable_data(self):
        X, Y = blobs(n=600, seed=12)
        model = t.Ensemble(np.random.RandomState(12), m=15, max_depth=8)
        model.fit(X[:400], Y[:400])
        self.assertGreaterEqual(accuracy(model, X[400:], Y[400:]), 0.95)

    def test_accuracy_with_missing_values(self):
        X, Y = blobs(n=600, seed=13)
        Xn = with_nans(X, fraction=0.2, seed=13)
        model = t.Ensemble(np.random.RandomState(13), m=15, max_depth=8)
        model.fit(Xn[:400], Y[:400])
        self.assertGreaterEqual(accuracy(model, Xn[400:], Y[400:]), 0.90)

    def test_multiclass(self):
        rng = np.random.RandomState(14)
        X = rng.randn(600, 5)
        Y = (X[:, 0] > 0).astype(int) + 2 * (X[:, 1] > 0).astype(int)
        model = t.Ensemble(np.random.RandomState(14), m=20, max_depth=10)
        model.fit(X[:400], Y[:400])
        self.assertGreaterEqual(accuracy(model, X[400:], Y[400:]), 0.85)

    def test_forest_beats_a_single_tree(self):
        """Averaging only helps if the trees actually differ."""
        rng = np.random.RandomState(30)
        X = rng.randn(500, 6)
        Y = (X[:, 0] + X[:, 1] > 0).astype(int)
        one = t.Ensemble(np.random.RandomState(30), m=1, max_depth=4)
        one.fit(X[:300], Y[:300])
        many = t.Ensemble(np.random.RandomState(30), m=25, max_depth=4)
        many.fit(X[:300], Y[:300])
        self.assertGreaterEqual(accuracy(many, X[300:], Y[300:]),
                                accuracy(one, X[300:], Y[300:]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
