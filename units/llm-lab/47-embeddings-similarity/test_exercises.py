"""Tests for 47 — Embeddings and similarity search."""

import importlib
import os
import unittest
import numpy as np

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestL2Normalize(unittest.TestCase):
    def test_1d_vector_normal(self):
        v = np.array([3.0, 4.0])
        normalized = m.l2_normalize(v)
        np.testing.assert_allclose(normalized, np.array([0.6, 0.8]))
        # Ensure original was not modified
        np.testing.assert_allclose(v, np.array([3.0, 4.0]))

    def test_1d_vector_zero(self):
        v = np.array([0.0, 0.0])
        normalized = m.l2_normalize(v)
        np.testing.assert_allclose(normalized, np.array([0.0, 0.0]))

    def test_2d_matrix_normal(self):
        mat = np.array([[3.0, 4.0], [0.0, 5.0]])
        normalized = m.l2_normalize(mat)
        expected = np.array([[0.6, 0.8], [0.0, 1.0]])
        np.testing.assert_allclose(normalized, expected)

    def test_2d_matrix_with_zero_row(self):
        mat = np.array([[3.0, 4.0], [0.0, 0.0]])
        normalized = m.l2_normalize(mat)
        expected = np.array([[0.6, 0.8], [0.0, 0.0]])
        np.testing.assert_allclose(normalized, expected)

    def test_epsilon_threshold(self):
        # epsilon is 1e-12 by default
        v_below = np.array([1e-13, 0.0])
        v_above = np.array([1e-11, 0.0])
        np.testing.assert_allclose(m.l2_normalize(v_below), np.array([0.0, 0.0]))
        np.testing.assert_allclose(m.l2_normalize(v_above), np.array([1.0, 0.0]))

    def test_invalid_dimensions(self):
        v = np.zeros((2, 2, 2))
        with self.assertRaises(ValueError):
            m.l2_normalize(v)


class TestCosineSimilarity(unittest.TestCase):
    def test_identical_vectors(self):
        X = np.array([[1.0, 2.0]])
        Y = np.array([[1.0, 2.0]])
        sim = m.cosine_similarity(X, Y)
        np.testing.assert_allclose(sim, np.array([[1.0]]))

    def test_orthogonal_vectors(self):
        X = np.array([[1.0, 0.0]])
        Y = np.array([[0.0, 1.0]])
        sim = m.cosine_similarity(X, Y)
        np.testing.assert_allclose(sim, np.array([[0.0]]), atol=1e-7)

    def test_batch_similarity(self):
        X = np.array([[1.0, 0.0], [0.0, 1.0]])
        Y = np.array([[1.0, 0.0], [1.0, 1.0]])
        sim = m.cosine_similarity(X, Y)
        expected = np.array([[1.0, 1.0 / np.sqrt(2)], [0.0, 1.0 / np.sqrt(2)]])
        np.testing.assert_allclose(sim, expected)

    def test_zero_vectors_handled_safely(self):
        X = np.array([[0.0, 0.0]])
        Y = np.array([[1.0, 0.0]])
        sim = m.cosine_similarity(X, Y)
        np.testing.assert_allclose(sim, np.array([[0.0]]))

    def test_negative_similarity(self):
        X = np.array([[1.0, 1.0]])
        Y = np.array([[-1.0, -1.0]])
        sim = m.cosine_similarity(X, Y)
        np.testing.assert_allclose(sim, np.array([[-1.0]]))


class TestTopK(unittest.TestCase):
    def test_basic_top_k(self):
        scores = np.array([10.0, 50.0, 30.0, 40.0])
        indices, values = m.top_k(scores, k=2)
        np.testing.assert_array_equal(indices, np.array([1, 3]))
        np.testing.assert_allclose(values, np.array([50.0, 40.0]))

    def test_stable_tie_breaking(self):
        # Scores have a tie at 30.0. Index 1 should come before index 3.
        scores = np.array([10.0, 30.0, 20.0, 30.0])
        indices, values = m.top_k(scores, k=3)
        np.testing.assert_array_equal(indices, np.array([1, 3, 2]))
        np.testing.assert_allclose(values, np.array([30.0, 30.0, 20.0]))

    def test_k_larger_than_array(self):
        scores = np.array([10.0, 20.0])
        indices, values = m.top_k(scores, k=5)
        np.testing.assert_array_equal(indices, np.array([1, 0]))
        np.testing.assert_allclose(values, np.array([20.0, 10.0]))

    def test_negative_scores(self):
        scores = np.array([-10.0, -50.0, -30.0])
        indices, values = m.top_k(scores, k=2)
        np.testing.assert_array_equal(indices, np.array([0, 2]))
        np.testing.assert_allclose(values, np.array([-10.0, -30.0]))

    def test_single_element(self):
        scores = np.array([42.0])
        indices, values = m.top_k(scores, k=1)
        np.testing.assert_array_equal(indices, np.array([0]))
        np.testing.assert_allclose(values, np.array([42.0]))

    def test_all_ties(self):
        scores = np.array([5.0, 5.0, 5.0, 5.0])
        indices, values = m.top_k(scores, k=2)
        np.testing.assert_array_equal(indices, np.array([0, 1]))
        np.testing.assert_allclose(values, np.array([5.0, 5.0]))


class TestTfidfEmbedder(unittest.TestCase):
    def test_fit_vocabulary_and_idf(self):
        corpus = [
            "the cat sat",
            "the dog barked"
        ]
        embedder = m.TfidfEmbedder()
        embedder.fit(corpus)
        
        # Vocabulary should be alphabetically sorted
        expected_vocab = {"barked": 0, "cat": 1, "dog": 2, "sat": 3, "the": 4}
        self.assertEqual(embedder.vocabulary_, expected_vocab)
        
        # N = 2
        # DF("the") = 2 -> IDF = log(3/3) + 1 = 1.0
        # DF("cat") = 1 -> IDF = log(3/2) + 1 = 0.405465 + 1 = 1.405465
        expected_idf = np.array([
            np.log(3.0 / 2.0) + 1.0,  # barked
            np.log(3.0 / 2.0) + 1.0,  # cat
            np.log(3.0 / 2.0) + 1.0,  # dog
            np.log(3.0 / 2.0) + 1.0,  # sat
            np.log(3.0 / 3.0) + 1.0,  # the
        ])
        np.testing.assert_allclose(embedder.idf_, expected_idf)

    def test_transform_produces_l2_normalized_vectors(self):
        corpus = [
            "the cat sat",
            "the dog barked"
        ]
        embedder = m.TfidfEmbedder()
        embedder.fit(corpus)
        vectors = embedder.transform(["cat the", "unknown word"])
        
        # First document has "cat" and "the"
        # Second document has only unknown words (should be zero vector)
        self.assertEqual(vectors.shape, (2, 5))
        
        # Row 0 should be L2 normalized
        row0_norm = np.linalg.norm(vectors[0])
        self.assertAlmostEqual(row0_norm, 1.0)
        
        # Row 1 should be all zeros
        np.testing.assert_allclose(vectors[1], np.zeros(5))

    def test_empty_corpus(self):
        embedder = m.TfidfEmbedder()
        embedder.fit([])
        self.assertEqual(embedder.vocabulary_, {})
        self.assertEqual(len(embedder.idf_), 0)
        
        vectors = embedder.transform(["hello"])
        self.assertEqual(vectors.shape, (1, 0))

    def test_transform_oov_only(self):
        corpus = ["apple banana"]
        embedder = m.TfidfEmbedder().fit(corpus)
        vectors = embedder.transform(["cherry date"])
        np.testing.assert_allclose(vectors, np.zeros((1, 2)))


class TestMeanPool(unittest.TestCase):
    def test_basic_pooling(self):
        # Batch size 2, Sequence length 3, Dimension 2
        embeddings = np.array([
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]
        ])
        # First sentence has all 3 tokens active
        # Second sentence has only first 2 tokens active
        mask = np.array([
            [1, 1, 1],
            [1, 1, 0]
        ])
        pooled = m.mean_pool(embeddings, mask)
        
        expected_row0 = np.array([3.0, 4.0])  # mean of [1,2], [3,4], [5,6]
        expected_row1 = np.array([20.0, 30.0]) # mean of [10,20], [30,40]
        
        np.testing.assert_allclose(pooled[0], expected_row0)
        np.testing.assert_allclose(pooled[1], expected_row1)

    def test_fully_padded_sentence(self):
        embeddings = np.array([
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]]
        ])
        # Second sentence is fully padded
        mask = np.array([
            [1, 1],
            [0, 0]
        ])
        pooled = m.mean_pool(embeddings, mask)
        
        np.testing.assert_allclose(pooled[0], np.array([2.0, 3.0]))
        np.testing.assert_allclose(pooled[1], np.array([0.0, 0.0]))

    def test_single_token_sequences(self):
        embeddings = np.array([
            [[4.0, 5.0]],
            [[10.0, 11.0]]
        ])
        mask = np.array([
            [1],
            [0]
        ])
        pooled = m.mean_pool(embeddings, mask)
        np.testing.assert_allclose(pooled[0], np.array([4.0, 5.0]))
        np.testing.assert_allclose(pooled[1], np.array([0.0, 0.0]))

    def test_all_zeros_mask_and_embeddings(self):
        embeddings = np.zeros((3, 4, 5))
        mask = np.zeros((3, 4))
        pooled = m.mean_pool(embeddings, mask)
        np.testing.assert_allclose(pooled, np.zeros((3, 5)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
