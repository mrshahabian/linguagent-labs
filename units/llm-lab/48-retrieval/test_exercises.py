"""Tests for 48 — Retrieval: chunking, BM25 and hybrid search."""

import importlib
import os
import unittest

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestChunkText(unittest.TestCase):
    def test_basic_chunking(self):
        text = "The quick brown fox jumps over the lazy dog."
        # Words: ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog."]
        # chunk_size = 3, overlap = 1 -> step = 2
        # Chunk 1: words 0..3 -> "The quick brown"
        # Chunk 2: words 2..5 -> "brown fox jumps"
        # Chunk 3: words 4..7 -> "jumps over the"
        # Chunk 4: words 6..9 -> "the lazy dog."
        chunks = m.chunk_text(text, chunk_size=3, overlap=1)
        self.assertEqual(len(chunks), 4)
        self.assertEqual(chunks[0]["text"], "The quick brown")
        self.assertEqual(chunks[0]["start"], 0)
        self.assertEqual(chunks[0]["end"], 15)

        self.assertEqual(chunks[1]["text"], "brown fox jumps")
        self.assertEqual(chunks[1]["start"], 10)
        self.assertEqual(chunks[1]["end"], 25)

    def test_chunking_preserves_spacing(self):
        text = "Hello,   world!\nThis is\tcool."
        # Words: ["Hello,", "world!", "This", "is", "cool."]
        chunks = m.chunk_text(text, chunk_size=3, overlap=1)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["text"], "Hello,   world!\nThis")
        self.assertEqual(chunks[1]["text"], "This is\tcool.")

    def test_empty_text(self):
        self.assertEqual(m.chunk_text("", 3, 1), [])
        self.assertEqual(m.chunk_text("   \n\t  ", 3, 1), [])

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            m.chunk_text("Some text", 0, 1)
        with self.assertRaises(ValueError):
            m.chunk_text("Some text", 3, -1)
        with self.assertRaises(ValueError):
            m.chunk_text("Some text", 3, 3)
        with self.assertRaises(ValueError):
            m.chunk_text("Some text", 3, 4)

    def test_chunk_size_larger_than_text(self):
        text = "Short text"
        chunks = m.chunk_text(text, chunk_size=5, overlap=2)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["text"], "Short text")
        self.assertEqual(chunks[0]["start"], 0)
        self.assertEqual(chunks[0]["end"], 10)

    def test_chunk_size_one_no_overlap(self):
        text = "A B C"
        chunks = m.chunk_text(text, chunk_size=1, overlap=0)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0]["text"], "A")
        self.assertEqual(chunks[1]["text"], "B")
        self.assertEqual(chunks[2]["text"], "C")

    def test_overlap_almost_equal_to_size(self):
        text = "one two three four"
        chunks = m.chunk_text(text, chunk_size=3, overlap=2)
        # step = 1
        # Chunk 1: one two three
        # Chunk 2: two three four
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["text"], "one two three")
        self.assertEqual(chunks[1]["text"], "two three four")


class TestBM25Index(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            "The quick brown fox",
            "Jump over the lazy dog",
            "The quick brown dog jumps",
            "No matching terms here",
        ]
        self.index = m.BM25Index(self.corpus, k1=1.5, b=0.75)

    def test_score_calculation(self):
        score_0 = self.index.score("quick brown", 0)
        score_1 = self.index.score("quick brown", 1)
        self.assertGreater(score_0, 0.0)
        self.assertEqual(score_1, 0.0)

    def test_search_ranking(self):
        results = self.index.search("quick dog", top_n=3)
        self.assertEqual(results[0][0], 2)
        self.assertTrue(results[0][1] > results[1][1])

    def test_oov_query_terms(self):
        results = self.index.search("xyzabc", top_n=5)
        self.assertEqual(len(results), 0)

    def test_parameter_influence(self):
        index_no_norm = m.BM25Index(self.corpus, k1=1.5, b=0.0)
        score_norm = self.index.score("dog", 2)
        score_no_norm = index_no_norm.score("dog", 2)
        self.assertNotEqual(score_norm, score_no_norm)

    def test_empty_corpus(self):
        empty_index = m.BM25Index([], k1=1.5, b=0.75)
        self.assertEqual(empty_index.search("test"), [])

    def test_single_document_corpus(self):
        single_index = m.BM25Index(["hello world"], k1=1.5, b=0.75)
        self.assertGreater(single_index.score("hello", 0), 0.0)
        self.assertEqual(single_index.score("missing", 0), 0.0)

    def test_all_documents_contain_term(self):
        # If all documents contain a term, IDF might be low but should remain valid
        all_index = m.BM25Index(["test", "test", "test"], k1=1.5, b=0.75)
        score = all_index.score("test", 0)
        self.assertGreater(score, 0.0)


class TestRRF(unittest.TestCase):
    def test_basic_fusion(self):
        ranking_1 = ["doc_a", "doc_b", "doc_c"]
        ranking_2 = ["doc_b", "doc_a"]
        fused = m.reciprocal_rank_fusion([ranking_1, ranking_2], k=60)
        self.assertEqual(len(fused), 3)
        self.assertEqual(fused[0][0], "doc_a")
        self.assertEqual(fused[1][0], "doc_b")
        self.assertEqual(fused[2][0], "doc_c")
        self.assertAlmostEqual(fused[2][1], 1 / 63)

    def test_empty_rankings(self):
        self.assertEqual(m.reciprocal_rank_fusion([]), [])
        self.assertEqual(m.reciprocal_rank_fusion([[], []]), [])

    def test_single_ranking(self):
        ranking = ["x", "y", "z"]
        fused = m.reciprocal_rank_fusion([ranking], k=0)
        self.assertEqual(fused[0], ("x", 1.0))
        self.assertEqual(fused[1], ("y", 0.5))
        self.assertEqual(fused[2], ("z", 1 / 3))

    def test_rrf_with_different_k(self):
        ranking_1 = ["doc_a", "doc_b"]
        ranking_2 = ["doc_b"]
        fused_k10 = m.reciprocal_rank_fusion([ranking_1, ranking_2], k=10)
        # doc_a: 1/(10+1) = 1/11
        # doc_b: 1/(10+2) + 1/(10+1) = 1/12 + 1/11
        self.assertEqual(fused_k10[0][0], "doc_b")
        self.assertAlmostEqual(fused_k10[1][1], 1 / 11)

    def test_rrf_tie_breaking(self):
        ranking_1 = ["z", "y", "x"]
        # All have equal scores if we fuse only one ranking
        fused = m.reciprocal_rank_fusion([ranking_1], k=60)
        # Should be sorted descending by score, then alphabetically by item ascending
        self.assertEqual(fused[0][0], "z")  # z has score 1/61
        self.assertEqual(fused[1][0], "y")  # y has score 1/62


class TestDedupeResults(unittest.TestCase):
    def test_no_duplicates(self):
        chunks = [
            {"text": "The quick brown fox"},
            {"text": "Jump over the lazy dog"},
        ]
        deduped = m.dedupe_results(chunks, threshold=0.5)
        self.assertEqual(len(deduped), 2)

    def test_exact_duplicates(self):
        chunks = [
            {"text": "The quick brown fox"},
            {"text": "The quick brown fox"},
        ]
        deduped = m.dedupe_results(chunks, threshold=0.5)
        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0]["text"], "The quick brown fox")

    def test_near_duplicates(self):
        chunks = [
            {"text": "The quick brown fox jumps over the lazy dog"},
            {"text": "quick brown fox jumps over the lazy dog indeed"},
            {"text": "completely different text here"},
        ]
        deduped = m.dedupe_results(chunks, threshold=0.5)
        self.assertEqual(len(deduped), 2)
        self.assertEqual(deduped[0]["text"], "The quick brown fox jumps over the lazy dog")
        self.assertEqual(deduped[1]["text"], "completely different text here")

    def test_empty_and_punctuation_only(self):
        chunks = [
            {"text": "!!!"},
            {"text": "???"},
        ]
        deduped = m.dedupe_results(chunks, threshold=0.5)
        self.assertEqual(len(deduped), 2)

    def test_dedupe_all_filtered(self):
        chunks = [
            {"text": "same same same"},
            {"text": "same same same"},
            {"text": "same same same"},
        ]
        deduped = m.dedupe_results(chunks, threshold=0.1)
        self.assertEqual(len(deduped), 1)

    def test_dedupe_threshold_boundaries(self):
        chunks = [
            {"text": "apple banana"},
            {"text": "apple cherry"},  # Jaccard = 1/3 = 0.333
        ]
        deduped_high = m.dedupe_results(chunks, threshold=0.4)
        deduped_low = m.dedupe_results(chunks, threshold=0.3)
        self.assertEqual(len(deduped_high), 2)
        self.assertEqual(len(deduped_low), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
