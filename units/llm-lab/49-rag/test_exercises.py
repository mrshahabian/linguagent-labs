"""Tests for 49 - Retrieval-augmented generation."""

import importlib
import os
import unittest

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestSelectContext(unittest.TestCase): 
    def setUp(self):
        self.chunks = [
            {"id": "A", "text": "Python is a language.", "score": 0.9},
            {"id": "B", "text": "It supports multiple paradigms.", "score": 0.8},
            {"id": "C", "text": "Guido created it.", "score": 0.7},
            {"id": "D", "text": "Short text.", "score": 0.6}
        ]
        self.word_counter = lambda text: len(text.split())

    def test_empty_chunks(self):
        self.assertEqual(m.select_context([], 10, self.word_counter), [])

    def test_all_fit(self):
        selected = m.select_context(self.chunks, 100, self.word_counter)
        self.assertEqual(len(selected), 4)
        self.assertEqual(selected[0]["id"], "A")

    def test_greedy_packing_skips_and_continues(self):
        # Budget: 8 words.
        # Chunk A: 4 words -> Fits (4 left)
        # Chunk B: 4 words -> Fits (0 left)
        # Chunk C: 3 words -> Exceeds (0 left)
        # Chunk D: 2 words -> Exceeds (0 left)
        selected = m.select_context(self.chunks, 8, self.word_counter)
        self.assertEqual([c["id"] for c in selected], ["A", "B"])

    def test_greedy_packing_skips_large_fits_small(self):
        # Budget: 7 words.
        # Chunk A: 4 words -> Fits (3 left)
        # Chunk B: 4 words -> Exceeds (3 left)
        # Chunk C: 3 words -> Fits (0 left)
        # Chunk D: 2 words -> Exceeds (0 left)
        selected = m.select_context(self.chunks, 7, self.word_counter)
        self.assertEqual([c["id"] for c in selected], ["A", "C"])

    def test_zero_budget(self):
        selected = m.select_context(self.chunks, 0, self.word_counter)
        self.assertEqual(selected, [])


class TestBuildPrompt(unittest.TestCase):
    def test_default_system_instruction(self):
        chunks = [{"text": "Python is fun."}]
        prompt = m.build_prompt("What is Python?", chunks)
        
        self.assertIn("You are a helpful assistant.", prompt)
        self.assertIn("Sources:\n[1] Python is fun.", prompt)
        self.assertIn("Question: What is Python?", prompt)
        self.assertTrue(prompt.endswith("Answer:"))

    def test_custom_system_instruction(self):
        chunks = [{"text": "Python is fun."}]
        prompt = m.build_prompt("What is Python?", chunks, system_instruction="Custom instruction.")
        
        self.assertTrue(prompt.startswith("Custom instruction."))
        self.assertIn("Sources:\n[1] Python is fun.", prompt)

    def test_multiple_sources_formatting(self):
        chunks = [{"text": "Chunk one."}, {"text": "Chunk two."}]
        prompt = m.build_prompt("Query", chunks)
        expected_sources = "Sources:\n[1] Chunk one.\n[2] Chunk two."
        self.assertIn(expected_sources, prompt)

    def test_whitespace_stripping(self):
        chunks = [{"text": "  Chunk with spaces.  "}]
        prompt = m.build_prompt("  Query?  ", chunks, system_instruction="  System.  ")
        self.assertTrue(prompt.startswith("System."))
        self.assertIn("[1] Chunk with spaces.", prompt)
        self.assertIn("Question: Query?", prompt)


class TestParseCitations(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {"text": "Python is a language."},
            {"text": "It supports OOP."},
            {"text": "Guido van Rossum created it."}
        ]

    def test_no_citations(self):
        answer = "Python is a programming language with dynamic typing."
        self.assertEqual(m.parse_citations(answer, self.chunks), {})

    def test_valid_citations(self):
        answer = "According to [1], Python is a language. It supports OOP [2]."
        expected = {
            1: "Python is a language.",
            2: "It supports OOP."
        }
        self.assertEqual(m.parse_citations(answer, self.chunks), expected)

    def test_hallucinated_citations_ignored(self):
        answer = "Python is great [1], but some say [5] it is slow."
        expected = {
            1: "Python is a language."
        }
        self.assertEqual(m.parse_citations(answer, self.chunks), expected)

    def test_duplicate_citations(self):
        answer = "Python is a language [1] and again [1]."
        expected = {
            1: "Python is a language."
        }
        self.assertEqual(m.parse_citations(answer, self.chunks), expected)


class TestGroundingScore(unittest.TestCase):
    def setUp(self):
        self.citations = {
            1: "Python is a programming language.",
            2: "It is widely used in machine learning."
        }

    def test_perfect_grounding(self):
        # Trigrams: "python is a", "is a programming", "a programming language"
        answer = "Python is a programming language [1]."
        score = m.grounding_score(answer, self.citations, n=3)
        self.assertEqual(score, 1.0)

    def test_no_citations_is_zero(self):
        answer = "Python is a programming language."
        score = m.grounding_score(answer, self.citations, n=3)
        self.assertEqual(score, 0.0)

    def test_partial_grounding(self):
        # Sentence: "Python is a language [1]."
        # Cleaned words: ["python", "is", "a", "language"]
        # Trigrams: ("python", "is", "a"), ("is", "a", "language")
        # Source words: ["python", "is", "a", "programming", "language"]
        # Source trigrams: ("python", "is", "a"), ("is", "a", "programming"), ("a", "programming", "language")
        # Match: ("python", "is", "a") matches. ("is", "a", "language") does not.
        # Precision: 1 / 2 = 0.5
        answer = "Python is a language [1]."
        score = m.grounding_score(answer, self.citations, n=3)
        self.assertEqual(score, 0.5)

    def test_short_sentence_fallback(self):
        # Sentence has only 2 words, but n=3. It should default to 1.0 if cited.
        answer = "Indeed [1]."
        score = m.grounding_score(answer, self.citations, n=3)
        self.assertEqual(score, 1.0)

    def test_multiple_sentences_average(self):
        # Sentence 1: Perfect grounding (1.0)
        # Sentence 2: No citation (0.0)
        answer = "Python is a programming language [1]. It is also cool."
        score = m.grounding_score(answer, self.citations, n=3)
        self.assertEqual(score, 0.5)

    def test_empty_answer(self):
        self.assertEqual(m.grounding_score("", self.citations), 0.0)


class TestAnswerOrAbstain(unittest.TestCase):
    def test_above_threshold(self):
        answer = "Python is great [1]."
        result = m.answer_or_abstain(answer, 0.8, threshold=0.7)
        self.assertEqual(result, answer)

    def test_below_threshold(self):
        answer = "Python is great [1]."
        result = m.answer_or_abstain(answer, 0.5, threshold=0.7)
        self.assertEqual(result, "I cannot answer this question based on the provided sources.")

    def test_custom_abstain_message(self):
        answer = "Python is great [1]."
        result = m.answer_or_abstain(answer, 0.5, threshold=0.7, abstain_message="Sorry!")
        self.assertEqual(result, "Sorry!")


if __name__ == "__main__":
    unittest.main(verbosity=2)
