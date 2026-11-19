"""Tests for 43 — Memory: windows, summaries and recall."""

import importlib
import os
import unittest

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestCountTokens(unittest.TestCase):
    def test_simple_counting(self):
        self.assertEqual(m.count_tokens("hello world"), 2)

    def test_empty_string(self):
        self.assertEqual(m.count_tokens(""), 0)

    def test_whitespace_only(self):
        self.assertEqual(m.count_tokens("   \n  \t "), 0)

    def test_multiple_spaces(self):
        self.assertEqual(m.count_tokens("hello     world   again"), 3)


class TestTrimHistory(unittest.TestCase):
    def test_fits_entirely(self):
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        # Total tokens: 2 + 1 + 2 = 5
        trimmed = m.trim_history(messages, max_tokens=10)
        self.assertEqual(trimmed, messages)

    def test_basic_trimming(self):
        messages = [
            {"role": "system", "content": "System prompt"},  # 2 tokens
            {"role": "user", "content": "First question"},   # 2 tokens
            {"role": "assistant", "content": "First answer"}, # 2 tokens
            {"role": "user", "content": "Second question"},  # 2 tokens
        ]
        # Budget: 6 tokens. System (2) + Second question (2) + First answer (2) = 6. First question dropped.
        trimmed = m.trim_history(messages, max_tokens=6)
        expected = [
            {"role": "system", "content": "System prompt"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"},
        ]
        self.assertEqual(trimmed, expected)

    def test_no_system_message(self):
        messages = [
            {"role": "user", "content": "One"},
            {"role": "user", "content": "Two"},
            {"role": "user", "content": "Three"},
        ]
        # No system index matches. Budget: 2 tokens.
        trimmed = m.trim_history(messages, max_tokens=2, system_index=-1)
        expected = [
            {"role": "user", "content": "Two"},
            {"role": "user", "content": "Three"},
        ]
        self.assertEqual(trimmed, expected)

    def test_system_message_exceeds_budget(self):
        messages = [
            {"role": "system", "content": "Very long system instruction prompt"},  # 5 tokens
            {"role": "user", "content": "Hi"},
        ]
        with self.assertRaises(ValueError):
            m.trim_history(messages, max_tokens=3)

    def test_negative_budget(self):
        messages = [{"role": "user", "content": "Hi"}]
        with self.assertRaises(ValueError):
            m.trim_history(messages, max_tokens=-5)

    def test_preserves_chronological_order(self):
        messages = [
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "A"},
            {"role": "user", "content": "B"},
            {"role": "user", "content": "C"},
        ]
        trimmed = m.trim_history(messages, max_tokens=3)
        # Sys (1) + C (1) + B (1) = 3 tokens. Chronological order: Sys, B, C
        self.assertEqual(trimmed[0]["content"], "Sys")
        self.assertEqual(trimmed[1]["content"], "B")
        self.assertEqual(trimmed[2]["content"], "C")


class TestSummarizeOverflow(unittest.TestCase):
    def test_no_overflow_needed(self):
        messages = [
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "Hello"},
        ]
        summarizer = lambda msgs: "Should not be called"
        result = m.summarize_overflow(messages, max_tokens=10, summarizer=summarizer)
        self.assertEqual(result, messages)

    def test_basic_summarization(self):
        messages = [
            {"role": "system", "content": "Sys"}, # 1 token
            {"role": "user", "content": "very long message indeed yes"}, # 5 tokens
            {"role": "user", "content": "another very long message indeed"}, # 5 tokens
            {"role": "user", "content": "New"}, # 1 token
        ]

        def mock_summarizer(dropped):
            return "short"

        # Total tokens: 1 + 5 + 5 + 1 = 12.
        # max_tokens = 7.
        # trim_history(messages, 7) keeps: Sys (1), another... (5), New (1) -> total 7.
        # Dropped: "very long message indeed yes".
        # Summarizer returns "short".
        # Summary message: "Summary of earlier conversation: short" -> 5 tokens.
        # Fixed overhead: Sys (1) + Summary (5) = 6 tokens.
        # Remaining budget: 7 - 6 = 1 token.
        # Scanning backwards from kept: New (1) fits, another... (5) does not.
        # Final result: [Sys, Summary, New]
        result = m.summarize_overflow(messages, max_tokens=7, summarizer=mock_summarizer)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["content"], "Sys")
        self.assertIn("short", result[1]["content"])
        self.assertEqual(result[2]["content"], "New")

    def test_system_and_summary_exceed_budget(self):
        messages = [
            {"role": "system", "content": "Sys"}, # 1 token
            {"role": "user", "content": "Overflowing conversation history"}, # 3 tokens
        ]
        # Total tokens = 4. Let's set max_tokens = 3 so it triggers overflow.
        summarizer = lambda msgs: "Extremely long summary that will exceed the budget"
        with self.assertRaises(ValueError):
            m.summarize_overflow(messages, max_tokens=3, summarizer=summarizer)

    def test_no_system_message_summarization(self):
        messages = [
            {"role": "user", "content": "First very long message indeed"}, # 5 tokens
            {"role": "user", "content": "Second very long message indeed"}, # 5 tokens
        ]
        summarizer = lambda msgs: "Summary"
        # Total tokens = 10. Let's set max_tokens = 6.
        # trim_history(messages, 6, system_index=-1) keeps: Second... (5 tokens).
        # Dropped: First...
        # Summary message: "Summary of earlier conversation: Summary" -> 5 tokens.
        # Fixed overhead: 5 tokens.
        # Remaining budget: 6 - 5 = 1 token.
        # Kept newest: Second... (5 tokens) does not fit.
        # Final result: [Summary]
        result = m.summarize_overflow(messages, max_tokens=6, summarizer=summarizer, system_index=-1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["role"], "system")
        self.assertIn("Summary", result[0]["content"])


class TestMemoryStore(unittest.TestCase):
    def test_empty_store(self):
        store = m.MemoryStore()
        self.assertEqual(store.search("test"), [])

    def test_single_document(self):
        store = m.MemoryStore()
        store.add("Python is awesome")
        results = store.search("python", k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "Python is awesome")
        self.assertGreater(results[0][1], 0.0)

    def test_ranking_relevance(self):
        store = m.MemoryStore()
        store.add("The quick brown fox jumps over the lazy dog")
        store.add("Artificial intelligence and machine learning")
        store.add("Python programming language and software design")

        results = store.search("machine learning", k=2)
        self.assertEqual(results[0][0], "Artificial intelligence and machine learning")
        self.assertGreater(results[0][1], 0.0)
        self.assertEqual(results[1][1], 0.0)  # No overlap with Python doc

    def test_no_overlapping_words(self):
        store = m.MemoryStore()
        store.add("Apple banana cherry")
        results = store.search("dog cat", k=1)
        self.assertEqual(results[0][1], 0.0)

    def test_tie_breaking_preserves_insertion_order(self):
        store = m.MemoryStore()
        store.add("Document A")
        store.add("Document B")
        # Searching for a term not in either document ensures both have 0.0 similarity
        results = store.search("unrelated", k=2)
        self.assertEqual(results[0][0], "Document A")
        self.assertEqual(results[1][0], "Document B")

    def test_case_insensitivity_and_punctuation(self):
        store = m.MemoryStore()
        store.add("Database credentials: admin/secret!")
        results = store.search("DATABASE", k=1)
        self.assertEqual(results[0][0], "Database credentials: admin/secret!")
        self.assertGreater(results[0][1], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
