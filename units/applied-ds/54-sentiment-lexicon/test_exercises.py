"""Tests for 54 — Sentiment analysis from a lexicon."""

import importlib
import os
import unittest

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestTokenize(unittest.TestCase):
    def test_basic_tokenization(self):
        self.assertEqual(
            m.tokenize("The movie was great"),
            ["the", "movie", "was", "great"]
        )

    def test_preserves_emoticons(self):
        self.assertEqual(
            m.tokenize("happy :) sad :("),
            ["happy", ":)", "sad", ":("]
        )
        self.assertEqual(
            m.tokenize("excited :D wink ;)"),
            ["excited", ":d", "wink", ";)"]
        )

    def test_preserves_apostrophes_and_punctuation_tokens(self):
        self.assertEqual(
            m.tokenize("I didn't like it!!!"),
            ["i", "didn't", "like", "it", "!", "!", "!"]
        )

    def test_ignores_other_punctuation(self):
        self.assertEqual(
            m.tokenize("Hello, world... This is a test, right?"),
            ["hello", "world", "this", "is", "a", "test", "right", "?"]
        )

    def test_empty_string(self):
        self.assertEqual(m.tokenize(""), [])


class TestScoreTokens(unittest.TestCase):
    def setUp(self):
        self.lexicon = {"good": 2.0, "bad": -2.0, "happy": 3.0}
        self.negations = {"not", "never", "no"}
        self.boosters = {"very": 1.0, "slightly": -0.5}

    def test_no_lexicon_matches(self):
        tokens = ["the", "sky", "is", "blue"]
        self.assertEqual(m.score_tokens(tokens, self.lexicon, self.negations, self.boosters), [0.0, 0.0, 0.0, 0.0])

    def test_basic_lexicon_match(self):
        tokens = ["good", "and", "happy"]
        self.assertEqual(m.score_tokens(tokens, self.lexicon, self.negations, self.boosters), [2.0, 0.0, 3.0])

    def test_negation_window(self):
        # Negation immediately preceding
        self.assertEqual(
            m.score_tokens(["not", "good"], self.lexicon, self.negations, self.boosters),
            [0.0, -1.0]
        )
        # Negation 2 tokens away
        self.assertEqual(
            m.score_tokens(["not", "very", "good"], self.lexicon, self.negations, self.boosters),
            [0.0, 0.0, -2.0]
        )
        # Negation 3 tokens away
        self.assertEqual(
            m.score_tokens(["never", "been", "so", "happy"], self.lexicon, self.negations, self.boosters),
            [0.0, 0.0, 0.0, -1.5]
        )
        # Negation 4 tokens away (outside window)
        self.assertEqual(
            m.score_tokens(["no", "one", "is", "really", "good"], self.lexicon, self.negations, self.boosters),
            [0.0, 0.0, 0.0, 0.0, 2.0]
        )

    def test_booster_words(self):
        # Positive booster
        self.assertEqual(
            m.score_tokens(["very", "good"], self.lexicon, self.negations, self.boosters),
            [0.0, 3.0]
        )
        # Negative booster
        self.assertEqual(
            m.score_tokens(["slightly", "bad"], self.lexicon, self.negations, self.boosters),
            [0.0, -1.5]
        )

    def test_negation_and_booster_combined(self):
        # "not very good" -> good is 2.0. Negated -> -1.0. Preceded by booster "very" (1.0) -> -1.0 - 1.0 = -2.0.
        self.assertEqual(
            m.score_tokens(["not", "very", "good"], self.lexicon, self.negations, self.boosters),
            [0.0, 0.0, -2.0]
        )


class TestNormalizeCompound(unittest.TestCase):
    def test_zero_score(self):
        self.assertEqual(m.normalize_compound(0.0), 0.0)

    def test_positive_score(self):
        score = 4.0
        # 4.0 / sqrt(16 + 15) = 4.0 / sqrt(31) = 4.0 / 5.56776 = 0.71842
        self.assertAlmostEqual(m.normalize_compound(score, alpha=15.0), 0.7184212, places=5)

    def test_negative_score(self):
        score = -2.0
        # -2.0 / sqrt(4 + 10) = -2.0 / sqrt(14) = -2.0 / 3.74165 = -0.53452
        self.assertAlmostEqual(m.normalize_compound(score, alpha=10.0), -0.5345224, places=5)


class TestClassifySentiment(unittest.TestCase):
    def test_default_thresholds(self):
        self.assertEqual(m.classify_sentiment(0.1), "positive")
        self.assertEqual(m.classify_sentiment(-0.1), "negative")
        self.assertEqual(m.classify_sentiment(0.01), "neutral")
        self.assertEqual(m.classify_sentiment(0.05), "positive")
        self.assertEqual(m.classify_sentiment(-0.05), "negative")

    def test_custom_thresholds(self):
        self.assertEqual(m.classify_sentiment(0.1, pos_threshold=0.2, neg_threshold=-0.2), "neutral")
        self.assertEqual(m.classify_sentiment(0.25, pos_threshold=0.2, neg_threshold=-0.2), "positive")
        self.assertEqual(m.classify_sentiment(-0.25, pos_threshold=0.2, neg_threshold=-0.2), "negative")


class TestEvaluatePredictions(unittest.TestCase):
    def test_perfect_predictions(self):
        y_true = ["positive", "neutral", "negative"]
        y_pred = ["positive", "neutral", "negative"]
        acc, matrix = m.evaluate_predictions(y_true, y_pred)
        self.assertEqual(acc, 1.0)
        self.assertEqual(matrix["positive"]["positive"], 1)
        self.assertEqual(matrix["neutral"]["neutral"], 1)
        self.assertEqual(matrix["negative"]["negative"], 1)
        self.assertEqual(matrix["positive"]["negative"], 0)

    def test_imperfect_predictions(self):
        y_true = ["positive", "positive", "negative", "neutral"]
        y_pred = ["positive", "neutral", "negative", "neutral"]
        acc, matrix = m.evaluate_predictions(y_true, y_pred)
        self.assertEqual(acc, 0.75)
        self.assertEqual(matrix["positive"]["positive"], 1)
        self.assertEqual(matrix["positive"]["neutral"], 1)
        self.assertEqual(matrix["negative"]["negative"], 1)
        self.assertEqual(matrix["neutral"]["neutral"], 1)

    def test_empty_inputs(self):
        acc, matrix = m.evaluate_predictions([], [])
        self.assertEqual(acc, 0.0)
        for t in ["positive", "neutral", "negative"]:
            for p in ["positive", "neutral", "negative"]:
                self.assertEqual(matrix[t][p], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
