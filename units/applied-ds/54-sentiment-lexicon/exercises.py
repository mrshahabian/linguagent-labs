"""54 — Sentiment analysis from a lexicon

Implement each function so the tests in test_exercises.py pass. Read
README.md first; every exercise below is covered there.
"""

import math
import re


def tokenize(text: str) -> list[str]:
    """Tokenize input text into lowercased tokens.
    
    Preserve emoticons (specifically :) :( :D ;) as single tokens),
    words with internal apostrophes (like "don't"), and exclamation/question
    marks (! and ?) as separate tokens. Discard all other punctuation.
    
    Example:
        tokenize("I'm happy :)!") -> ["i'm", "happy", ":)", "!"]
    """
    raise NotImplementedError("your code here")


def score_tokens(
    tokens: list[str],
    lexicon: dict[str, float],
    negations: set[str],
    boosters: dict[str, float]
) -> list[float]:
    """Calculate the sentiment score for each token using contextual rules.
    
    Rules:
    1. If a token is not in the lexicon, its score is 0.0.
    2. If a token is in the lexicon, start with its base score.
    3. If any of the 3 preceding tokens (indices i-1, i-2, i-3) are in the
       negations set, multiply the score by -0.5.
    4. If the immediately preceding token (index i-1) is in the boosters dict,
       adjust the score: add the booster value if the score is positive,
       or subtract the booster value if the score is negative.
       
    Example:
        lexicon = {"good": 2.0}
        negations = {"not"}
        boosters = {"very": 1.0}
        score_tokens(["not", "very", "good"], lexicon, negations, boosters)
        -> [0.0, 0.0, -2.0]
    """
    raise NotImplementedError("your code here")


def normalize_compound(score: float, alpha: float = 15.0) -> float:
    """Normalize a raw sum score into a range between -1.0 and 1.0.
    
    Formula: score / sqrt(score^2 + alpha)
    If score is 0.0, return 0.0.
    """
    raise NotImplementedError("your code here")


def classify_sentiment(
    compound_score: float,
    pos_threshold: float = 0.05,
    neg_threshold: float = -0.05
) -> str:
    """Classify a compound score into 'positive', 'negative', or 'neutral'.
    
    - 'positive' if compound_score >= pos_threshold
    - 'negative' if compound_score <= neg_threshold
    - 'neutral' otherwise
    """
    raise NotImplementedError("your code here")


def evaluate_predictions(
    y_true: list[str],
    y_pred: list[str]
) -> tuple[float, dict[str, dict[str, int]]]:
    """Calculate overall accuracy and a 3x3 confusion matrix.
    
    The confusion matrix must be a nested dictionary where the outer keys
    are true labels and the inner keys are predicted labels. Both outer and
    inner keys must contain all three classes: 'positive', 'neutral', 'negative'.
    
    If y_true is empty, return 0.0 accuracy and a matrix with all 0 counts.
    
    Returns:
        (accuracy, confusion_matrix)
    """
    raise NotImplementedError("your code here")
