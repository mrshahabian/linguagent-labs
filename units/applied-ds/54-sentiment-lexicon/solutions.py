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
    pattern = r"[;:][()DdPp]|[a-zA-Z0-9']+|[!?]"
    return re.findall(pattern, text.lower())


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
    scores = []
    for i, token in enumerate(tokens):
        base_score = lexicon.get(token, 0.0)
        if base_score == 0.0:
            scores.append(0.0)
            continue
        
        # Check negation in the preceding 3 tokens
        is_negated = False
        for j in range(max(0, i - 3), i):
            if tokens[j] in negations:
                is_negated = True
                break
        
        score = base_score
        if is_negated:
            score *= -0.5
            
        # Check booster in the immediately preceding token
        if i > 0:
            prev_token = tokens[i - 1]
            if prev_token in boosters:
                booster_val = boosters[prev_token]
                if score > 0:
                    score += booster_val
                elif score < 0:
                    score -= booster_val
                    
        scores.append(score)
    return scores


def normalize_compound(score: float, alpha: float = 15.0) -> float:
    """Normalize a raw sum score into a range between -1.0 and 1.0.
    
    Formula: score / sqrt(score^2 + alpha)
    If score is 0.0, return 0.0.
    """
    if score == 0.0:
        return 0.0
    return score / math.sqrt(score**2 + alpha)


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
    if compound_score >= pos_threshold:
        return "positive"
    elif compound_score <= neg_threshold:
        return "negative"
    else:
        return "neutral"


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
    classes = ["positive", "neutral", "negative"]
    matrix = {t: {p: 0 for p in classes} for t in classes}
    
    if not y_true:
        return 0.0, matrix
        
    correct = 0
    for t, p in zip(y_true, y_pred):
        if t == p:
            correct += 1
        if t in matrix and p in matrix[t]:
            matrix[t][p] += 1
            
    accuracy = correct / len(y_true)
    return accuracy, matrix
