"""51 - Evaluating LLM systems

Implement each function so the tests in test_exercises.py pass. Read
README.md first; every exercise below is covered there.

The docstrings are the specification. Standard library and numpy only.
"""

import collections
import re
import string
import numpy as np


def normalize_answer(text: str) -> str:
    """Normalize text for QA evaluation:
    1. Lowercase the text.
    2. Strip punctuation.
    3. Strip English articles ('a', 'an', 'the') using word boundaries.
    4. Normalize whitespace (split and rejoin with single spaces, strip ends).
    
    Args:
        text: The raw string to normalize.
        
    Returns:
        The normalized string.
    """
    raise NotImplementedError("your code here")


def exact_match(prediction: str, ground_truth: str) -> bool:
    """Return True if the normalized prediction matches the normalized ground truth.
    
    Args:
        prediction: The predicted answer string.
        ground_truth: The correct answer string.
        
    Returns:
        True if they match after normalization, False otherwise.
    """
    raise NotImplementedError("your code here")


def token_level_f1(prediction: str, ground_truth: str) -> float:
    """Compute SQuAD-style token-level F1 score.
    
    Treats normalized prediction and ground truth as bags of words.
    If both are empty after normalization, returns 1.0.
    If only one is empty, returns 0.0.
    
    Args:
        prediction: The predicted answer string.
        ground_truth: The correct answer string.
        
    Returns:
        The F1 score as a float between 0.0 and 1.0.
    """
    raise NotImplementedError("your code here")


def recall_at_k(retrieved_ids: list, ground_truth_ids: list, k: int) -> float:
    """Compute Recall@K.
    
    Recall@K is the proportion of ground truth IDs that are present in the
    top K retrieved IDs. If ground_truth_ids is empty, return 0.0.
    
    Args:
        retrieved_ids: Ordered list of retrieved document/chunk IDs.
        ground_truth_ids: List of relevant document/chunk IDs.
        k: The cutoff rank.
        
    Returns:
        Recall@K as a float between 0.0 and 1.0.
    """
    raise NotImplementedError("your code here")


def mean_reciprocal_rank(retrieved_batches: list[list], ground_truth_batches: list[list]) -> float:
    """Compute Mean Reciprocal Rank (MRR) across a batch of queries.
    
    For each query, find the 1-based rank of the first retrieved item that
    is present in the ground truth list. The reciprocal rank is 1 / rank.
    If no retrieved items match the ground truth, the reciprocal rank is 0.0.
    MRR is the mean of reciprocal ranks across all queries.
    
    If retrieved_batches is empty, return 0.0.
    
    Args:
        retrieved_batches: List of retrieved ID lists (one list per query).
        ground_truth_batches: List of ground truth ID lists (one list per query).
        
    Returns:
        The MRR as a float between 0.0 and 1.0.
    """
    raise NotImplementedError("your code here")


def win_rate(judgements: list[str]) -> float:
    """Compute the win rate of Model A from pairwise judgements.
    
    Judgements is a list of strings, where each element is one of:
    - 'A': Model A won
    - 'B': Model B won
    - 'tie': Both models tied
    
    The formula is: (wins_A + 0.5 * ties) / total_judgements.
    If judgements is empty, return 0.0.
    
    Args:
        judgements: List of judgement strings.
        
    Returns:
        The win rate of Model A as a float between 0.0 and 1.0.
    """
    raise NotImplementedError("your code here")


def bootstrap_ci(data: list, metric_fn, num_resamples: int = 1000, confidence_level: float = 0.95, seed: int = 42) -> tuple[float, float]:
    """Compute the confidence interval of a metric using percentile bootstrapping.
    
    Uses np.random.default_rng(seed) to generate resamples with replacement.
    For each resample, computes the metric using metric_fn.
    Returns the lower and upper bounds corresponding to the confidence level.
    
    Args:
        data: List of data points (e.g., scores, judgements).
        metric_fn: A callable that takes a list of data points and returns a float.
        num_resamples: Number of bootstrap resamples to perform.
        confidence_level: The confidence level (e.g., 0.95 for 95% CI).
        seed: Seed for the random number generator.
        
    Returns:
        A tuple (lower_bound, upper_bound) as floats.
    """
    raise NotImplementedError("your code here")
