"""51 - Evaluating LLM systems

Reference solutions for the exercises.
"""

import collections
import re
import string
import numpy as np


def normalize_answer(text: str) -> str:
    """Normalize text for QA evaluation: lowercase, strip punctuation, strip articles, normalize whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def exact_match(prediction: str, ground_truth: str) -> bool:
    """Return True if the normalized prediction matches the normalized ground truth."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def token_level_f1(prediction: str, ground_truth: str) -> float:
    """Compute SQuAD-style token-level F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 1.0 if len(pred_tokens) == len(gt_tokens) else 0.0
    
    common = collections.Counter(pred_tokens) & collections.Counter(gt_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def recall_at_k(retrieved_ids: list, ground_truth_ids: list, k: int) -> float:
    """Compute Recall@K. Proportion of ground_truth_ids found in top k of retrieved_ids."""
    if not ground_truth_ids:
        return 0.0
    retrieved_k = set(retrieved_ids[:k])
    gt_set = set(ground_truth_ids)
    intersection = retrieved_k.intersection(gt_set)
    return len(intersection) / len(gt_set)


def mean_reciprocal_rank(retrieved_batches: list[list], ground_truth_batches: list[list]) -> float:
    """Compute Mean Reciprocal Rank (MRR) across a batch of queries."""
    if not retrieved_batches:
        return 0.0
    
    rr_sum = 0.0
    for retrieved, ground_truth in zip(retrieved_batches, ground_truth_batches):
        gt_set = set(ground_truth)
        for rank, item in enumerate(retrieved, start=1):
            if item in gt_set:
                rr_sum += 1.0 / rank
                break
    return rr_sum / len(retrieved_batches)


def win_rate(judgements: list[str]) -> float:
    """Compute the win rate of Model A from pairwise judgements."""
    if not judgements:
        return 0.0
    wins_a = judgements.count("A")
    ties = judgements.count("tie")
    return (wins_a + 0.5 * ties) / len(judgements)


def bootstrap_ci(data: list, metric_fn, num_resamples: int = 1000, confidence_level: float = 0.95, seed: int = 42) -> tuple[float, float]:
    """Compute the confidence interval of a metric using percentile bootstrapping."""
    if not data:
        raise ValueError("Data cannot be empty")
    
    rng = np.random.default_rng(seed)
    n = len(data)
    boot_metrics = []
    
    for _ in range(num_resamples):
        indices = rng.choice(n, size=n, replace=True)
        resample = [data[i] for i in indices]
        boot_metrics.append(metric_fn(resample))
    
    lower_pct = (1.0 - confidence_level) / 2.0 * 100
    upper_pct = (1.0 + confidence_level) / 2.0 * 100
    
    lower_bound = np.percentile(boot_metrics, lower_pct)
    upper_bound = np.percentile(boot_metrics, upper_pct)
    return float(lower_bound), float(upper_bound)
