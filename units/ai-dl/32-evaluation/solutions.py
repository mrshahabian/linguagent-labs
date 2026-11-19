"""Reference answers for 32 - evaluation metrics."""

import math
from collections import Counter

import torch


def precision_recall_f1(predictions, labels):
    """Binary precision, recall and F1 from 0/1 predictions and labels.

    Precision answers "of what I predicted positive, how much was
    right" - it is what a false-positive-heavy model gets punished for.
    Recall answers "of what was actually positive, how much did I find"
    - it is what a false-negative-heavy (overly conservative) model gets
    punished for. Neither alone tells you whether a model is usable; a
    model that predicts everything positive gets perfect recall and
    terrible precision. F1 is their harmonic mean specifically because
    the harmonic mean is dominated by the smaller of the two - a model
    cannot hide a bad precision behind a great recall in its F1 score
    the way an arithmetic mean would let it.
    """
    true_positive = sum(1 for p, y in zip(predictions, labels) if p == 1 and y == 1)
    false_positive = sum(1 for p, y in zip(predictions, labels) if p == 1 and y == 0)
    false_negative = sum(1 for p, y in zip(predictions, labels) if p == 0 and y == 1)

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def average_precision(confidences, labels):
    """Area under the precision-recall curve for one class - the "AP" in mAP.

    Sort predictions by confidence, descending, and walk down the list
    computing precision and recall as if the threshold were placed right
    after each prediction. AP is the sum, over every step where recall
    INCREASES, of (recall_increase * precision_at_that_step) - the same
    non-interpolated definition scikit-learn's average_precision_score
    uses. mAP, the metric actually reported for detection/retrieval
    benchmarks, is simply the mean of this number across every class.

    A confident, correct model front-loads its true positives, so
    precision stays high while recall climbs - a high AP. A model that
    is equally likely to be right or wrong at any confidence level earns
    a low AP even with the same overall accuracy, because AP specifically
    rewards RANKING true positives ahead of false positives.
    """
    total_positive = sum(labels)
    if total_positive == 0:
        return 0.0
    order = sorted(range(len(confidences)), key=lambda i: confidences[i], reverse=True)
    true_positive = 0
    false_positive = 0
    ap = 0.0
    previous_recall = 0.0
    for i in order:
        if labels[i] == 1:
            true_positive += 1
        else:
            false_positive += 1
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / total_positive
        ap += (recall - previous_recall) * precision
        previous_recall = recall
    return ap


def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu_n(candidate, reference, n=4):
    """Single-reference BLEU-n: n-gram precision, geometrically averaged, with
    a brevity penalty.

    For each order k in 1..n, count how many of candidate's k-grams
    appear in reference (CLIPPED - a k-gram can't count more times than
    it actually occurs in the reference, or repeating one good word
    would trivially inflate the score). The geometric mean of these
    per-order precisions rewards a candidate that matches at EVERY
    granularity, not just individual words - a candidate with all the
    right words in the wrong order scores well on 1-grams but collapses
    on 4-grams.

    The brevity penalty exists because n-gram precision alone rewards
    being short: a one-word candidate that happens to appear in the
    reference gets 100% precision. BP < 1 whenever the candidate is
    shorter than the reference, exactly cancelling that incentive.
    """
    if not candidate:
        return 0.0
    precisions = []
    for k in range(1, n + 1):
        if len(candidate) < k:
            precisions.append(0.0)
            continue
        cand_ngrams = _ngrams(candidate, k)
        ref_ngrams = _ngrams(reference, k)
        clipped = sum(min(count, ref_ngrams[gram]) for gram, count in cand_ngrams.items())
        total = sum(cand_ngrams.values())
        precisions.append(clipped / total if total else 0.0)
    if min(precisions) == 0.0:
        return 0.0
    geometric_mean = math.exp(sum(math.log(p) for p in precisions) / n)
    brevity_penalty = 1.0 if len(candidate) >= len(reference) else math.exp(1 - len(reference) / len(candidate))
    return brevity_penalty * geometric_mean


def _lcs_length(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l(candidate, reference):
    """ROUGE-L F1: precision and recall built from the longest common subsequence.

    Unlike BLEU's n-grams, the LCS does not require matched words to be
    CONSECUTIVE - only in the same relative order - so ROUGE-L gives
    partial credit to a candidate that has the right content in the
    right order but with extra or missing words interspersed, which
    n-gram precision penalises harshly. This is why summarisation is
    usually scored with ROUGE (recall of the reference's content
    matters most) while translation is usually scored with BLEU
    (precision of exact phrasing matters most).
    """
    if not candidate or not reference:
        return 0.0
    lcs = _lcs_length(candidate, reference)
    precision = lcs / len(candidate)
    recall = lcs / len(reference)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def recall_at_k(query_embed, gallery_embeds, correct_index, k):
    """True if the correct gallery item is among the top-k nearest by cosine similarity.

    This is the metric an embedding-based retrieval or face-verification
    system is actually judged on - not "is the top-1 result correct"
    (recall@1, a special case of this) but "does the right answer show
    up somewhere in what the user would actually see" (the first page
    of results, the top-k shortlist). Averaging this indicator over many
    queries gives Recall@k, the number usually reported in a retrieval
    paper's results table.
    """
    query = query_embed / query_embed.norm().clamp_min(1e-8)
    gallery = gallery_embeds / gallery_embeds.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    similarities = gallery @ query
    top_k_indices = torch.topk(similarities, k).indices
    return correct_index in top_k_indices.tolist()
