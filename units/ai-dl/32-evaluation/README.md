# 32 — Evaluation metrics

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/ai-dl/32-evaluation/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/ai-dl/evaluation?utm_source=github&utm_medium=labs)

*~50 minutes · 20 tests*

A model that trains without erroring is not the same as a model that works.
This module is the arithmetic behind the numbers that actually answer "does
it work": precision/recall/F1 for classification, mAP's building block for
detection and retrieval, BLEU and ROUGE for text generation, and recall@k
for embedding search. (Perplexity already has its own exercise in module
28, and IoU/NMS in module 27 — this module is the metrics that build on
top of those, not a repeat of them.)

Every metric here is checked against a hand-computed or `scikit-learn`-
verified reference value, so the numbers are pinned to the standard
definitions actually reported in papers and leaderboards, not to
approximations of them.

Work in `exercises.py`. Check your work with `python -m labkit check 32`, or open the notebook (`lab.ipynb`) in Colab.

Each section below ends with **Test it** — the exact command for just that
piece, run from `units/ai-dl/32-evaluation/`.

## What you'll build

| Name | You'll implement | Shows up in real code as |
|---|---|---|
| `precision_recall_f1` | `precision_recall_f1` | The three numbers behind every binary-classification eval report |
| `average_precision` | `average_precision` | The "AP" in mAP — the standard metric for detection and retrieval |
| `bleu_n` | `_ngrams`, `bleu_n` | The default machine-translation metric, still widely reported |
| `rouge_l` | `_lcs_length`, `rouge_l` | The default summarisation metric |
| `recall_at_k` | `recall_at_k` | What an embedding search / face-verification system is actually judged on |

---

## 1. Precision, recall, F1 — and why no single one is enough

```python
precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
f1 = 2 * precision * recall / (precision + recall)
```

Precision answers *"of what I predicted positive, how much was right"* —
it's what a false-positive-happy model gets punished for. Recall answers
*"of what was actually positive, how much did I find"* — what an
overly-conservative, false-negative-heavy model gets punished for. A
model that predicts everything positive scores a perfect 1.0 recall and
a terrible precision; a model that predicts almost nothing positive can
score a great precision on the few calls it does make. Neither alone
tells you whether the model is usable.

F1 is the **harmonic** mean specifically, not the arithmetic mean —
harmonic mean is dominated by the smaller of the two inputs, so a model
cannot hide a bad precision behind a great recall the way `(precision +
recall) / 2` would let it.

```python
>>> precision_recall_f1([1, 1, 0, 1, 0, 0, 1, 0], [1, 0, 0, 1, 0, 1, 1, 0])
(0.75, 0.75, 0.75)
```

**In `exercises.py`:** `precision_recall_f1`.

**Test it:** `python3 -m unittest test_exercises.TestPrecisionRecallF1 -v`

## 2. Average precision: the "AP" in mAP

```python
order = sorted(range(len(confidences)), key=lambda i: confidences[i], reverse=True)
# walk down in confidence order, tracking cumulative precision/recall,
# and sum precision * (the recall increase) at every step recall increases
```

Precision and recall alone need a fixed threshold. **Average precision**
sidesteps picking one: sort every prediction by confidence, walk down the
list as if the decision threshold were placed right after each one, and
sum up `precision * (recall increase)` at every point recall actually
increases. This is the exact definition `sklearn.metrics.
average_precision_score` uses — the area under the precision-recall
curve, non-interpolated.

A confident, correct model front-loads its true positives — precision
stays high while recall climbs — which is a high AP. A model with the
same overall accuracy but no correlation between confidence and
correctness earns a much lower AP, because AP specifically rewards
**ranking** true positives ahead of false positives, not just getting
the right total count. Object-detection and retrieval benchmarks report
**mAP** — the *mean* of this number across every class — which is why
this exercise computes AP for one class at a time: mAP is nothing more
than averaging what you're about to build.

```python
>>> average_precision([0.9, 0.8, 0.7, 0.6], labels=[1, 1, 0, 0])
1.0
>>> average_precision([0.9, 0.8, 0.7, 0.6], labels=[0, 0, 1, 1])  # positives ranked last
0.4166...
```

**In `exercises.py`:** `average_precision`.

**Test it:** `python3 -m unittest test_exercises.TestAveragePrecision -v`

## 3. BLEU: n-gram precision, geometric mean, brevity penalty

```python
clipped = sum(min(count, ref_ngrams[gram]) for gram, count in cand_ngrams.items())
precision_k = clipped / total_candidate_ngrams
```

For each n-gram order `k` from 1 to `n`, BLEU counts how many of the
candidate's `k`-grams appear in the reference — **clipped**, so a
`k`-gram can't be counted more times than it genuinely occurs in the
reference. Without clipping, repeating one correct word over and over
would trivially inflate the score. The **geometric mean** across orders
1 through `n` rewards matching at every granularity at once: a candidate
with all the right words in the wrong order scores fine on unigrams and
collapses on 4-grams, and a geometric mean (unlike an arithmetic one)
is dragged all the way down by that single collapsed order.

```python
brevity_penalty = 1.0 if len(candidate) >= len(reference) else math.exp(1 - len(reference) / len(candidate))
```

n-gram precision alone rewards being **short** — a one-word candidate
that happens to appear in the reference scores 100% precision on that
one word. The brevity penalty exists purely to cancel that: it drops
below 1 whenever the candidate is shorter than the reference, exactly
offsetting the incentive to truncate.

```python
>>> bleu_n(["the", "cat", "sat", "on", "the", "mat"], reference_same, n=4)
1.0
>>> bleu_n(["the", "cat", "sat"], reference_six_words, n=2)   # correct but truncated
0.3678794...
```

**In `exercises.py`:** `_ngrams`, `bleu_n`.

**Test it:** `python3 -m unittest test_exercises.TestBleuN -v`

## 4. ROUGE-L: credit for the right content, out of order

```python
lcs = _lcs_length(candidate, reference)     # longest common SUBsequence
precision = lcs / len(candidate)
recall = lcs / len(reference)
f1 = 2 * precision * recall / (precision + recall)
```

The **longest common subsequence** does not require matched tokens to be
*consecutive* — only in the same relative order — unlike BLEU's n-grams,
which require an exact consecutive run. That's the whole difference:
ROUGE-L gives partial credit to a candidate that contains the reference's
content in the right order with extra or missing words interspersed,
which n-gram precision penalises far more harshly.

This is also why the two metrics are conventionally used for different
tasks. **Translation** is usually scored with BLEU, because exact
phrasing (precision of consecutive spans) is most of what "a correct
translation" means. **Summarisation** is usually scored with ROUGE,
because covering the reference's content (recall of the underlying
information, even if reworded or reordered) matters more than matching
its exact wording.

```python
>>> rouge_l(candidate, candidate)   # identical
1.0
>>> rouge_l(["the", "mat", "sat", "cat"], ["the", "cat", "sat", "on", "the", "mat"])
0.4   # same words, out of order - some credit, not full credit
```

**In `exercises.py`:** `_lcs_length`, `rouge_l`.

**Test it:** `python3 -m unittest test_exercises.TestRougeL -v`

## 5. Recall@k: what a retrieval system is actually judged on

```python
similarities = gallery / gallery.norm(dim=-1, keepdim=True) @ (query / query.norm())
top_k_indices = torch.topk(similarities, k).indices
return correct_index in top_k_indices.tolist()
```

An embedding-search or face-verification system is not judged on whether
its single top result is correct — it's judged on whether the right
answer shows up **somewhere the user would actually see it**: the top-k
shortlist, the first page of results. Recall@1 (the special case `k=1`)
is the strict version; Recall@k for larger `k` is what's usually
reported in a retrieval paper's results table, averaged over many
queries.

```python
>>> recall_at_k(query, gallery, correct_index=3, k=1)    # top-1 must be exactly right
True
>>> recall_at_k(query, gallery, correct_index=3, k=10)   # correct answer anywhere in top-10
True
```

**In `exercises.py`:** `recall_at_k`.

**Test it:** `python3 -m unittest test_exercises.TestRecallAtK -v`

---

## The exercises

| Function | What it teaches |
|---|---|
| `precision_recall_f1` | Why F1 uses the harmonic, not arithmetic, mean |
| `average_precision` | Area under the precision-recall curve; the AP behind mAP |
| `bleu_n` | Clipped n-gram precision, geometric mean, brevity penalty |
| `rouge_l` | LCS-based scoring — order matters, adjacency doesn't |
| `recall_at_k` | The metric a retrieval/verification system is actually scored on |

Every function in this table has a stub in `exercises.py` and test
coverage in `test_exercises.py` — a clean 1:1 mapping.

## Check your understanding

1. **Why is F1 the harmonic mean of precision and recall instead of the
   arithmetic mean?**
   <details><summary>Answer</summary>The harmonic mean is dominated by
   the smaller of its two inputs, so a model can't compensate for a bad
   precision with a great recall (or vice versa) the way a simple
   average would let it. A model with precision 1.0 and recall 0.1 gets
   an arithmetic mean of 0.55 but a harmonic mean (F1) of about 0.18 —
   much closer to reflecting that it's barely usable.</details>

2. **What does average precision reward that overall accuracy at a
   single threshold does not?**
   <details><summary>Answer</summary>AP rewards RANKING true positives
   ahead of false positives across every possible confidence threshold,
   not just getting the right count at one fixed cutoff. A model can
   have the same accuracy as another at some threshold but a much lower
   AP if its confidence scores don't correlate well with
   correctness.</details>

3. **Why does BLEU need a brevity penalty at all — what would happen
   without one?**
   <details><summary>Answer</summary>n-gram precision alone rewards
   being short: a very short candidate that happens to be a subset of
   the reference's n-grams can score high precision by containing
   almost nothing. The brevity penalty explicitly cancels that
   incentive by scaling the score down whenever the candidate is
   shorter than the reference.</details>

4. **Why is ROUGE more commonly used for summarisation and BLEU more
   commonly used for translation?**
   <details><summary>Answer</summary>ROUGE (via longest common
   subsequence) gives credit for the reference's content appearing in
   the right relative order, even with words interspersed or reworded —
   which matches what "a good summary" means: capturing the content,
   not the exact phrasing. BLEU's n-gram precision requires exact
   consecutive matches, which is closer to what "a correct translation"
   means: matching specific phrasing, not just covering the same
   ideas.</details>

5. **What's the practical difference between Recall@1 and Recall@10 for
   a retrieval system, and why would a product care about the
   difference?**
   <details><summary>Answer</summary>Recall@1 only counts it as correct
   if the single top result is right; Recall@10 counts it as correct if
   the right answer appears anywhere in the top 10. A product that shows
   users a list of results (search, recommendations) cares more about
   Recall@10 — the user can scan a short list — while a product that
   only ever shows or acts on the single top result (an autocomplete
   suggestion, a fully automated decision) cares about
   Recall@1.</details>

## Interview questions this module answers

- *Why would you report F1 instead of accuracy for an imbalanced
  classification problem?*
- *What does mean Average Precision actually average, and over what?*
- *Why does BLEU penalise short outputs, and what would happen without
  that penalty?*
- *Why is ROUGE the standard metric for summarisation instead of
  something like BLEU?*
- *How do you evaluate whether an embedding-based search system is
  actually working?*
