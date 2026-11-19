# 51 — Evaluating LLM systems

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/llm-lab/51-llm-evaluation/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/llm-lab/llm-evaluation?utm_source=github&utm_medium=labs)

*~45 minutes · 24 tests · Advanced*

Building an LLM application is easy; knowing if you actually made it better is incredibly hard. When you tweak a prompt, swap an embedding model, or change your chunk size, how do you know if the output improved? Relying on "vibe checks"—manually reading 5 or 10 outputs—is a recipe for silent regressions. You need automated, quantitative evaluation.

In this unit, we will build the core metrics used to evaluate both retrieval and generation. We will start with classic lexical matching metrics (Exact Match and SQuAD-style token-level F1), move to retrieval metrics (Recall@K and Mean Reciprocal Rank), implement pairwise win rates (the engine behind LLM-as-a-judge), and finally, build a bootstrapping engine to calculate confidence intervals. This statistical foundation ensures you don't mistake random noise for a genuine model improvement.

These techniques are the exact building blocks used in production evaluation frameworks like Hugging Face's Lighteval, Ragas, and LMSYS Chatbot Arena. Implementing them from scratch will give you a deep, mathematical intuition for how LLM systems are measured and compared.

## What you'll build

Seven functions in `exercises.py`, in this order. Keep this table open as you go — each section below tells you which row it's building:

| Function / class | You'll implement | Shows up in real code as |
|---|---|---|
| `normalize_answer` | Text cleaning (lowercase, strip punctuation, strip articles) | Standard preprocessing for QA evaluation (SQuAD style) |
| `exact_match` | Binary exact string equivalence after normalization | Strict accuracy tracking for factual and short-answer QA |
| `token_level_f1` | Word-overlap precision, recall, and F1 score | SQuAD evaluation metric for span extraction and generation |
| `recall_at_k` | Proportion of relevant documents retrieved in top K | Evaluating search engines and vector database retrieval |
| `mean_reciprocal_rank` | Average of the reciprocal rank of the first correct result | Ranking quality metric for search and recommendation systems |
| `win_rate` | Pairwise comparison score accounting for ties | LLM-as-a-judge leaderboards (Chatbot Arena, AlpacaEval) |
| `bootstrap_ci` | Non-parametric confidence intervals via resampling | Statistical significance testing for model comparisons |

---

## 1. Normalization and Exact Match

```python
>>> normalize_answer("The Eiffel Tower.")
'eiffel tower'
>>> exact_match("The Eiffel Tower.", "eiffel tower")
True
```

If the ground truth answer is "the Eiffel Tower" and your model outputs "Eiffel Tower.", they are semantically identical. A naive string comparison `==` will fail. To fix this, we implement standard SQuAD-style normalization:
1. Lowercase the text.
2. Remove punctuation characters (using `string.punctuation`).
3. Remove English articles ("a", "an", "the") using word-boundary regular expressions.
4. Normalize whitespace (split by whitespace and rejoin with a single space, stripping leading/trailing spaces).

Once normalized, `exact_match` is a simple binary comparison of the cleaned prediction and ground truth.

**The Pitfall:**
A common mistake is using a simple `.replace("the", "")` for article removal. This accidentally destroys words containing those substrings—turning "theater" into "ater" or "another" into "anoer". Always use word boundaries (`\b`) with regular expressions to target only standalone articles.

**Test it:** `python3 -m unittest test_exercises.TestNormalizeAnswer test_exercises.TestExactMatch -v`

---

## 2. Token-level F1

```python
# Pred: "red ball", GT: "blue ball"
# Common: "ball" (1)
# Precision: 1/2, Recall: 1/2 -> F1: 0.5
>>> token_level_f1("the red ball", "the blue ball")
0.5
```

Sometimes Exact Match is too harsh. If the ground truth is "the red and blue ball" and the model outputs "red and blue ball", they share almost all words. We want a metric that measures this overlap. SQuAD-style token-level F1 does exactly this by treating both strings as bags of words (tokens):
- **Precision**: The fraction of predicted tokens that are present in the ground truth.
- **Recall**: The fraction of ground truth tokens that are present in the prediction.
- **F1 Score**: The harmonic mean of precision and recall: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$.

If both strings are empty after normalization, F1 is defined as `1.0`. If only one is empty, F1 is `0.0`.

**The Pitfall:**
Using simple Python sets (`set(pred_tokens) & set(gt_tokens)`) to find overlapping tokens. If the prediction is "blue blue blue" and the ground truth is "blue", a set-based intersection says they match perfectly (100% precision and recall), which is wrong. You must use a multiset intersection (e.g., `collections.Counter`) to account for token frequencies.

**Test it:** `python3 -m unittest test_exercises.TestTokenLevelF1 -v`

---

## 3. Retrieval Metrics: Recall@K and MRR

```python
# Recall@2: retrieved[:2] is [1, 2]. GT is [2, 5]. Intersection is {2}. Recall = 1/2 = 0.5
>>> recall_at_k(retrieved_ids=[1, 2, 3], ground_truth_ids=[2, 5], k=2)
0.5
```

Before an LLM can generate a RAG response, your retrieval system must fetch the right context. We evaluate retrieval using two key metrics:
- **Recall@K**: Out of all the ground truth relevant documents, what fraction was successfully retrieved in the top $K$ positions? If the ground truth is empty, return `0.0`.
- **Mean Reciprocal Rank (MRR)**: Evaluates the position of the *first* relevant document. For each query, the Reciprocal Rank is $1 / \text{rank}$ of the first correct document (1-based index). If no relevant document is found in the retrieved list, the score is $0.0$. MRR is the average of these reciprocal ranks across all queries in a batch. If the batch is empty, return `0.0`.

**The Pitfall:**
For MRR, forgetting that rank is 1-based, not 0-based. If the first correct document is at index 0 of your retrieved list, its rank is 1, and its reciprocal rank is $1/1 = 1.0$. If you divide by the 0-based index, you'll get a division-by-zero error!

**Test it:** `python3 -m unittest test_exercises.TestRetrievalMetrics -v`

---

## 4. Pairwise Win Rates

```python
# Judgements: ["A", "tie", "B", "A"]
# Wins A: 2, Ties: 1, Total: 4
# Win Rate A = (2 + 0.5) / 4 = 0.625
>>> win_rate(["A", "tie", "B", "A"])
0.625
```

For complex generative tasks, automated metrics like F1 fail because there are many valid ways to write a response. The industry standard is "LLM-as-a-judge": prompting a powerful model (like GPT-4) to compare two outputs (Model A vs Model B) and declare a winner.

To compute a win rate for Model A, we must handle ties fairly. The standard formula used by platforms like LMSYS Chatbot Arena is:
$$\text{Win Rate}_A = \frac{\text{Wins}_A + 0.5 \times \text{Ties}}{\text{Total Judgements}}$$
This treats a tie as half a win for each model, ensuring the win rate is bounded between 0.0 and 1.0. If judgements is empty, return `0.0`.

**The Pitfall:**
Discarding ties from the denominator. If Model A wins 1 time, Model B wins 0 times, and there are 99 ties, discarding ties yields a 100% win rate for Model A. This is highly misleading! Keeping ties in the denominator correctly yields a win rate of $(1 + 49.5) / 100 = 50.5\%$, reflecting that the models are nearly identical.

**Test it:** `python3 -m unittest test_exercises.TestWinRate -v`

---

## 5. Bootstrapping and Confidence Intervals

```python
>>> import numpy as np
>>> data = [1, 2, 3, 4, 5]
>>> rng = np.random.default_rng(42)
>>> resample = rng.choice(data, size=len(data), replace=True)
>>> resample
array([1, 5, 3, 3, 5])
```

Suppose Model A has a win rate of 52% over Model B across 100 test cases. Is Model A actually better, or did it just get lucky with the sample selection? To answer this, we use **bootstrapping**: a non-parametric statistical method that estimates the uncertainty of a metric by repeatedly resampling our dataset *with replacement*.

The process:
1. Given a dataset of size $N$, sample $N$ items with replacement.
2. Compute the metric (e.g., win rate) on this resampled dataset.
3. Repeat this process $M$ times (e.g., 1000 resamples) to build a distribution of the metric.
4. Find the $\alpha/2$ and $1 - \alpha/2$ percentiles of this distribution to get the confidence interval (e.g., 2.5th and 97.5th percentiles for a 95% confidence interval).

We will use `numpy.percentile` to extract the bounds from our resampled distribution.

**The Pitfall:**
Using a non-seeded random number generator. If your bootstrapping function doesn't accept or use a seeded generator (`np.random.default_rng(seed)`), your confidence intervals will change slightly every time you run your evaluation pipeline. This makes tests flaky and results non-reproducible.

**Test it:** `python3 -m unittest test_exercises.TestBootstrapCI -v`

---

## The exercises

| Function | What it teaches |
|---|---|
| `normalize_answer` | SQuAD-style text normalization |
| `exact_match` | Strict equivalence check after normalization |
| `token_level_f1` | Multiset-based token overlap F1 score |
| `recall_at_k` | Evaluating retrieval coverage at a specific cutoff |
| `mean_reciprocal_rank` | Evaluating retrieval rank quality across queries |
| `win_rate` | Pairwise comparison score accounting for ties |
| `bootstrap_ci` | Non-parametric confidence intervals via resampling |

## Check your understanding

1. **Why do we normalize answers before computing Exact Match or F1?**
   <details><summary>Answer</summary>To prevent minor formatting differences (like capitalization, trailing punctuation, or optional articles like "the") from penalizing correct answers. This isolates semantic correctness from stylistic variation.</details>

2. **Why is set intersection incorrect for token-level F1?**
   <details><summary>Answer</summary>Set intersection ignores token frequency. If a prediction is "blue blue blue" and the ground truth is "blue", set intersection would treat them as a perfect match. Using a multiset intersection (like <code>collections.Counter</code>) correctly penalizes the redundant words.</details>

3. **What is the difference between Recall@K and MRR?**
   <details><summary>Answer</summary>Recall@K measures <em>coverage</em>—what fraction of relevant documents were found in the top K. MRR measures <em>ranking quality</em>—how high up the first relevant document was placed. A system can have high Recall@K but low MRR if the relevant documents are always placed at the very bottom of the retrieved list.</details>

4. **Why is discarding ties from pairwise win rates dangerous?**
   <details><summary>Answer</summary>Discarding ties artificially inflates the win rate of the slightly better model. If Model A wins 1 time, Model B wins 0 times, and they tie 99 times, discarding ties yields a 100% win rate for Model A. Keeping ties in the denominator yields 50.5%, which accurately reflects that they are nearly identical.</details>

5. **What does bootstrapping tell us that a single metric score cannot?**
   <details><summary>Answer</summary>Bootstrapping provides a confidence interval, which quantifies the statistical uncertainty of our metric. If the 95% confidence interval for Model A's win rate is [0.48, 0.56], the interval contains 0.50, meaning we cannot confidently say Model A is better than Model B.</details>

## Interview questions this module answers

- *How do you evaluate a Question Answering (QA) system when exact string matching is too strict?*
- *What metrics would you use to evaluate the retrieval component of a RAG system?*
- *How does Mean Reciprocal Rank (MRR) penalize poor search rankings compared to Recall@K?*
- *How do you calculate a win rate from pairwise LLM judgements, and how do you handle ties?*
- *How can you prove that a 2% improvement in model accuracy is statistically significant without making assumptions about the data's distribution?*
