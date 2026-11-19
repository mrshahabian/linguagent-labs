# 50 — Decoding: greedy, temperature, top-k and top-p

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/llm-lab/50-decoding/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/llm-lab/decoding?utm_source=github&utm_medium=labs)

*~45 minutes · 20 tests · Advanced*

When a Large Language Model (LLM) processes a prompt, its final layer outputs a vector of raw, unnormalized scores called **logits**—one for every single token in its vocabulary. Turning these raw numbers into coherent, creative, and human-like text is the job of the **decoding pipeline**.

In this module, you will build the core components of modern LLM generation engines from scratch. You will implement the mathematical transformations that turn logits into probabilities, the filters that control creativity and prevent nonsense, and the search algorithms that plan multiple steps ahead. Understanding these mechanics is essential for optimizing inference engines, debugging repetitive outputs, and mastering the generation APIs of OpenAI, Hugging Face, or vLLM.

## What you'll build

You will implement seven core functions in `exercises.py` to build a complete, highly configurable decoding pipeline:

| Function / class | You'll implement | Shows up in real code as |
|---|---|---|
| `softmax_with_temperature` | Numerically stable softmax with temperature scaling | The `temperature` parameter in OpenAI / Hugging Face generation |
| `top_k_filter` | Truncating the probability distribution to the top $k$ tokens | The `top_k` generation parameter in vLLM / Hugging Face |
| `top_p_filter` | Nucleus sampling: keeping the smallest set of tokens summing to $p$ | The `top_p` generation parameter in OpenAI / Cohere |
| `apply_repetition_penalty` | Multiplicative penalty on logits of already generated tokens | The `repetition_penalty` parameter in Hugging Face / vLLM |
| `sample_next_token` | Drawing a token index using a seeded random generator | The stochastic sampling step in autoregressive generation |
| `greedy_decode` | Step-by-step argmax generation over a transition matrix | Fast, deterministic generation (`temperature=0`) |
| `beam_search` | Keeping track of multiple candidate paths to find the global optimum | Beam search decoding in translation and summarization |

---

## 1. Softmax with Temperature

The raw logits $z$ produced by an LLM are real numbers ranging from negative to positive infinity. To turn them into a valid probability distribution where each value is in $[0, 1]$ and all values sum to $1.0$, we use the **softmax** function:

$$P(x_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

To control the "creativity" or entropy of the distribution, we introduce a parameter called **temperature** ($T > 0$) that scales the logits before applying softmax:

$$P(x_i) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

- A high temperature ($T > 1.0$) flattens the distribution, making less likely tokens more probable (higher creativity, but higher risk of gibberish).
- A low temperature ($T < 1.0$) sharpens the distribution, making the most likely tokens even more dominant. As $T \to 0$, it approaches greedy selection.

```python
# Illustrative example of temperature scaling (not the solution)
import numpy as np

def simple_softmax(logits, T):
    scaled = logits / T
    return np.exp(scaled) / np.sum(np.exp(scaled))
```

### The Pitfall: Numerical Overflow
If a logit $z_i$ is large (e.g., $800.0$), $e^{800}$ will overflow to `inf` in 64-bit floating point math, resulting in `nan` values after division. To prevent this, we must subtract the maximum logit before exponentiating:

$$z'_i = \frac{z_i - \max(z)}{T}$$

Subtracting a constant from all inputs to softmax does not change the output probabilities, but it guarantees that the largest input is $0.0$, so $e^0 = 1.0$, completely eliminating overflow. Note that the division by $T$ must happen *before* subtracting the maximum, or we must scale the maximum too. The cleanest way is to scale the logits first: `scaled_logits = logits / temperature`, then subtract the maximum of `scaled_logits`.

**Test it:** `python3 -m unittest test_exercises.TestSoftmaxWithTemperature -v`

---

## 2. Top-K Filtering

Even with temperature scaling, the "long tail" of the vocabulary contains thousands of highly unlikely tokens. If we sample from the entire distribution, there is always a non-zero chance of picking a completely irrelevant token, leading to incoherent text.

**Top-K filtering** addresses this by keeping only the $k$ tokens with the highest probabilities, setting all other probabilities to $0.0$, and then renormalizing the remaining probabilities so they sum to $1.0$.

```python
# Illustrative example of top-k filtering (not the solution)
def simple_top_k(probs, k):
    threshold = sorted(probs, reverse=True)[k-1]
    filtered = np.where(probs >= threshold, probs, 0.0)
    return filtered / np.sum(filtered)
```

### The Pitfall: Renormalization and Ties
If you zero out the bottom elements, the remaining elements will sum to less than $1.0$. If you pass this unnormalized array to a sampling function, it will either raise an error or sample incorrectly. Always re-normalize!
Additionally, if there are ties at the boundary, a simple threshold check might keep more than $k$ elements. To be perfectly precise and deterministic, you should sort the indices of the probabilities in descending order, keep exactly the first $k$ indices, and set all other indices to $0.0$.

**Test it:** `python3 -m unittest test_exercises.TestTopKFilter -v`

---

## 3. Top-P (Nucleus) Filtering

While Top-K works well, a fixed $k$ can be problematic. If the model is highly confident, the top 1 token might have $99\%$ probability, and the next 9 tokens are junk—yet Top-K with $k=10$ still forces us to keep those 9 junk tokens. Conversely, if the model is highly uncertain, the probability might be spread evenly across 50 tokens, but Top-K with $k=10$ cuts off 40 viable candidates.

**Top-P (Nucleus) filtering** solves this by dynamically sizing the candidate pool. It keeps the smallest set of tokens whose cumulative probability exceeds or equals a threshold $p$ (where $0.0 < p \le 1.0$).

To implement this:
1. Sort the probabilities in descending order.
2. Compute the cumulative sum of the sorted probabilities.
3. Find the first index where the cumulative sum is $\ge p$.
4. Zero out all elements in the sorted array that come after this index.
5. Re-normalize the remaining probabilities.

```python
# Illustrative example of top-p filtering (not the solution)
def simple_top_p(probs, p):
    sorted_indices = np.argsort(probs)[::-1]
    cum_sum = np.cumsum(probs[sorted_indices])
    cutoff = np.where(cum_sum >= p)[0][0]
    # ... zero out and renormalize ...
```

### The Pitfall: Keeping At Least One Token
If $p$ is very small (e.g., $0.05$), we might end up keeping zero tokens if we aren't careful. We must always keep at least the single highest-probability token to ensure we have a valid token to sample.

**Test it:** `python3 -m unittest test_exercises.TestTopPFilter -v`

---

## 4. Repetition Penalty

Autoregressive models often get stuck in repetitive loops (e.g., repeating the same phrase over and over). To combat this, we can apply a **repetition penalty** (popularized by the CTRL paper) to the logits of tokens that have already been generated.

For each token index $i$ in the set of already generated tokens:
- If the logit $z_i$ is positive ($\ge 0$), we divide it by the penalty $\theta \ge 1.0$: $z_i \leftarrow z_i / \theta$.
- If the logit $z_i$ is negative ($< 0$), we multiply it by the penalty: $z_i \leftarrow z_i \times \theta$.

```python
# Illustrative example of repetition penalty (not the solution)
def simple_penalty(logits, history, penalty):
    for token in set(history):
        logits[token] /= penalty
    return logits
```

### The Pitfall: Negative Logits
Why do we multiply negative logits instead of dividing them?
If a logit is negative (e.g., $-5.0$), dividing it by a penalty of $2.0$ would make it $-2.5$, which is *larger* (closer to zero) than $-5.0$. This would actually *increase* the probability of generating that token! Multiplying it by $2.0$ makes it $-10.0$, which correctly suppresses it.
Also, make sure to apply the penalty only once per unique token in the generated history, regardless of how many times it has appeared.

**Test it:** `python3 -m unittest test_exercises.TestApplyRepetitionPenalty -v`

---

## 5. Sampling with a Generator

Once we have a clean, filtered, and normalized probability distribution, we need to draw a sample.
In production and testing, we must use a seeded random number generator to ensure reproducibility. In modern NumPy, this is done using `np.random.Generator` (created via `np.random.default_rng(seed)`).

```python
# Illustrative example of sampling (not the solution)
def simple_sample(probs, seed=42):
    rng = np.random.default_rng(seed)
    return rng.choice(len(probs), p=probs)
```

### The Pitfall: Global State
Using global state like `np.random.choice` makes your code non-thread-safe and hard to test. Always pass an explicit `Generator` instance and use its `.choice()` method.

**Test it:** `python3 -m unittest test_exercises.TestSampleNextToken -v`

---

## 6. Greedy Decoding

Greedy decoding is the simplest decoding strategy: at each step, we pick the token with the highest probability (or log probability).
While fast and simple, greedy decoding is short-sighted. It makes the optimal choice at each individual step, but this can lead to poor long-term sequences if a highly likely token is followed by only extremely low-probability tokens.

```python
# Illustrative example of greedy decoding (not the solution)
def simple_greedy(transition_matrix, start, length):
    seq = [start]
    for _ in range(length - 1):
        seq.append(np.argmax(transition_matrix[seq[-1]]))
    return seq
```

**Test it:** `python3 -m unittest test_exercises.TestGreedyDecode -v`

---

## 7. Beam Search

Beam search is a heuristic search algorithm that explores multiple paths simultaneously. Instead of keeping only the single best token at each step (greedy), it maintains a "beam" of the top $B$ (beam width) most likely sequences.

At each generation step:
1. Expand each of the $B$ active sequences to all possible next tokens.
2. Compute the cumulative log probability for each of these expanded sequences.
3. Sort all expanded sequences by their cumulative log probability in descending order.
4. Keep the top $B$ sequences as the new active beam.
5. Repeat until the target sequence length is reached.

```python
# Illustrative example of beam search (not the solution)
def simple_beam(transition_matrix, start, beam_width, length):
    candidates = [([start], 0.0)]
    # ... expand, score, sort, prune ...
```

### The Pitfall: Underflow
Always add log probabilities instead of multiplying raw probabilities. Multiplying many small probabilities quickly underflows to $0.0$ in floating-point representations.

**Test it:** `python3 -m unittest test_exercises.TestBeamSearch -v`

---

## The exercises

| Function | What it teaches |
|---|---|
| `softmax_with_temperature` | Temperature scaling and numerical stability |
| `top_k_filter` | Truncating the probability distribution to the top $k$ elements |
| `top_p_filter` | Nucleus sampling with dynamic candidate pool sizing |
| `apply_repetition_penalty` | CTRL-style repetition penalty for positive and negative logits |
| `sample_next_token` | Reproducible sampling using a seeded random generator |
| `greedy_decode` | Step-by-step argmax generation |
| `beam_search` | Multi-path search to find globally optimal sequences |

## Check your understanding

1. **Why do we divide logits by temperature before subtracting the maximum logit?**
   <details><summary>Answer</summary>Because subtraction of the maximum is a mathematical trick to prevent overflow in the exponentiation step of softmax. The actual mathematical function we want to compute is softmax on the scaled logits $z / T$. Therefore, the scaling must happen first, and then we subtract the maximum of those scaled logits to ensure the largest exponent is exactly $0.0$.</details>

2. **What happens to the output of `top_k_filter` if $k$ is greater than the vocabulary size?**
   <details><summary>Answer</summary>The filter should return the original probability distribution unmodified, as all tokens are within the top $k$ elements.</details>

3. **Why does `top_p_filter` need to keep at least one token even if $p$ is extremely small?**
   <details><summary>Answer</summary>If $p$ is smaller than the probability of the single most likely token, the cumulative sum would be empty if we strictly required elements *after* the sum exceeds $p$. We must keep at least the top token to ensure we have a valid probability distribution to sample from, rather than an empty set of options.</details>

4. **Why is dividing a negative logit by a repetition penalty incorrect?**
   <details><summary>Answer</summary>Dividing a negative number by a positive penalty $> 1.0$ makes it closer to zero (i.e., larger). For example, $-4.0 / 2.0 = -2.0$. This would actually *increase* the logit's value and make the token *more* likely to be generated, which is the opposite of a repetition penalty. Multiplying by the penalty makes it more negative (e.g., $-4.0 \times 2.0 = -8.0$), correctly suppressing it.</details>

5. **Why does beam search often produce better text than greedy decoding?**
   <details><summary>Answer</summary>Greedy decoding is short-sighted; it only looks one step ahead. Beam search keeps track of multiple promising paths, allowing it to choose a path that might have a slightly lower-probability first step but leads to much higher-probability steps later on.</details>

## Interview questions this module answers

- *How does temperature affect the creativity and coherence of an LLM's output?*
- *What is the difference between Top-K and Top-P (nucleus) sampling?*
- *How do you implement a numerically stable softmax function?*
- *How does the CTRL repetition penalty handle negative logits differently from positive ones?*
- *Why do we use log probabilities instead of raw probabilities in beam search?*
