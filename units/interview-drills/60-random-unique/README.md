# 60 — Random unique numbers

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/interview-drills/60-random-unique/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/interview-drills/random-unique?utm_source=github&utm_medium=labs)

*~45 minutes · 21 tests · Interview*

Generating random numbers is easy, but generating *unique* random numbers fairly and efficiently is surprisingly tricky. In production systems, you often need to sample a subset of items without duplicates—whether you're selecting a random group of users for an A/B test, shuffling a deck of cards in a multiplayer game, or drawing a representative sample from a massive stream of incoming logs.

In interviews, questions about randomness test your understanding of probability, space-time complexity trade-offs, and clever data structures. A naive approach that works for a small range of numbers will often crash, run out of memory, or hang indefinitely when scaled up. This module covers the core algorithms used to solve these problems under strict resource constraints.

## What you'll build

| Function / class | You'll implement | Shows up in real code as |
|---|---|---|
| `sample_without_replacement` | Partial Fisher-Yates shuffle | Drawing a small subset from a known collection without modifying the original. |
| `reservoir_sample` | Reservoir sampling | Sampling a fixed-size subset from an infinite or unknown stream of data. |
| `UniqueRandomGenerator` | Lazy swap map (Fisher-Yates) | Generating unique IDs or game events on-the-fly over a massive range with $O(1)$ memory per draw. |
| `weighted_choice` | Cumulative weights + binary search | Selecting items with non-uniform probabilities, such as game loot drops or load-balancing requests. |

---

## 1. Partial Fisher-Yates Shuffle

If you need to select $k$ unique items from a list of size $n$, a naive approach is to shuffle the entire list and take the first $k$ elements:

```python
# Naive approach: O(n) time and O(n) space
import random

def naive_sample(lst, k):
    shuffled = list(lst)
    random.shuffle(shuffled)  # Shuffles all n elements
    return shuffled[:k]
```

**The Pitfall:** If $n = 1,000,000$ and $k = 5$, shuffling the entire list is incredibly wasteful. It performs $1,000,000$ swaps when we only needed $5$. 

To optimize this, we can perform a **partial Fisher-Yates shuffle**. Instead of shuffling the entire array, we only run the shuffle loop $k$ times. In each iteration $i$ (from $0$ to $k-1$), we select a random index $j$ from $i$ to $n-1$ and swap the elements at $i$ and $j$. This guarantees that the first $k$ elements are a fair, random, and unique sample of the entire list, completed in $O(k)$ time.

**In `exercises.py`:** Implement `sample_without_replacement` using this partial shuffle strategy. Make sure to copy the input list first to avoid mutating the caller's data.

**Test it:** `python3 -m unittest test_exercises.TestSampleWithoutReplacement -v`

---

## 2. Reservoir Sampling

What if you need to sample $k$ unique items from a stream of data, but you don't know the total number of items in advance, or the stream is too large to fit in memory? 

```python
# Naive approach: Blows up memory if the iterator is huge
def naive_stream_sample(iterator, k):
    all_items = list(iterator)  # O(n) memory!
    return random.sample(all_items, k)
```

**The Pitfall:** If you are sampling log lines from a running server, you cannot read all lines into memory. You need an algorithm that processes items one-by-one, keeping only $k$ items in memory at any time, while ensuring every item in the stream has an equal probability of being selected.

**Reservoir Sampling** solves this beautifully:
1. Keep a list (the "reservoir") of the first $k$ items from the stream.
2. For every subsequent item at index $i$ (where $i$ is 0-indexed, so $i \ge k$):
   - Choose a random integer $j$ between $0$ and $i$ inclusive.
   - If $j < k$, replace `reservoir[j]` with the new item.
   - Otherwise, discard the new item.

By the end of the stream, every item has had an exact $k/n$ chance of ending up in the reservoir.

**In `exercises.py`:** Implement `reservoir_sample` to sample $k$ items from an arbitrary iterator.

**Test it:** `python3 -m unittest test_exercises.TestReservoirSample -v`

---

## 3. Lazy Swap Map (UniqueRandomGenerator)

Suppose you need to generate unique random integers from $0$ to $n-1$, one at a time, until the caller stops or all $n$ numbers are exhausted. If $n = 10^9$, you cannot allocate an array of size $n$ to perform a Fisher-Yates shuffle, nor can you keep a `set` of seen numbers (as the set grows, finding an unseen number takes more and more random guesses, eventually hanging when the set is nearly full).

We can solve this by combining the Fisher-Yates shuffle with a **lazy swap map** (a hash map). Instead of allocating a physical array of size $n$, we use a dictionary to track only the indices that have been swapped. 

To draw the $i$-th unique number:
1. Pick a random index $j$ in the range $[i, n-1]$.
2. Look up the values currently associated with indices $i$ and $j$. If an index is not in our dictionary, its value is implicitly itself (i.e., $i$ or $j$).
3. Swap these values in our dictionary: the value at $j$ becomes the old value of $i$, and the value at $i$ becomes the old value of $j$.
4. Return the new value at $i$.

This gives us $O(1)$ time per draw and $O(d)$ space, where $d$ is the number of draws actually made. If we only draw $5$ numbers out of a billion, we only store a few entries in our dictionary!

**In `exercises.py`:** Implement the `UniqueRandomGenerator` class using this lazy swap map strategy.

**Test it:** `python3 -m unittest test_exercises.TestUniqueRandomGenerator -v`

---

## 4. Weighted Choice with Binary Search

Sometimes, choices should not be equally likely. For example, in a game, a common item might have a weight of $70$, a rare item $25$, and a legendary item $5$. 

```python
# Naive approach: O(n) per selection
def naive_weighted_choice(weights):
    total = sum(weights)
    r = random.uniform(0, total)
    running_sum = 0
    for idx, w in enumerate(weights):
        running_sum += w
        if r < running_sum:
            return idx
```

**The Pitfall:** While a linear scan is fine for a few items, it becomes a bottleneck ($O(n)$ time) if you have thousands of items and need to make millions of selections.

We can optimize this to $O(\log n)$ per selection using **cumulative weights** (prefix sums) and **binary search**:
1. Precompute the cumulative weights: `[70, 70+25, 70+25+5]` $\rightarrow$ `[70, 95, 100]`.
2. Generate a random float $r$ in the range $[0, 100)$.
3. Use binary search (`bisect.bisect_right`) to find where $r$ fits in the cumulative weights array. The resulting index is our selected item.

**In `exercises.py`:** Implement `weighted_choice` using cumulative weights and Python's built-in `bisect` module.

**Test it:** `python3 -m unittest test_exercises.TestWeightedChoice -v`

---

## The exercises

| Function / Class | What it teaches |
|---|---|
| `sample_without_replacement` | Partial Fisher-Yates shuffle for $O(k)$ time sampling. |
| `reservoir_sample` | Streaming selection over an iterator with $O(k)$ space. |
| `UniqueRandomGenerator` | Lazy swap map for $O(1)$ time and $O(d)$ space unique draws over huge ranges. |
| `weighted_choice` | $O(\log n)$ selection using prefix sums and binary search. |

## Check your understanding

1. **Why is a partial Fisher-Yates shuffle better than `random.sample` when $k$ is very small relative to $n$?**
   <details><summary>Answer</summary>If the input is already a list, <code>random.sample</code> actually uses a similar set-based or shuffle-based optimization internally depending on the size. However, implementing it yourself via a partial Fisher-Yates shuffle demonstrates how to achieve exactly $O(k)$ swaps and $O(1)$ auxiliary space (if mutating in place) or $O(n)$ space (if copying), avoiding the overhead of tracking a set of seen indices.</details>

2. **In reservoir sampling, why does the $i$-th item have a $k/(i+1)$ chance of being placed in the reservoir?**
   <details><summary>Answer</summary>At step $i$ (using 0-based indexing, so the stream has seen $i+1$ items), we choose a random index $j$ from $0$ to $i$. The probability that $j < k$ is exactly $k / (i+1)$. If $j < k$, we place the item in the reservoir at index $j$. Thus, the probability of the new item entering the reservoir is exactly $k / (i+1)$. Mathematical induction proves that this keeps the overall probability of selection for all previous items perfectly uniform at $k / (i+1)$ as well.</details>

3. **How does the lazy swap map in `UniqueRandomGenerator` avoid allocating $O(n)$ memory?**
   <details><summary>Answer</summary>Instead of creating an array of size $n$, we use a dictionary to store only the indices that have been modified by a swap. If an index is not in the dictionary, we treat its value as being equal to its index (e.g., index 500 holds value 500). This allows us to simulate a full Fisher-Yates shuffle while only consuming memory proportional to the number of elements actually drawn.</details>

4. **Why do we use `bisect.bisect_right` instead of `bisect.bisect_left` in `weighted_choice`?**
   <details><summary>Answer</summary>If our cumulative weights are <code>[10, 30, 60]</code> and our random value $r$ is exactly <code>10.0</code>, <code>bisect_left</code> would return index <code>0</code>, whereas the range for index <code>0</code> is $[0, 10)$. The range for index <code>1</code> is $[10, 30)$. Using <code>bisect_right</code> correctly places $r = 10.0$ at index <code>1</code>, ensuring correct boundary behavior.</details>

## Interview questions this module answers

- *How would you randomly select $k$ items from a stream of unknown size using constant memory?*
- *How can you generate unique random numbers up to 1 billion without running out of memory or looping indefinitely?*
- *How do you implement a fair weighted random choice algorithm that runs in logarithmic time?*
- *What is the Fisher-Yates shuffle, and how can it be optimized if we only need a few elements?*
