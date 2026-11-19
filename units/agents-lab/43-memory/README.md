# 43 — Memory: windows, summaries and recall

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/agents-lab/43-memory/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/agents-lab/memory?utm_source=github&utm_medium=labs)

*~45 minutes · 20 tests*

LLMs are fundamentally stateless. Every time you send a prompt to an LLM, it has no memory of the previous prompt. To create the illusion of a continuous conversation, an AI agent must send the *entire* history of the conversation with every new request.

However, LLMs have a strict context window limit (e.g., 8k, 32k, or 128k tokens). Even if the window is large, sending massive histories is slow, expensive, and degrades the model's attention (the "lost in the middle" phenomenon). In this unit, you will build the core memory management systems used by production agent frameworks: token-based sliding windows, conversation summarization, and a local, zero-dependency semantic recall store using TF-IDF and NumPy.

Work in `exercises.py`. Run your tests using the commands provided at the end of each section.

## What you'll build

| Function / class | You'll implement | Shows up in real code as |
|---|---|---|
| `count_tokens` | A simple, fast word-based token estimator | Lightweight token budgeting before calling heavy API-based tokenizers |
| `trim_history` | A sliding window that preserves the system prompt and newest turns | Context window safety guardrails in production agent loops |
| `summarize_overflow` | Condensing dropped history into a single running summary message | Long-term conversational memory without losing historical context |
| `MemoryStore` | A local document store with TF-IDF cosine similarity search | A lightweight, zero-dependency vector-like database for agent recall |

---

## 1. Token estimation and counting

Before we can manage a budget, we must be able to measure consumption. In production, you would use a library like `tiktoken` or `sentencepiece` to count the exact tokens of a specific model. However, during local development, testing, or when building offline-first agents, a fast, word-based estimator is an excellent proxy.

```python
def count_tokens(text: str) -> int:
    if not text.strip():
        return 0
    return len(text.split())
```

**The Pitfall:** Many developers forget to handle empty strings, whitespace-only strings, or multiple consecutive spaces correctly. Simple character counts or naive splitting can lead to off-by-one errors that accumulate over long conversation histories, causing unexpected API budget overflows.

**In `exercises.py`:** Implement `count_tokens` to split text by whitespace and return the count. Ensure empty or whitespace-only strings return `0`.

**Test it:** `python3 -m unittest test_exercises.TestCountTokens -v`

---

## 2. Sliding windows and context budgets

When a conversation exceeds the token budget, the simplest strategy is a **sliding window**: discard the oldest messages to make room for new ones. However, we cannot simply slice the list from the left. 

Almost every agent relies on a **system prompt** (usually the first message) containing its instructions, tool definitions, and persona. If we discard the system prompt, the agent loses its identity and instructions. Therefore, a proper sliding window must:
1. Always preserve the system prompt (if present).
2. Fill the remaining budget with the *newest* messages from the end of the history.
3. Maintain the original chronological order of the kept messages.

```python
# Conceptual sliding window logic
system_message = messages[0]
remaining_budget = max_tokens - count_tokens(system_message)
kept_messages = []
for msg in reversed(messages[1:]):
    if count_tokens(msg) <= remaining_budget:
        kept_messages.append(msg)
        remaining_budget -= count_tokens(msg)
    else:
        break
return [system_message] + list(reversed(kept_messages))
```

**The Pitfall:** Scanning backwards and simply appending to a list reverses the conversation order. You must reverse the collected non-system messages back to their original chronological order before returning them. Additionally, if the system message itself exceeds the budget, you must raise a `ValueError` immediately rather than entering an invalid state.

**In `exercises.py`:** Implement `trim_history` using the provided specification.

**Test it:** `python3 -m unittest test_exercises.TestTrimHistory -v`

---

## 3. Summarizing overflow

While sliding windows prevent context crashes, they suffer from complete amnesia regarding dropped turns. To solve this, we can use **summarization memory**. 

Instead of simply discarding old messages, we pass the overflowing messages to a summarizer (which, in production, is a fast LLM call). We then insert a single summary message right after the system prompt, containing the condensed history of the conversation so far.

```
[System Prompt] -> Always kept
[Summary of turns 1-4] -> Generated from overflow
[Turn 5] -> Kept
[Turn 6] -> Kept
```

This keeps the total token count within the budget while preserving the semantic essence of the past.

**The Pitfall:** The summary message itself consumes tokens! When calculating how many of the newest messages can fit, you must account for the combined size of the system prompt *and* the newly generated summary message. If this combined "fixed overhead" exceeds the budget, the system cannot proceed and must raise a `ValueError`.

**In `exercises.py`:** Implement `summarize_overflow` to orchestrate this budget allocation.

**Test it:** `python3 -m unittest test_exercises.TestSummarizeOverflow -v`

---

## 4. Vector-like recall with TF-IDF

Sliding windows and summaries are great for conversational flow, but what if the agent needs to recall specific facts from hours or days ago? For this, we use **semantic recall** (Retrieval-Augmented Generation, or RAG).

While production systems use vector databases (like Chroma or Pinecone) with deep-learning embeddings, you can build a highly effective, zero-dependency, and fully deterministic semantic recall store using **TF-IDF** (Term Frequency-Inverse Document Frequency) and **NumPy**.

### The Math
1. **Vocabulary**: A sorted list of all unique lowercase words across all documents in the store.
2. **Term Frequency (TF)**: The raw count of a term in a document.
3. **Inverse Document Frequency (IDF)**: Measures how common or rare a word is across the entire corpus. We use the standard smoothed formula:
   $$\text{IDF}(t) = \ln\left(\frac{1 + N}{1 + \text{DF}(t)}\right) + 1$$
   where $N$ is the total number of documents, and $\text{DF}(t)$ is the number of documents containing term $t$.
4. **Cosine Similarity**: The dot product of two TF-IDF vectors divided by the product of their Euclidean norms:
   $$\text{sim}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$

**The Pitfall:** If a query or a document contains no words from the vocabulary, its vector will be all zeros. Attempting to calculate cosine similarity will result in a division-by-zero error. You must explicitly catch this and return `0.0` similarity.

**In `exercises.py`:** Implement the `MemoryStore` class using NumPy for vector operations.

**Test it:** `python3 -m unittest test_exercises.TestMemoryStore -v`

---

## The exercises

| Function / Class | What it teaches |
|---|---|
| `count_tokens` | Basic token estimation and input sanitization |
| `trim_history` | Sliding window context budgeting with system prompt preservation |
| `summarize_overflow` | Summarization-based memory management and overhead calculation |
| `MemoryStore` | TF-IDF vector math and cosine similarity ranking using NumPy |

## Check your understanding

1. **Why is a simple word-count estimator useful if it doesn't match the exact tokenizer of the LLM?**
   <details><summary>Answer</summary>It runs entirely offline, has zero dependencies, is extremely fast, and serves as a highly accurate relative proxy for budgeting during unit tests and local development.</details>

2. **In `trim_history`, why do we scan backwards from the end of the message list?**
   <details><summary>Answer</summary>Because the newest messages are the most relevant to the current conversation turn. Scanning backwards ensures we keep the most recent context while dropping the oldest.</details>

3. **What happens if the system prompt plus the summary message exceeds the total token budget?**
   <details><summary>Answer</summary>The agent cannot run because the minimum required context is larger than the allowed budget. The code must raise a <code>ValueError</code> to prevent silent failures or infinite loops.</details>

4. **Why do we use TF-IDF instead of simple keyword matching for the `MemoryStore`?**
   <details><summary>Answer</summary>TF-IDF down-weights common words (like "the" or "and") and up-weights rare, highly descriptive words (like "database" or "credentials"), leading to much more accurate semantic retrieval.</details>

5. **How does `MemoryStore` handle ties in similarity scores?**
   <details><summary>Answer</summary>By leveraging Python's stable sorting algorithm (Timsort), documents with identical similarity scores naturally preserve their original insertion order.</details>

## Interview questions this module answers

- *How do you prevent an AI agent from crashing due to context window limits?*
- *What is the difference between sliding window memory and summarization memory?*
- *How would you implement a lightweight vector search engine without using external databases or APIs?*
- *Why is the system prompt treated differently when trimming conversation history?*
- *How do you handle division-by-zero edge cases when calculating cosine similarity?*
