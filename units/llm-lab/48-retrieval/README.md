# 48 — Retrieval: chunking, BM25 and hybrid search

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/llm-lab/48-retrieval/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/llm-lab/retrieval?utm_source=github&utm_medium=labs)

*~45 minutes · 25 tests · Advanced*

In production Retrieval-Augmented Generation (RAG) systems, the quality of your generation is strictly bounded by the quality of your retrieval. If your search pipeline returns irrelevant, truncated, or redundant text chunks, even the most powerful LLM will produce hallucinations or incomplete answers.

While vector search gets most of the hype, real-world search engines almost never rely on dense embeddings alone. Out-of-vocabulary terms, exact product IDs, serial numbers, and specific names are notoriously difficult for vector models to match. To build a production-grade search system, you must combine traditional lexical search (BM25) with vector search, fuse their results robustly, and clean up redundant chunks before feeding them to the LLM.

Work in `exercises.py`. Run your tests using the commands at the end of each section.

## What you'll build

You will implement the core components of a hybrid search and retrieval pipeline. This includes text preprocessing, ranking, rank fusion, and post-retrieval deduplication:

| Function / class | You'll implement | Shows up in real code as |
|---|---|---|
| `chunk_text` | Sliding-window word chunking with character-level source offsets | Document ingestion pipelines (e.g., LangChain, LlamaIndex, or custom chunkers) |
| `BM25Index` | Best Matching 25 scoring and search index with TF-IDF and length normalization | Lexical search engines (e.g., Elasticsearch, OpenSearch, `rank_bm25`) |
| `reciprocal_rank_fusion` | Rank-based fusion of multiple search result lists | Hybrid search coordinators combining sparse and dense retrieval |
| `dedupe_results` | Jaccard similarity-based near-duplicate filtering | Post-retrieval reranking and context compression steps |

---

## 1. Text Chunking with Character Offsets

Before documents can be indexed or searched, they must be broken down into manageable pieces called "chunks." If you chunk text purely by character count (e.g., every 500 characters), you will inevitably slice words, sentences, and code blocks in half, destroying their semantic meaning.

A better approach is **word-based chunking with overlap**. We split the text into individual words, group them into windows of a fixed size, and slide the window forward by a step size that allows for overlap between adjacent chunks. Crucially, we must also track the exact character offsets (`start` and `end` indices) of each chunk in the original text. This allows us to highlight matches or map retrieved chunks back to their precise source locations.

```python
# Example of word-based chunking with overlap
text = "The quick brown fox jumps over the lazy dog."
# Words: ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog."]
# If chunk_size=4 and overlap=2:
# Chunk 1: "The quick brown fox"
# Chunk 2: "brown fox jumps over"
# Chunk 3: "jumps over the lazy"
# Chunk 4: "the lazy dog."
```

### The Pitfall
When reconstructing the chunk text from the original string, naive implementations simply join the words with a single space. This discards the original formatting, extra whitespaces, and newlines of the source document. To preserve formatting, you should find the character start index of the first word in the chunk and the character end index of the last word in the chunk, and then slice the original string directly using `text[start:end]`.

**Test it:** `python3 -m unittest test_exercises.TestChunkText -v`

---

## 2. Lexical Search with BM25

BM25 (Best Matching 25) is the industry-standard algorithm for lexical (keyword) search. It improves upon classic TF-IDF by addressing two major issues: term frequency saturation and document length normalization.

1. **Term Frequency Saturation ($k_1$):** In simple TF-IDF, if a term appears 20 times in a document, it is scored nearly 20 times higher than if it appeared once. In reality, the difference between 1 occurrence and 2 occurrences is huge, but the difference between 19 and 20 is negligible. The parameter $k_1$ controls how quickly the term frequency score saturates.
2. **Document Length Normalization ($b$):** Long documents naturally contain more words, which increases the chance of keyword matches. The parameter $b$ penalizes long documents, scaling the penalty based on how much longer the document is compared to the average document length across the entire corpus.

The BM25 score for a document $D$ given a query $Q$ is calculated as:

$$\text{score}(D, Q) = \sum_{q \in Q} \text{IDF}(q) \cdot \frac{f(q, D) \cdot (k_1 + 1)}{f(q, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

Where:
- $f(q, D)$ is the frequency of query term $q$ in document $D$.
- $|D|$ is the length of document $D$ in tokens.
- $\text{avgdl}$ is the average document length across the corpus.
- $\text{IDF}(q)$ is the Inverse Document Frequency, calculated as:

$$\text{IDF}(q) = \ln\left(1 + \frac{N - n(q) + 0.5}{n(q) + 0.5}\right)$$

where $N$ is the total number of documents in the corpus, and $n(q)$ is the number of documents containing term $q$.

```python
# Tokenization strategy: lowercase and extract alphanumeric words
import re
def tokenize(text):
    return re.findall(r'\w+', text.lower())
```

### The Pitfall
When calculating BM25, ensure that query terms not present anywhere in the corpus are handled gracefully (they should contribute `0` to the score). Additionally, when sorting search results, you must filter out documents with a score of `0.0` or less, and sort ties deterministically (e.g., by document index) to prevent flaky behavior in production pipelines.

**Test it:** `python3 -m unittest test_exercises.TestBM25Index -v`

---

## 3. Reciprocal Rank Fusion (RRF)

In a hybrid search system, you run both a lexical search (BM25) and a vector search (using cosine similarity on dense embeddings). This leaves you with two completely different lists of ranked documents. How do you combine them?

You cannot simply add or average their raw scores because they are on completely different scales: BM25 scores are unbounded positive numbers, while cosine similarities range from -1 to 1. 

**Reciprocal Rank Fusion (RRF)** solves this by ignoring raw scores entirely and focusing solely on the **rank** of each document in each list. The RRF score for an item $d$ is defined as:

$$RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$

Where:
- $M$ is the set of ranking systems (e.g., BM25 and Vector Search).
- $r_m(d)$ is the 1-based rank of item $d$ in ranking system $m$.
- $k$ is a constant (typically set to `60`) that dampens the influence of high rankings, ensuring that a single extremely high rank doesn't completely overwhelm moderately high ranks across multiple systems.

```python
# Example of RRF calculation with k = 60
ranking_1 = ["doc_a", "doc_b"]
ranking_2 = ["doc_b", "doc_c"]

# doc_a: rank 1 in system 1, absent in system 2
# score(doc_a) = 1 / (60 + 1) = 1/61

# doc_b: rank 2 in system 1, rank 1 in system 2
# score(doc_b) = 1 / (60 + 2) + 1 / (60 + 1) = 1/62 + 1/61
```

### The Pitfall
Remember that ranks are **1-based** ($1, 2, 3, \dots$), not 0-based. Using 0-based indices in the denominator will skew the math and can cause division-by-zero errors if $k=0$.

**Test it:** `python3 -m unittest test_exercises.TestRRF -v`

---

## 4. Near-Duplicate Deduplication

When retrieving chunks from a large corpus, especially with overlapping sliding windows, your search engine will often return highly redundant chunks. For example, if Chunk A is "The quick brown fox jumps over the lazy dog" and Chunk B is "quick brown fox jumps over the lazy dog. He", sending both to an LLM wastes context tokens and can confuse the model.

To prevent this, we perform post-retrieval **deduplication** using **Jaccard Similarity**. The Jaccard similarity between two token sets $A$ and $B$ is the size of their intersection divided by the size of their union:

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

We iterate through the retrieved chunks in order of relevance. For each chunk, we compute its Jaccard similarity with all chunks we have *already accepted*. If the similarity meets or exceeds a specified `threshold` (e.g., `0.5`), we discard the chunk as a near-duplicate. Otherwise, we keep it.

```python
# Jaccard similarity example
set_a = {"the", "quick", "brown", "fox"}
set_b = {"quick", "brown", "fox", "jumps"}
# Intersection: {"quick", "brown", "fox"} (size 3)
# Union: {"the", "quick", "brown", "fox", "jumps"} (size 5)
# Jaccard = 3 / 5 = 0.6
```

### The Pitfall
Always compare the candidate chunk against the *already accepted* list of chunks, not the original unfiltered list. Comparing against the original list would cause you to discard chunks that are similar to other discarded chunks, leading to over-filtering.

**Test it:** `python3 -m unittest test_exercises.TestDedupeResults -v`

---

## The exercises

| Function | What it teaches |
|---|---|
| `chunk_text` | Word-based sliding-window chunking with exact character offsets |
| `BM25Index` | Lexical indexing, TF-IDF, document length normalization, and search ranking |
| `reciprocal_rank_fusion` | Combining disparate search rankings using rank-based scoring |
| `dedupe_results` | Filtering out redundant context using Jaccard similarity on token sets |

## Check your understanding

1. **Why is word-based chunking with character offsets preferred over simple character-based slicing?**
   <details><summary>Answer</summary>Character-based slicing cuts words and sentences in half, destroying semantic meaning. Word-based chunking preserves word boundaries. Keeping character offsets is essential because it allows the system to map the chunk back to its exact location in the original document for highlighting or citation.</details>

2. **What role does the parameter $b$ play in BM25?**
   <details><summary>Answer</summary>The parameter $b$ controls document length normalization. If $b = 1.0$, the algorithm fully penalizes documents for being longer than the average document length. If $b = 0.0$, length normalization is completely turned off, meaning long documents with high term counts are not penalized.</details>

3. **Why can't we just add raw BM25 scores and vector cosine similarity scores together in hybrid search?**
   <details><summary>Answer</summary>They are on completely different scales. Cosine similarity is bounded between -1.0 and 1.0 (often 0.0 to 1.0 in text embeddings), while BM25 scores are positive and theoretically unbounded. Adding them directly would allow the BM25 score to completely dominate the vector score.</details>

4. **What is the purpose of the constant $k$ in Reciprocal Rank Fusion?**
   <details><summary>Answer</summary>The constant $k$ (usually 60) dampens the impact of top-ranked items. Without $k$, an item ranked 1st in one list and nowhere in another would get a score of $1/1 = 1.0$, which would completely overpower an item ranked 2nd and 3rd in two different lists ($1/2 + 1/3 = 0.83$). With $k=60$, the difference between rank 1 and rank 2 is much smaller ($1/61 \approx 0.0164$ vs $1/62 \approx 0.0161$), allowing consensus across multiple rankings to win.</details>

5. **What happens to Jaccard similarity if both token sets are empty?**
   <details><summary>Answer</summary>If both token sets are empty, the union is empty, which would lead to a division-by-zero error. In this case, the Jaccard similarity should be defined as 0.0.</details>

## Interview questions this module answers

- *How does BM25 improve upon standard TF-IDF?*
- *What is Reciprocal Rank Fusion (RRF), and why is it used in hybrid search systems?*
- *How do you handle near-duplicate search results in a RAG pipeline without calling an LLM?*
- *How do you implement a sliding-window text chunker that preserves original document formatting?*
