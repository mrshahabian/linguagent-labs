# 47 — Embeddings and similarity search

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/llm-lab/47-embeddings-similarity/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/llm-lab/embeddings-similarity?utm_source=github&utm_medium=labs)

*~45 minutes · 25 tests · Advanced*

In modern LLM applications, text is rarely matched using raw string comparisons. Instead, we represent text as dense or sparse vectors (embeddings) and find the most relevant pieces of information using vector similarity. This is the core engine behind Retrieval-Augmented Generation (RAG), semantic search, and classification systems.

While production systems rely on highly optimized libraries like `tiktoken`, `FAISS`, or `scikit-learn`'s `TfidfVectorizer`, understanding the mathematical and structural mechanics under the hood is crucial. If you do not know how vector normalization, matrix broadcasting, stable tie-breaking, and attention-masked pooling work, you will build retrieval pipelines that suffer from silent bugs—such as NaN propagation, slow Python-loop bottlenecks, and biased search results.

In this unit, you will build a complete vector search and text embedding pipeline from first principles using NumPy. You will implement safe L2 normalization, batch cosine similarity, fast top-k retrieval with stable tie-breaking, a TF-IDF sparse embedder, and attention-masked mean pooling for dense transformer outputs.

## What you'll build

Five core functions and classes in `exercises.py`, in this order. Keep this table open as you go:

| Function / class | You'll implement | Shows up in real code as |
|---|---|---|
| `l2_normalize` | Safe L2 normalization (handling zero vectors) | Preventing division-by-zero errors when handling empty or fully-masked text |
| `cosine_similarity` | Pairwise similarity matrix via NumPy broadcasting | Computing similarity scores across a batch of queries and documents efficiently |
| `top_k` | Fast, stable top-k retrieval using `argpartition` | Finding the most relevant documents in a vector database (e.g., FAISS) |
| `TfidfEmbedder` | Sparse TF-IDF vectorizer (fit and transform) | Standard baseline for keyword-based search and sparse retrieval |
| `mean_pool` | Attention-masked mean pooling | Converting token-level transformer outputs into a single sentence embedding |

---

## 1. Safe L2 normalization

To compute the cosine similarity between two vectors, we divide their dot product by the product of their L2 norms:

$$\cos(\theta) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$

If we normalize our vectors to have an L2 norm of exactly $1.0$ beforehand, the cosine similarity simplifies to a simple dot product: $u \cdot v$. This makes search incredibly fast. However, if a vector is entirely zeros (for example, an empty document or a document containing only out-of-vocabulary words), its L2 norm is $0$. Dividing by $0$ results in `NaN` values, which will propagate through your entire system and crash downstream models.

```python
>>> import numpy as np
>>> v = np.array([0.0, 0.0])
>>> norm = np.linalg.norm(v)
>>> v / norm
RuntimeWarning: invalid value encountered in divide
array([nan, nan])
```

To prevent this, we must implement a safe L2 normalization function. If the L2 norm of a vector is below a small threshold (epsilon), we should return a vector of all zeros instead of dividing by zero.

**In `exercises.py`:** Implement `l2_normalize` to handle both 1D vectors and 2D matrices (normalizing each row independently) without raising warnings or producing `NaN`s.

**Test it:** `python3 -m unittest test_exercises.TestL2Normalize -v`

---

## 2. Pairwise cosine similarity with NumPy broadcasting

In production retrieval, you often have a batch of $M$ query vectors and a batch of $N$ document vectors, each of dimension $D$. Writing nested Python loops to compute similarity between every query and every document is incredibly slow and will bottleneck your application.

Instead, we leverage NumPy's underlying C-optimized matrix multiplication. If we represent our queries as a matrix $Q$ of shape $(M, D)$ and our documents as a matrix $D_{oc}$ of shape $(N, D)$, and both matrices are L2-normalized along their rows, we can compute all pairwise cosine similarities in a single operation:

$$\text{Similarities} = Q \cdot D_{oc}^T$$

The resulting matrix has shape $(M, N)$, where the element at index $(i, j)$ is the cosine similarity between query $i$ and document $j$.

```python
>>> Q = np.array([[1.0, 0.0], [0.0, 1.0]])  # 2 queries
>>> Doc = np.array([[1.0, 0.0], [0.8, 0.6]]) # 2 documents
>>> np.dot(Q, Doc.T)
array([[1. , 0.8],
       [0. , 0.6]])
```

**In `exercises.py`:** Implement `cosine_similarity` to accept two 2D matrices, normalize them safely, and return their pairwise similarity matrix.

**Test it:** `python3 -m unittest test_exercises.TestCosineSimilarity -v`

---

## 3. Fast, stable top-k retrieval with argpartition

Once we have computed similarity scores, we need to find the top $k$ highest-scoring documents. The naive approach is to sort the entire array of scores in descending order and take the first $k$ elements. However, sorting $N$ elements takes $O(N \log N)$ time.

When $N$ is large (e.g., millions of documents), this is highly inefficient. We only care about the top $k$ elements; we do not care about the relative order of the remaining $N - k$ elements. NumPy provides `np.argpartition`, which partitions an array around the $k$-th element in $O(N)$ linear time. 

Another critical requirement is **stable tie-breaking**. If two documents have the exact same similarity score, we must break the tie deterministically (e.g., by preferring the document with the smaller index). `np.argpartition` does not guarantee stable sorting or even that the top $k$ elements are returned in sorted order. We must partition first, and then sort *only* the top $k$ elements to return them in descending order of similarity with stable tie-breaking.

```python
>>> scores = np.array([10.0, 30.0, 20.0, 30.0])
>>> # We want top 2. The indices with highest scores are 1 and 3 (both have score 30.0).
>>> # A stable tie-breaker should return indices [1, 3] in that order.
```

**In `exercises.py`:** Implement `top_k` using `np.argpartition` and a structured array or compound sorting key to ensure $O(N)$ average complexity and stable tie-breaking.

**Test it:** `python3 -m unittest test_exercises.TestTopK -v`

---

## 4. Sparse embeddings: TF-IDF from scratch

Before dense neural embeddings became popular, Term Frequency-Inverse Document Frequency (TF-IDF) was the gold standard for text retrieval, and it remains an excellent, highly interpretable baseline.

TF-IDF represents a document as a sparse vector where each dimension corresponds to a unique word in the vocabulary. 
1. **Term Frequency (TF)**: The raw count of a term $t$ in a document $d$.
2. **Inverse Document Frequency (IDF)**: Measures how common or rare a term is across the entire corpus of $N$ documents. We use the smoothed formulation (similar to scikit-learn):

$$\text{IDF}(t) = \log \left( \frac{1 + N}{1 + \text{DF}(t)} \right) + 1$$

where $\text{DF}(t)$ is the number of documents containing term $t$, and $\log$ is the natural logarithm.

To build a `TfidfEmbedder`:
- During `fit`, you tokenize the corpus, build an alphabetically sorted vocabulary of unique words, and compute the IDF value for each word.
- During `transform`, you compute the TF-IDF vector for each document and L2-normalize the resulting vectors.

To keep tokenization simple and deterministic, we define a token as any sequence of alphanumeric characters, converted to lowercase. We can extract tokens using the regular expression `re.findall(r'\w+', text.lower())`.

**In `exercises.py`:** Implement the `TfidfEmbedder` class with `fit` and `transform` methods.

**Test it:** `python3 -m unittest test_exercises.TestTfidfEmbedder -v`

---

## 5. Mean pooling with an attention mask

When using dense transformer models (like BERT or RoBERTa) to generate sentence embeddings, the model outputs a tensor of token-level embeddings of shape `(batch_size, sequence_length, hidden_dimension)`. To get a single vector representing the entire sentence, we must pool these token embeddings.

A simple average over the sequence length dimension is incorrect because sentences in a batch are padded with dummy tokens to match the length of the longest sentence. Including these padding tokens in the average will corrupt the embedding.

Instead, we must perform **attention-masked mean pooling**. We use an `attention_mask` of shape `(batch_size, sequence_length)` containing $1$s for real tokens and $0$s for padding tokens. We multiply each token embedding by its corresponding mask value, sum them up, and divide by the actual number of non-padded tokens in that sentence.

```python
>>> # If a sentence is completely empty or fully padded, the sum of its mask is 0.
>>> # Your pooling must handle this edge case safely without returning NaNs!
```

**In `exercises.py`:** Implement `mean_pool` to perform masked pooling over a batch of token embeddings.

**Test it:** `python3 -m unittest test_exercises.TestMeanPool -v`

---

## The exercises

| Function / Class | What it teaches |
|---|---|
| `l2_normalize` | How to safely normalize vectors/matrices without division-by-zero errors |
| `cosine_similarity` | Vectorized pairwise similarity computation using NumPy matrix operations |
| `top_k` | High-performance retrieval with $O(N)$ partitioning and stable tie-breaking |
| `TfidfEmbedder` | Building a sparse text representation pipeline from scratch |
| `mean_pool` | Correctly pooling transformer token embeddings using attention masks |

## Check your understanding

1. **Why is dividing by `np.linalg.norm(v)` directly dangerous in production?**
   <details><summary>Answer</summary>If a document contains only stop words that are filtered out, or is empty, its vector representation will be all zeros. The norm of a zero vector is <code>0</code>. Dividing by <code>0</code> produces <code>NaN</code> values, which propagate through downstream operations (like dot products) and turn all subsequent calculations into <code>NaN</code>.</details>

2. **How does pre-normalizing vectors make similarity search faster?**
   <details><summary>Answer</summary>Cosine similarity is defined as the dot product divided by the product of the norms. If we pre-normalize all vectors to have an L2 norm of <code>1.0</code>, the denominator is always <code>1.0</code>. This allows us to compute cosine similarity using only a single matrix multiplication (dot product), which is highly optimized in modern hardware.</details>

3. **Why is `np.argpartition` preferred over `np.argsort` for finding the top-k elements?**
   <details><summary>Answer</summary><code>np.argsort</code> performs a full sort of the array, taking $O(N \log N)$ time. <code>np.argpartition</code> only guarantees that the top $k$ elements are moved to the front of the array, which can be done in $O(N)$ linear time. This is significantly faster when $N$ is large and $k$ is small.</details>

4. **What is the purpose of smoothing in the IDF formula?**
   <details><summary>Answer</summary>The terms <code>1 + N</code> and <code>1 + DF(t)</code> prevent division-by-zero errors if a term in the test set never appeared in the training corpus (meaning $DF(t) = 0$). It also ensures that terms that appear in every single document do not get an IDF score of <code>0</code>, which would completely erase their term frequency contribution.</details>

5. **What happens if you perform mean pooling without an attention mask on padded sequences?**
   <details><summary>Answer</summary>The padding tokens (which are usually represented by zero vectors or arbitrary vectors) will be included in the average. This drags the final sentence embedding toward the padding token's embedding, heavily biasing the representation based on the amount of padding rather than the actual semantic content of the text.</details>

## Interview questions this module answers

- *How do you implement cosine similarity efficiently for a batch of queries and documents?*
- *What is the time complexity of finding the top-k nearest neighbors, and how can you optimize it?*
- *How do you handle zero-vector division errors in vector normalization?*
- *What is the mathematical formulation of TF-IDF, and how do you handle out-of-vocabulary terms during inference?*
- *How do you pool token-level embeddings from a transformer model to get a single sentence representation?*
