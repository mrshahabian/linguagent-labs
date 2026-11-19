"""Reference solutions for 47 — Embeddings and similarity search."""

import math
import re
import numpy as np
from typing import List, Tuple, Dict


def l2_normalize(v: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if v.ndim == 1:
        norm = np.linalg.norm(v)
        if norm < epsilon:
            return np.zeros_like(v)
        return v / norm
    elif v.ndim == 2:
        norm = np.linalg.norm(v, axis=-1, keepdims=True)
        safe_norm = np.where(norm < epsilon, 1.0, norm)
        normalized = v / safe_norm
        return np.where(norm < epsilon, 0.0, normalized)
    else:
        raise ValueError("Only 1D and 2D arrays are supported.")


def cosine_similarity(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    X_norm = l2_normalize(X)
    Y_norm = l2_normalize(Y)
    return np.dot(X_norm, Y_norm.T)


def top_k(scores: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    k = min(k, n)

    # Use a structured array to ensure stable sorting/partitioning.
    # We want descending order of scores (so we use negative scores for ascending sort)
    # and ascending order of indices for stable tie-breaking.
    dtype = [('neg_score', 'f8'), ('index', 'i8')]
    arr = np.empty(n, dtype=dtype)
    arr['neg_score'] = -scores
    arr['index'] = np.arange(n)

    # Partition to find the top k elements in O(N) time
    partitioned_indices = np.argpartition(arr, k - 1, order=['neg_score', 'index'])[:k]

    # Sort only the top k elements
    top_k_arr = arr[partitioned_indices]
    sorted_partition_indices = np.argsort(top_k_arr, order=['neg_score', 'index'])
    final_indices = partitioned_indices[sorted_partition_indices]
    
    return final_indices, scores[final_indices]


class TfidfEmbedder:
    def __init__(self):
        self.vocabulary_: Dict[str, int] = {}
        self.idf_: np.ndarray = np.array([])

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def fit(self, corpus: List[str]) -> 'TfidfEmbedder':
        tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        
        # Build vocabulary
        unique_terms = sorted(list(set(term for doc in tokenized_corpus for term in doc)))
        self.vocabulary_ = {term: idx for idx, term in enumerate(unique_terms)}
        
        # Compute IDF
        n_docs = len(corpus)
        idf_list = []
        for term in unique_terms:
            df = sum(1 for doc in tokenized_corpus if term in doc)
            idf = math.log((1 + n_docs) / (1 + df)) + 1
            idf_list.append(idf)
            
        self.idf_ = np.array(idf_list, dtype=float)
        return self

    def transform(self, corpus: List[str]) -> np.ndarray:
        n_docs = len(corpus)
        v_size = len(self.vocabulary_)
        features = np.zeros((n_docs, v_size), dtype=float)
        
        for doc_idx, doc in enumerate(corpus):
            tokens = self._tokenize(doc)
            for token in tokens:
                if token in self.vocabulary_:
                    col_idx = self.vocabulary_[token]
                    features[doc_idx, col_idx] += 1.0
                    
        # Multiply by IDF
        features = features * self.idf_
        
        # L2 normalize
        return l2_normalize(features)


def mean_pool(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    token_embeddings = np.asarray(token_embeddings, dtype=float)
    attention_mask = np.asarray(attention_mask, dtype=float)
    
    # Expand mask to (B, S, 1)
    mask_expanded = np.expand_dims(attention_mask, axis=-1)
    
    # Weighted sum of embeddings
    weighted_sum = np.sum(token_embeddings * mask_expanded, axis=1)
    
    # Sum of mask weights
    sum_mask = np.sum(mask_expanded, axis=1)
    
    # Avoid division by zero
    safe_sum_mask = np.where(sum_mask == 0, 1.0, sum_mask)
    pooled = weighted_sum / safe_sum_mask
    
    # Set fully padded rows to zero vector
    return np.where(sum_mask == 0, 0.0, pooled)
