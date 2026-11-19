"""48 — Retrieval: chunking, BM25 and hybrid search

Reference implementations for text chunking, BM25, RRF, and Jaccard deduplication.
"""

import math
import re
from collections import Counter
from typing import Any, Dict, List, Set, Tuple


def chunk_text(
    text: str, chunk_size: int, overlap: int
) -> List[Dict[str, Any]]:
    """Split text into word-based chunks with a sliding window and keep character offsets."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be strictly less than chunk_size")

    # Find all words and their start/end character offsets
    words = []
    for match in re.finditer(r"\S+", text):
        words.append((match.group(), match.start(), match.end()))

    if not words:
        return []

    chunks = []
    step = chunk_size - overlap
    i = 0
    while i < len(words):
        window = words[i : i + chunk_size]
        start_char = window[0][1]
        end_char = window[-1][2]
        chunk_text_str = text[start_char:end_char]
        chunks.append({"text": chunk_text_str, "start": start_char, "end": end_char})
        if i + chunk_size >= len(words):
            break
        i += step

    return chunks


class BM25Index:
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)

        # Tokenize corpus
        self.doc_tokens = []
        self.doc_lens = []
        self.doc_tfs = []
        self.df = Counter()

        for doc in corpus:
            tokens = re.findall(r"\w+", doc.lower())
            self.doc_tokens.append(tokens)
            self.doc_lens.append(len(tokens))
            tf = Counter(tokens)
            self.doc_tfs.append(tf)
            for token in tf:
                self.df[token] += 1

        self.avgdl = sum(self.doc_lens) / self.corpus_size if self.corpus_size > 0 else 0.0

        # Precompute IDF
        self.idf = {}
        for token, freq in self.df.items():
            self.idf[token] = math.log(
                1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5)
            )

    def score(self, query: str, doc_idx: int) -> float:
        if doc_idx < 0 or doc_idx >= self.corpus_size:
            raise IndexError("Document index out of bounds")

        query_tokens = re.findall(r"\w+", query.lower())
        doc_len = self.doc_lens[doc_idx]
        tf_dict = self.doc_tfs[doc_idx]

        score = 0.0
        for q in query_tokens:
            if q not in self.idf:
                continue
            tf = tf_dict.get(q, 0)
            if tf == 0:
                continue
            idf_val = self.idf[q]
            numerator = tf * (self.k1 + 1.0)
            doc_len_ratio = (doc_len / self.avgdl) if self.avgdl > 0 else 0.0
            denominator = tf + self.k1 * (
                1.0 - self.b + self.b * doc_len_ratio
            )
            score += idf_val * (numerator / denominator)

        return score

    def search(self, query: str, top_n: int = 5) -> List[Tuple[int, float]]:
        results = []
        for idx in range(self.corpus_size):
            s = self.score(query, idx)
            if s > 0.0:
                results.append((idx, s))

        # Sort descending by score, then ascending by doc index
        results.sort(key=lambda x: (-x[1], x[0]))
        return results[:top_n]


def reciprocal_rank_fusion(
    rankings: List[List[Any]], k: int = 60
) -> List[Tuple[Any, float]]:
    rrf_scores = {}
    for ranking in rankings:
        for rank_idx, item in enumerate(ranking):
            rank = rank_idx + 1
            rrf_scores[item] = rrf_scores.get(item, 0.0) + 1.0 / (k + rank)

    # Sort descending by score, then ascending by item
    try:
        sorted_items = sorted(rrf_scores.items(), key=lambda x: (-x[1], x[0]))
    except TypeError:
        sorted_items = sorted(rrf_scores.items(), key=lambda x: -x[1])
    return sorted_items


def dedupe_results(
    chunks: List[Dict[str, Any]], threshold: float = 0.5
) -> List[Dict[str, Any]]:
    accepted_chunks = []
    accepted_token_sets = []

    for chunk in chunks:
        tokens = set(re.findall(r"\w+", chunk["text"].lower()))
        is_duplicate = False

        for accepted_set in accepted_token_sets:
            if not tokens and not accepted_set:
                similarity = 0.0
            else:
                intersection = len(tokens.intersection(accepted_set))
                union = len(tokens.union(accepted_set))
                similarity = intersection / union if union > 0 else 0.0

            if similarity >= threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            accepted_chunks.append(chunk)
            accepted_token_sets.append(tokens)

    return accepted_chunks
