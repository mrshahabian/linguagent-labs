"""43 — Memory: windows, summaries and recall

Reference implementation of the memory management utilities and the TF-IDF semantic recall store.
"""

import math
import re
from typing import Callable, Any, Dict, List, Tuple
import numpy as np


def count_tokens(text: str) -> int:
    """Estimate the number of tokens in a string."""
    if not text.strip():
        return 0
    return len(text.split())


def trim_history(
    messages: List[Dict[str, Any]],
    max_tokens: int,
    system_index: int = 0,
    token_counter: Callable[[str], int] = count_tokens,
) -> List[Dict[str, Any]]:
    """Trim conversation history to fit within max_tokens."""
    if max_tokens < 0:
        raise ValueError("Token budget cannot be negative")

    has_sys = 0 <= system_index < len(messages)
    sys_msg = messages[system_index] if has_sys else None
    sys_tokens = token_counter(sys_msg["content"]) if sys_msg else 0

    if sys_tokens > max_tokens:
        raise ValueError("System message exceeds token budget")

    remaining_budget = max_tokens - sys_tokens
    kept_non_sys = []

    for i in range(len(messages) - 1, -1, -1):
        if has_sys and i == system_index:
            continue
        msg = messages[i]
        tokens = token_counter(msg["content"])
        if tokens <= remaining_budget:
            kept_non_sys.append((i, msg))
            remaining_budget -= tokens
        else:
            break

    kept_non_sys.reverse()

    result = []
    if sys_msg is not None:
        result.append(sys_msg)
    for _, msg in kept_non_sys:
        result.append(msg)

    return result


def summarize_overflow(
    messages: List[Dict[str, Any]],
    max_tokens: int,
    summarizer: Callable[[List[Dict[str, Any]]], str],
    system_index: int = 0,
    token_counter: Callable[[str], int] = count_tokens,
) -> List[Dict[str, Any]]:
    """Condense conversation history by summarizing overflow messages."""
    total_tokens = sum(token_counter(m["content"]) for m in messages)
    if total_tokens <= max_tokens:
        return messages

    trimmed = trim_history(messages, max_tokens, system_index, token_counter)
    trimmed_ids = {id(m) for m in trimmed}

    has_sys = 0 <= system_index < len(messages)
    sys_msg = messages[system_index] if has_sys else None

    dropped = []
    for i, m in enumerate(messages):
        if has_sys and i == system_index:
            continue
        if id(m) not in trimmed_ids:
            dropped.append(m)

    if not dropped:
        return trimmed

    summary_text = summarizer(dropped)
    summary_msg = {
        "role": "system",
        "content": f"Summary of earlier conversation: {summary_text}",
    }

    sys_tokens = token_counter(sys_msg["content"]) if sys_msg else 0
    summary_tokens = token_counter(summary_msg["content"])
    fixed_tokens = sys_tokens + summary_tokens

    if fixed_tokens > max_tokens:
        raise ValueError("System message and summary exceed token budget")

    remaining_budget = max_tokens - fixed_tokens
    kept_newest = []

    for i in range(len(messages) - 1, -1, -1):
        if has_sys and i == system_index:
            continue
        if id(messages[i]) in trimmed_ids:
            msg = messages[i]
            tokens = token_counter(msg["content"])
            if tokens <= remaining_budget:
                kept_newest.append(msg)
                remaining_budget -= tokens
            else:
                break

    kept_newest.reverse()

    result = []
    if sys_msg:
        result.append(sys_msg)
    result.append(summary_msg)
    result.extend(kept_newest)
    return result


class MemoryStore:
    """A zero-dependency semantic memory store using TF-IDF and NumPy."""

    def __init__(self) -> None:
        self.docs: List[str] = []

    def add(self, text: str) -> None:
        self.docs.append(text)

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def search(self, query: str, k: int = 1) -> List[Tuple[str, float]]:
        if not self.docs:
            return []

        all_tokens = [self._tokenize(doc) for doc in self.docs]
        vocab_set = set()
        for tokens in all_tokens:
            vocab_set.update(tokens)
        vocab = sorted(list(vocab_set))

        if not vocab:
            return [(doc, 0.0) for doc in self.docs[:k]]

        N = len(self.docs)
        df = {}
        for term in vocab:
            df[term] = sum(1 for doc_tokens in all_tokens if term in doc_tokens)

        idf = {}
        for term in vocab:
            idf[term] = math.log((1 + N) / (1 + df[term])) + 1

        def vectorize(tokens: List[str]) -> np.ndarray:
            vec = np.zeros(len(vocab))
            for i, term in enumerate(vocab):
                tf = tokens.count(term)
                vec[i] = tf * idf[term]
            return vec

        doc_vectors = [vectorize(tokens) for tokens in all_tokens]
        query_tokens = self._tokenize(query)
        query_vector = vectorize(query_tokens)

        norm_q = np.linalg.norm(query_vector)
        similarities = []

        for doc, doc_vec in zip(self.docs, doc_vectors):
            norm_d = np.linalg.norm(doc_vec)
            if norm_q > 0 and norm_d > 0:
                sim = float(np.dot(query_vector, doc_vec) / (norm_q * norm_d))
            else:
                sim = 0.0
            similarities.append((doc, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
