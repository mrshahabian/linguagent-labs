"""48 — Retrieval: chunking, BM25 and hybrid search

Implement text chunking, BM25 ranking, reciprocal rank fusion, and Jaccard
deduplication from scratch using only standard libraries and numpy.
"""

import math
import re
from collections import Counter
from typing import Any, Dict, List, Set, Tuple


def chunk_text(
    text: str, chunk_size: int, overlap: int
) -> List[Dict[str, Any]]:
    """Split text into word-based chunks with a sliding window and keep character offsets.

    A word is defined as any sequence of non-whitespace characters.
    If the text has no words, return an empty list.

    Args:
        text: The input string to chunk.
        chunk_size: Maximum number of words in each chunk.
        overlap: Number of words to overlap between consecutive chunks.

    Returns:
        A list of dictionaries, where each dictionary has:
            - "text": The exact substring from the original text spanning from the
                      start of the first word in the chunk to the end of the last word.
            - "start": The character start index of the chunk in the original text.
            - "end": The character end index of the chunk in the original text.

    Raises:
        ValueError: If overlap is negative, chunk_size is non-positive, or overlap >= chunk_size.
    """
    raise NotImplementedError("your code here")


class BM25Index:
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        r"""Initialize the BM25 Index.

        Tokenize documents by converting to lowercase and extracting all alphanumeric
        word tokens using re.findall(r'\w+', text).

        Args:
            corpus: A list of document strings.
            k1: Term frequency saturation parameter.
            b: Document length normalization parameter.
        """
        self.k1 = k1
        self.b = b
        raise NotImplementedError("your code here")

    def score(self, query: str, doc_idx: int) -> float:
        """Calculate the BM25 score for a single document given a query.

        Tokenize the query using the same alphanumeric lowercase strategy.
        Sum the BM25 term scores for each token in the tokenized query (with multiplicity).
        If a query token does not exist in the corpus vocabulary, its contribution is 0.

        Use the natural logarithm for IDF:
            IDF(q) = ln(1 + (N - n(q) + 0.5) / (n(q) + 0.5))
        where N is the total number of documents and n(q) is the number of documents
        containing term q.
        """
        raise NotImplementedError("your code here")

    def search(self, query: str, top_n: int = 5) -> List[Tuple[int, float]]:
        """Search the corpus and return the top_n document indices and their scores.

        Only return documents with a score strictly greater than 0.0.
        Sort results descending by score. If scores are tied, sort by document index ascending.
        """
        raise NotImplementedError("your code here")


def reciprocal_rank_fusion(
    rankings: List[List[Any]], k: int = 60
) -> List[Tuple[Any, float]]:
    """Combine multiple ranked lists using Reciprocal Rank Fusion (RRF).

    Args:
        rankings: A list of lists, where each inner list contains items in ranked order
                  (index 0 is rank 1, index 1 is rank 2, etc.).
        k: The constant parameter dampening the influence of top ranks.

    Returns:
        A list of tuples (item, rrf_score) sorted descending by rrf_score.
        If scores are tied, sort by the item itself ascending (assuming items are comparable).
    """
    raise NotImplementedError("your code here")


def dedupe_results(
    chunks: List[Dict[str, Any]], threshold: float = 0.5
) -> List[Dict[str, Any]]:
    r"""Deduplicate a list of chunks in order using Jaccard Similarity.

    Tokenize each chunk's "text" using re.findall(r'\w+', text.lower()).
    Keep chunks in their original order. For each chunk, compute its Jaccard similarity
    with all already accepted chunks. If the similarity is >= threshold with ANY accepted chunk,
    discard it.

    Jaccard similarity = |A intersect B| / |A union B|.
    If both token sets are empty, similarity is 0.0.

    Args:
        chunks: A list of dictionaries containing at least a "text" key.
        threshold: The Jaccard similarity threshold above which a chunk is considered a duplicate.

    Returns:
        A filtered list of chunk dictionaries.
    """
    raise NotImplementedError("your code here")
