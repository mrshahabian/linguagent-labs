"""46 — Tokenization: byte-pair encoding

Implement each function so the tests in test_exercises.py pass. Read
README.md first; every exercise below is covered there.

Standard library only.
"""

import collections
from typing import Dict, List, Tuple, Set


def get_pair_counts(word_freqs: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
    """Count the frequency of adjacent token pairs across all words."""
    counts = collections.defaultdict(int)
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            counts[pair] += freq
    return dict(counts)


def merge_pair(pair: Tuple[str, str], word_freqs: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, ...], int]:
    """Merge all occurrences of the given pair in the word-frequency dictionary."""
    new_freqs = collections.defaultdict(int)
    p0, p1 = pair
    merged_token = p0 + p1
    for word, freq in word_freqs.items():
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == p0 and word[i+1] == p1:
                new_word.append(merged_token)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_freqs[tuple(new_word)] += freq
    return dict(new_freqs)


def train_bpe(corpus: str, num_merges: int) -> Tuple[Set[str], List[Tuple[str, str]]]:
    """Train BPE on a corpus to learn a vocabulary and a list of merges."""
    base_chars = set(corpus)
    for ws in [' ', '\n', '\t', '\r']:
        base_chars.discard(ws)
    vocab = base_chars | {'</w>'}

    words = corpus.split()
    word_freqs = collections.defaultdict(int)
    for w in words:
        char_tuple = tuple(list(w) + ['</w>'])
        word_freqs[char_tuple] += 1

    merges = []
    for _ in range(num_merges):
        pairs = get_pair_counts(word_freqs)
        if not pairs:
            break
        # Sort by frequency descending, then lexicographically ascending
        sorted_pairs = sorted(pairs.items(), key=lambda item: (-item[1], item[0]))
        best_pair = sorted_pairs[0][0]
        merges.append(best_pair)
        word_freqs = merge_pair(best_pair, word_freqs)

    return vocab, merges


def encode(text: str, vocab: Set[str], merges: List[Tuple[str, str]]) -> List[str]:
    """Encode a text string into a list of tokens using the learned merges."""
    words = text.split()
    encoded_tokens = []
    for w in words:
        word_tokens = list(w) + ['</w>']
        for pair in merges:
            p0, p1 = pair
            merged_token = p0 + p1
            new_tokens = []
            i = 0
            while i < len(word_tokens):
                if i < len(word_tokens) - 1 and word_tokens[i] == p0 and word_tokens[i+1] == p1:
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(word_tokens[i])
                    i += 1
            word_tokens = new_tokens
        encoded_tokens.extend(word_tokens)
    return encoded_tokens


def decode(tokens: List[str]) -> str:
    """Decode a list of tokens back into a space-separated string."""
    joined = "".join(tokens)
    decoded = joined.replace("</w>", " ")
    return decoded.strip()


def count_tokens(text: str, vocab: Set[str], merges: List[Tuple[str, str]]) -> int:
    """Count the number of tokens in the encoded text."""
    return len(encode(text, vocab, merges))
