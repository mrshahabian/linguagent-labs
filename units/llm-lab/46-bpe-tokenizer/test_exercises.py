"""Tests for 46 — Tokenization: byte-pair encoding."""

import importlib
import os
import unittest

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestGetPairCounts(unittest.TestCase):
    def test_basic_counts(self):
        word_freqs = {
            ('h', 'u', 'g', '</w>'): 10,
            ('p', 'u', 'g', '</w>'): 5
        }
        expected = {
            ('h', 'u'): 10,
            ('u', 'g'): 15,
            ('g', '</w>'): 15,
            ('p', 'u'): 5
        }
        self.assertEqual(m.get_pair_counts(word_freqs), expected)

    def test_empty_freqs(self):
        self.assertEqual(m.get_pair_counts({}), {})

    def test_single_char_words(self):
        # Single-token words have no adjacent pairs
        word_freqs = {
            ('a',): 10,
            ('b',): 5
        }
        self.assertEqual(m.get_pair_counts(word_freqs), {})

    def test_multiple_occurrences_in_same_word(self):
        word_freqs = {
            ('a', 'a', 'a', '</w>'): 10
        }
        # Pairs are ('a', 'a') at index 0, ('a', 'a') at index 1, ('a', '</w>') at index 2
        expected = {
            ('a', 'a'): 20,
            ('a', '</w>'): 10
        }
        self.assertEqual(m.get_pair_counts(word_freqs), expected)

    def test_weighted_counts(self):
        word_freqs = {
            ('a', 'b'): 3,
            ('a', 'b', 'c'): 2
        }
        expected = {
            ('a', 'b'): 5,
            ('b', 'c'): 2
        }
        self.assertEqual(m.get_pair_counts(word_freqs), expected)


class TestMergePair(unittest.TestCase):
    def test_basic_merge(self):
        word_freqs = {('h', 'u', 'g', '</w>'): 10}
        pair = ('u', 'g')
        expected = {('h', 'ug', '</w>'): 10}
        self.assertEqual(m.merge_pair(pair, word_freqs), expected)

    def test_overlapping_merge(self):
        # Standard left-to-right greedy merge
        word_freqs = {('a', 'a', 'a', '</w>'): 10}
        pair = ('a', 'a')
        expected = {('aa', 'a', '</w>'): 10}
        self.assertEqual(m.merge_pair(pair, word_freqs), expected)

    def test_frequency_summing(self):
        # If two different words merge into the same representation, sum frequencies
        word_freqs = {
            ('a', 'b', 'c'): 2,
            ('ab', 'c'): 3
        }
        pair = ('a', 'b')
        expected = {('ab', 'c'): 5}
        self.assertEqual(m.merge_pair(pair, word_freqs), expected)

    def test_no_match(self):
        word_freqs = {('h', 'u', 'g', '</w>'): 10}
        pair = ('x', 'y')
        self.assertEqual(m.merge_pair(pair, word_freqs), word_freqs)

    def test_merge_at_beginning_and_end(self):
        word_freqs = {('a', 'b', 'a', 'b'): 5}
        pair = ('a', 'b')
        expected = {('ab', 'ab'): 5}
        self.assertEqual(m.merge_pair(pair, word_freqs), expected)


class TestTrainBPE(unittest.TestCase):
    def test_simple_corpus(self):
        corpus = "hug hug hug pug pug"
        vocab, merges = m.train_bpe(corpus, num_merges=2)
        # Initial chars: h, u, g, p, </w>
        self.assertTrue('u' in vocab)
        self.assertTrue('</w>' in vocab)
        self.assertEqual(merges, [('g', '</w>'), ('u', 'g</w>')])

    def test_tie_breaking(self):
        # Both ('a', 'b') and ('c', 'd') appear once.
        # Alphabetical tie-breaking should pick ('a', 'b') first.
        corpus = "ab cd"
        _, merges = m.train_bpe(corpus, num_merges=1)
        self.assertEqual(merges[0], ('a', 'b'))

    def test_zero_merges(self):
        corpus = "hello"
        vocab, merges = m.train_bpe(corpus, num_merges=0)
        self.assertEqual(merges, [])
        self.assertEqual(vocab, {'h', 'e', 'l', 'o', '</w>'})

    def test_empty_corpus(self):
        vocab, merges = m.train_bpe("", num_merges=5)
        self.assertEqual(merges, [])
        self.assertEqual(vocab, {'</w>'})

    def test_no_possible_merges(self):
        # Single letter corpus has no adjacent pairs to merge after the first merge
        corpus = "a"
        vocab, merges = m.train_bpe(corpus, num_merges=5)
        # The only word is "a", represented as ('a', '</w>').
        # The only pair is ('a', '</w>'), which gets merged to 'a</w>'.
        # After that, the word is ('a</w>',), which has length 1, so no more merges are possible.
        self.assertEqual(merges, [('a', '</w>')])
        self.assertEqual(vocab, {'a', '</w>'})


class TestEncode(unittest.TestCase):
    def test_encode_basic(self):
        vocab = {'h', 'u', 'g', 'p', '</w>'}
        merges = [('g', '</w>'), ('u', 'g</w>')]
        self.assertEqual(m.encode("hug", vocab, merges), ['h', 'ug</w>'])
        self.assertEqual(m.encode("pug", vocab, merges), ['p', 'ug</w>'])

    def test_encode_unseen_char(self):
        vocab = {'h', 'u', 'g', '</w>'}
        merges = [('g', '</w>')]
        # 'x' is unseen, should remain as individual token 'x'
        # In "hugx", the characters are ['h', 'u', 'g', 'x', '</w>'].
        # The pair ('g', '</w>') is not adjacent, so it cannot be merged.
        self.assertEqual(m.encode("hugx", vocab, merges), ['h', 'u', 'g', 'x', '</w>'])

    def test_encode_unseen_char_with_other_merges(self):
        vocab = {'h', 'u', 'g', '</w>'}
        merges = [('u', 'g')]
        # 'x' is unseen, but ('u', 'g') is adjacent and should be merged.
        self.assertEqual(m.encode("hugx", vocab, merges), ['h', 'ug', 'x', '</w>'])

    def test_encode_order(self):
        # Verifies that merges are applied in chronological order
        vocab = {'a', 'b', 'c', '</w>'}
        merges = [('a', 'b'), ('ab', 'c')]
        self.assertEqual(m.encode("abc", vocab, merges), ['abc', '</w>'])

    def test_encode_empty_string(self):
        vocab = {'a', '</w>'}
        merges = []
        self.assertEqual(m.encode("", vocab, merges), [])

    def test_encode_multiple_spaces(self):
        vocab = {'a', 'b', '</w>'}
        merges = [('a', 'b')]
        # Multiple spaces should be treated as single word boundaries (split behavior)
        self.assertEqual(m.encode("a   b", vocab, merges), ['a', '</w>', 'b', '</w>'])


class TestDecodeAndCount(unittest.TestCase):
    def test_decode_basic(self):
        tokens = ['h', 'ug</w>', 'p', 'ug</w>']
        self.assertEqual(m.decode(tokens), "hug pug")

    def test_round_trip(self):
        corpus = "the quick brown fox jumps over the lazy dog"
        vocab, merges = m.train_bpe(corpus, num_merges=10)
        encoded = m.encode(corpus, vocab, merges)
        decoded = m.decode(encoded)
        self.assertEqual(decoded, corpus)

    def test_count_tokens(self):
        corpus = "hello world"
        vocab, merges = m.train_bpe(corpus, num_merges=3)
        token_count = m.count_tokens("hello world", vocab, merges)
        self.assertEqual(token_count, len(m.encode("hello world", vocab, merges)))

    def test_decode_empty(self):
        self.assertEqual(m.decode([]), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
