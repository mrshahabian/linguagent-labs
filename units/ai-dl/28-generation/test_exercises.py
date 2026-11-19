"""Tests for 28 - decoding, sampling and evaluation.

Sampling is tested with a seeded generator and with distributions whose
support is exactly known, so the assertions are about which tokens CAN be
drawn - never about a particular draw.
"""

import importlib
import math
import os
import unittest

import torch
import torch.nn.functional as F

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))

NEG = -1e30          # anything below this counts as "filtered out"


def kept(logits):
    """The indices a filter left alive, as a sorted list."""
    return sorted((logits[0] > NEG).nonzero().flatten().tolist())


class TestTemperature(unittest.TestCase):
    def setUp(self):
        self.logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    def test_one_is_a_no_op(self):
        torch.testing.assert_close(m.apply_temperature(self.logits, 1.0),
                                   self.logits)

    def test_it_divides_the_logits(self):
        torch.testing.assert_close(m.apply_temperature(self.logits, 2.0),
                                   self.logits / 2)

    def test_low_temperature_sharpens(self):
        cold = m.apply_temperature(self.logits, 0.5).softmax(-1)
        warm = m.apply_temperature(self.logits, 2.0).softmax(-1)
        self.assertGreater(cold.max().item(), warm.max().item())

    def test_high_temperature_flattens_towards_uniform(self):
        hot = m.apply_temperature(self.logits, 100.0).softmax(-1)
        torch.testing.assert_close(hot, torch.full((1, 4), 0.25),
                                   atol=1e-2, rtol=0)

    def test_the_ranking_is_never_changed(self):
        for temperature in (0.1, 1.0, 10.0):
            out = m.apply_temperature(self.logits, temperature)
            self.assertEqual(out.argmax(-1).item(), 3)

    def test_zero_is_greedy(self):
        out = m.apply_temperature(self.logits, 0.0)
        probabilities = out.softmax(-1)
        self.assertAlmostEqual(probabilities[0, 3].item(), 1.0, places=5)

    def test_zero_does_not_produce_nan(self):
        self.assertTrue(torch.isfinite(
            m.apply_temperature(self.logits, 0.0).softmax(-1)).all())

    def test_negative_raises(self):
        with self.assertRaises(ValueError):
            m.apply_temperature(self.logits, -1.0)


class TestTopK(unittest.TestCase):
    def setUp(self):
        self.logits = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])

    def test_keeps_exactly_k(self):
        self.assertEqual(len(kept(m.top_k_filter(self.logits, 2))), 2)

    def test_keeps_the_largest(self):
        self.assertEqual(kept(m.top_k_filter(self.logits, 2)), [1, 4])

    def test_k_of_one_is_greedy(self):
        out = m.top_k_filter(self.logits, 1)
        self.assertAlmostEqual(out.softmax(-1)[0, 1].item(), 1.0, places=5)

    def test_k_larger_than_the_vocab_is_a_no_op(self):
        torch.testing.assert_close(m.top_k_filter(self.logits, 99),
                                   self.logits)

    def test_surviving_logits_are_unchanged(self):
        out = m.top_k_filter(self.logits, 3)
        torch.testing.assert_close(out[0, 1], self.logits[0, 1])

    def test_the_result_still_normalises(self):
        probabilities = m.top_k_filter(self.logits, 3).softmax(-1)
        torch.testing.assert_close(probabilities.sum(-1), torch.ones(1))

    def test_k_zero_raises(self):
        with self.assertRaises(ValueError):
            m.top_k_filter(self.logits, 0)

    def test_batched(self):
        batch = torch.tensor([[1.0, 5.0, 3.0], [9.0, 1.0, 2.0]])
        out = m.top_k_filter(batch, 1)
        self.assertEqual(out.argmax(-1).tolist(), [1, 0])
        self.assertEqual((out > NEG).sum().item(), 2)


class TestTopP(unittest.TestCase):
    def setUp(self):
        # probabilities 0.5, 0.3, 0.15, 0.05
        self.logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))

    def test_the_nucleus_is_the_smallest_set_reaching_p(self):
        self.assertEqual(kept(m.top_p_filter(self.logits, 0.8)), [0, 1])

    def test_a_larger_p_widens_the_nucleus(self):
        self.assertEqual(kept(m.top_p_filter(self.logits, 0.9)), [0, 1, 2])

    def test_p_of_one_keeps_everything(self):
        self.assertEqual(kept(m.top_p_filter(self.logits, 1.0)), [0, 1, 2, 3])

    def test_the_top_token_is_always_kept(self):
        """Even when it alone exceeds p, or nothing could be sampled."""
        self.assertEqual(kept(m.top_p_filter(self.logits, 0.1)), [0])

    def test_a_tiny_p_still_leaves_a_valid_distribution(self):
        probabilities = m.top_p_filter(self.logits, 0.01).softmax(-1)
        torch.testing.assert_close(probabilities.sum(-1), torch.ones(1))
        self.assertFalse(torch.isnan(probabilities).any())

    def test_it_adapts_to_a_confident_distribution(self):
        """Where the model is sure, the nucleus is one token."""
        confident = torch.log(torch.tensor([[0.97, 0.01, 0.01, 0.01]]))
        self.assertEqual(kept(m.top_p_filter(confident, 0.9)), [0])

    def test_it_adapts_to_a_flat_distribution(self):
        """Where the model is unsure, the nucleus widens - top-k cannot."""
        flat = torch.log(torch.full((1, 8), 0.125))
        self.assertGreaterEqual(len(kept(m.top_p_filter(flat, 0.9))), 7)

    def test_surviving_logits_are_unchanged(self):
        out = m.top_p_filter(self.logits, 0.9)
        torch.testing.assert_close(out[0, :3], self.logits[0, :3])

    def test_it_does_not_depend_on_input_order(self):
        shuffled = self.logits[:, [3, 1, 0, 2]]
        self.assertEqual(kept(m.top_p_filter(shuffled, 0.8)), [1, 2])

    def test_invalid_p_raises(self):
        for bad in (0.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                m.top_p_filter(self.logits, bad)


class TestRepetitionPenalty(unittest.TestCase):
    def test_positive_logits_are_reduced(self):
        out = m.repetition_penalty(torch.tensor([[2.0, 1.0]]), [[0]], 2.0)
        self.assertAlmostEqual(out[0, 0].item(), 1.0)

    def test_negative_logits_are_also_reduced(self):
        """Dividing a negative logit would INCREASE it - the classic bug."""
        out = m.repetition_penalty(torch.tensor([[-2.0, 1.0]]), [[0]], 2.0)
        self.assertAlmostEqual(out[0, 0].item(), -4.0)

    def test_the_penalised_token_becomes_less_likely(self):
        logits = torch.tensor([[-2.0, -2.1]])
        before = logits.softmax(-1)[0, 0]
        after = m.repetition_penalty(logits, [[0]], 1.5).softmax(-1)[0, 0]
        self.assertLess(after.item(), before.item())

    def test_untouched_tokens_keep_their_logits(self):
        out = m.repetition_penalty(torch.tensor([[2.0, 1.0]]), [[0]], 2.0)
        self.assertAlmostEqual(out[0, 1].item(), 1.0)

    def test_a_penalty_of_one_is_a_no_op(self):
        logits = torch.tensor([[2.0, -1.0]])
        torch.testing.assert_close(m.repetition_penalty(logits, [[0, 1]], 1.0),
                                   logits)

    def test_repeated_tokens_are_penalised_once(self):
        once = m.repetition_penalty(torch.tensor([[4.0, 1.0]]), [[0]], 2.0)
        twice = m.repetition_penalty(torch.tensor([[4.0, 1.0]]), [[0, 0]], 2.0)
        torch.testing.assert_close(once, twice)

    def test_an_empty_history_changes_nothing(self):
        logits = torch.tensor([[2.0, 1.0]])
        torch.testing.assert_close(m.repetition_penalty(logits, [[]], 2.0),
                                   logits)

    def test_it_does_not_mutate_the_input(self):
        logits = torch.tensor([[2.0, 1.0]])
        before = logits.clone()
        m.repetition_penalty(logits, [[0]], 2.0)
        torch.testing.assert_close(logits, before)

    def test_rows_are_penalised_independently(self):
        logits = torch.tensor([[4.0, 4.0], [4.0, 4.0]])
        out = m.repetition_penalty(logits, [[0], [1]], 2.0)
        torch.testing.assert_close(out, torch.tensor([[2.0, 4.0],
                                                      [4.0, 2.0]]))

    def test_a_non_positive_penalty_raises(self):
        with self.assertRaises(ValueError):
            m.repetition_penalty(torch.tensor([[1.0]]), [[0]], 0.0)


class TestSampleNextToken(unittest.TestCase):
    def setUp(self):
        self.logits = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])

    def test_returns_one_index_per_row(self):
        out = m.sample_next_token(torch.randn(3, 10))
        self.assertEqual(out.shape, (3,))

    def test_it_is_reproducible_with_a_generator(self):
        first = m.sample_next_token(
            self.logits, generator=torch.Generator().manual_seed(0))
        second = m.sample_next_token(
            self.logits, generator=torch.Generator().manual_seed(0))
        self.assertEqual(first.item(), second.item())

    def test_top_k_one_is_deterministic(self):
        for _ in range(10):
            self.assertEqual(m.sample_next_token(self.logits, top_k=1).item(),
                             1)

    def test_temperature_zero_is_deterministic(self):
        for _ in range(10):
            self.assertEqual(
                m.sample_next_token(self.logits, temperature=0.0).item(), 1)

    def test_it_only_ever_samples_from_the_filtered_set(self):
        drawn = {m.sample_next_token(self.logits, top_k=2).item()
                 for _ in range(200)}
        self.assertTrue(drawn <= {1, 4}, f"sampled outside the top-2: {drawn}")

    def test_top_p_restricts_the_support(self):
        logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))
        drawn = {m.sample_next_token(logits, top_p=0.8).item()
                 for _ in range(200)}
        self.assertTrue(drawn <= {0, 1}, f"sampled outside the nucleus: {drawn}")

    def test_temperature_is_applied_before_the_filters(self):
        """Filtering first would give a different distribution.

        At T=100 the logits are nearly uniform, so top_p=0.5 keeps about
        half the vocabulary. Filtering before scaling would keep only one.
        """
        logits = torch.log(torch.tensor([[0.9, 0.04, 0.03, 0.03]]))
        drawn = {m.sample_next_token(logits, temperature=100.0,
                                     top_p=0.5).item() for _ in range(300)}
        self.assertGreater(len(drawn), 1)

    def test_it_does_not_mutate_the_logits(self):
        before = self.logits.clone()
        m.sample_next_token(self.logits, temperature=0.7, top_k=2)
        torch.testing.assert_close(self.logits, before)


class ToyModel:
    """A deterministic next-token oracle keyed on the last token.

    Encodes a trap greedy decoding walks into: token 1 looks best now but
    leads nowhere, while token 2 leads to a far better continuation.
    """

    EOS = 3
    TABLE = {
        0: [0.0, 0.6, 0.4, 0.0],     # start: 1 looks better than 2
        1: [0.0, 0.0, 0.0, 1.0],     # dead end: forced to eos
        2: [0.0, 0.0, 0.0, 1.0],     # also to eos, but from a better prefix
    }
    # Sequence probabilities: 0->1->eos = 0.6 * 1.0, 0->2->eos = 0.4 * 1.0
    # so with this table greedy is optimal; the trap version is below.

    def __call__(self, tokens):
        row = self.TABLE.get(tokens[-1], [0.0, 0.0, 0.0, 1.0])
        return torch.log(torch.tensor(row).clamp_min(1e-9))


class TrapModel(ToyModel):
    """Greedy takes the locally better first step and loses overall.

    Greedy: 0 -> 1 (0.6) -> eos (0.5) = 0.30
    Beam:   0 -> 2 (0.4) -> eos (0.9) = 0.36

    Both continuations end at eos, so the two candidates are the same
    length and the length penalty cannot explain the difference.
    """

    TABLE = {
        0: [0.0, 0.6, 0.4, 0.0],
        1: [0.2, 0.2, 0.1, 0.5],
        2: [0.05, 0.03, 0.02, 0.9],
    }


class TestGreedyDecode(unittest.TestCase):
    def test_it_appends_to_the_prompt(self):
        tokens = m.greedy_decode(ToyModel(), [0], max_new_tokens=2)
        self.assertEqual(tokens[0], 0)

    def test_it_picks_the_argmax(self):
        tokens = m.greedy_decode(ToyModel(), [0], max_new_tokens=1)
        self.assertEqual(tokens, [0, 1])

    def test_it_stops_at_eos(self):
        tokens = m.greedy_decode(ToyModel(), [0], max_new_tokens=10,
                                 eos_token=3)
        self.assertEqual(tokens, [0, 1, 3])

    def test_eos_is_included_in_the_output(self):
        tokens = m.greedy_decode(ToyModel(), [0], max_new_tokens=10,
                                 eos_token=3)
        self.assertEqual(tokens[-1], 3)

    def test_it_respects_max_new_tokens(self):
        tokens = m.greedy_decode(ToyModel(), [0], max_new_tokens=4)
        self.assertEqual(len(tokens), 5)

    def test_it_is_deterministic(self):
        first = m.greedy_decode(ToyModel(), [0], 5)
        self.assertEqual(m.greedy_decode(ToyModel(), [0], 5), first)

    def test_it_does_not_mutate_the_prompt(self):
        prompt = [0]
        m.greedy_decode(ToyModel(), prompt, 3)
        self.assertEqual(prompt, [0])

    def test_zero_new_tokens_returns_the_prompt(self):
        self.assertEqual(m.greedy_decode(ToyModel(), [0, 1], 0), [0, 1])


class TestBeamSearch(unittest.TestCase):
    def test_it_returns_a_sequence_starting_with_the_prompt(self):
        tokens = m.beam_search(ToyModel(), [0], max_new_tokens=2,
                               beam_width=2, eos_token=3)
        self.assertEqual(tokens[0], 0)

    def test_width_one_matches_greedy(self):
        beam = m.beam_search(ToyModel(), [0], 3, beam_width=1, eos_token=3)
        greedy = m.greedy_decode(ToyModel(), [0], 3, eos_token=3)
        self.assertEqual(beam, greedy)

    def test_it_beats_greedy_on_the_trap(self):
        """The whole reason beam search exists."""
        greedy = m.greedy_decode(TrapModel(), [0], 2, eos_token=3)
        beam = m.beam_search(TrapModel(), [0], 2, beam_width=3, eos_token=3)
        self.assertEqual(greedy, [0, 1, 3])
        self.assertEqual(beam, [0, 2, 3])

    def test_it_stops_when_every_beam_has_finished(self):
        tokens = m.beam_search(ToyModel(), [0], max_new_tokens=20,
                               beam_width=2, eos_token=3)
        self.assertLessEqual(len(tokens), 4)

    def test_it_respects_max_new_tokens(self):
        tokens = m.beam_search(ToyModel(), [0], max_new_tokens=2,
                               beam_width=2)
        self.assertLessEqual(len(tokens), 3)

    def test_length_penalty_zero_favours_shorter_sequences(self):
        """alpha=0 means no normalisation, so every extra token costs."""
        tokens = m.beam_search(TrapModel(), [0], 3, beam_width=3,
                               eos_token=3, length_penalty=0.0)
        self.assertEqual(tokens[-1], 3)

    def test_it_does_not_mutate_the_prompt(self):
        prompt = [0]
        m.beam_search(ToyModel(), prompt, 3, beam_width=2)
        self.assertEqual(prompt, [0])


class TestPerplexity(unittest.TestCase):
    def test_a_uniform_model_has_perplexity_equal_to_the_vocab(self):
        logits = torch.zeros(1, 4, 7)
        targets = torch.zeros(1, 4, dtype=torch.long)
        self.assertAlmostEqual(m.perplexity(logits, targets).item(), 7.0,
                               places=4)

    def test_a_confident_correct_model_approaches_one(self):
        logits = torch.full((1, 3, 5), -20.0)
        logits[0, :, 2] = 20.0
        targets = torch.full((1, 3), 2, dtype=torch.long)
        self.assertAlmostEqual(m.perplexity(logits, targets).item(), 1.0,
                               places=4)

    def test_a_confident_wrong_model_is_enormous(self):
        logits = torch.full((1, 2, 5), -10.0)
        logits[0, :, 0] = 10.0
        targets = torch.full((1, 2), 4, dtype=torch.long)
        self.assertGreater(m.perplexity(logits, targets).item(), 1000.0)

    def test_ignored_positions_do_not_count(self):
        logits = torch.zeros(1, 4, 7)
        targets = torch.tensor([[0, 0, -100, -100]])
        self.assertAlmostEqual(m.perplexity(logits, targets).item(), 7.0,
                               places=4)

    def test_padding_would_otherwise_flatter_the_score(self):
        """Confident on the padding, uniform on the real tokens."""
        logits = torch.zeros(1, 4, 7)
        logits[0, 2:, 0] = 30.0
        with_padding = m.perplexity(logits, torch.tensor([[3, 3, 0, 0]]))
        ignored = m.perplexity(logits, torch.tensor([[3, 3, -100, -100]]))
        self.assertLess(with_padding.item(), ignored.item())

    def test_it_returns_a_scalar_tensor(self):
        out = m.perplexity(torch.zeros(1, 2, 3),
                           torch.zeros(1, 2, dtype=torch.long))
        self.assertEqual(out.shape, ())

    def test_it_equals_exp_of_cross_entropy(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 5, 11)
        targets = torch.randint(0, 11, (2, 5))
        expected = F.cross_entropy(logits.reshape(-1, 11),
                                   targets.reshape(-1)).exp()
        torch.testing.assert_close(m.perplexity(logits, targets), expected)


class TestShiftForCausalLm(unittest.TestCase):
    def test_labels_are_the_inputs_shifted_by_one(self):
        inputs, labels = m.shift_for_causal_lm(torch.tensor([[1, 2, 3, 4]]))
        self.assertEqual(inputs.tolist(), [[1, 2, 3]])
        self.assertEqual(labels.tolist(), [[2, 3, 4]])

    def test_shapes_match_each_other(self):
        inputs, labels = m.shift_for_causal_lm(torch.randint(0, 9, (2, 6)))
        self.assertEqual(inputs.shape, labels.shape)
        self.assertEqual(inputs.shape, (2, 5))

    def test_position_i_predicts_token_i_plus_one(self):
        ids = torch.tensor([[5, 6, 7]])
        inputs, labels = m.shift_for_causal_lm(ids)
        for i in range(inputs.shape[1]):
            self.assertEqual(labels[0, i].item(), ids[0, i + 1].item())

    def test_it_is_not_the_identity(self):
        """Labels equal to inputs teaches the model to copy - fluent nonsense."""
        ids = torch.tensor([[1, 2, 3, 4]])
        inputs, labels = m.shift_for_causal_lm(ids)
        self.assertFalse(torch.equal(inputs, labels))

    def test_it_does_not_mutate_the_input(self):
        ids = torch.tensor([[1, 2, 3]])
        before = ids.clone()
        m.shift_for_causal_lm(ids)
        torch.testing.assert_close(ids, before)

    def test_writing_to_the_labels_does_not_touch_the_source(self):
        ids = torch.tensor([[1, 2, 3]])
        _, labels = m.shift_for_causal_lm(ids)
        labels[0, 0] = -100
        self.assertEqual(ids[0, 1].item(), 2)

    def test_a_single_token_raises(self):
        with self.assertRaises(ValueError):
            m.shift_for_causal_lm(torch.tensor([[1]]))

    def test_it_composes_with_perplexity(self):
        torch.manual_seed(0)
        ids = torch.randint(0, 11, (2, 6))
        inputs, labels = m.shift_for_causal_lm(ids)
        logits = torch.randn(*inputs.shape, 11)
        self.assertTrue(torch.isfinite(m.perplexity(logits, labels)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
