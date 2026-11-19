"""Tests for 35 - assembling transformers end to end."""

import importlib
import os
import unittest

import torch

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestSinusoidalPositionalEncoding(unittest.TestCase):
    def test_shape(self):
        enc = m.sinusoidal_positional_encoding(10, 16)
        self.assertEqual(enc.shape, (10, 16))

    def test_position_zero_is_sin0_cos0_pattern(self):
        enc = m.sinusoidal_positional_encoding(5, 8)
        self.assertTrue(torch.allclose(enc[0, 0::2], torch.zeros(4)))
        self.assertTrue(torch.allclose(enc[0, 1::2], torch.ones(4)))

    def test_different_positions_give_different_encodings(self):
        enc = m.sinusoidal_positional_encoding(5, 8)
        self.assertFalse(torch.allclose(enc[0], enc[1]))
        self.assertFalse(torch.allclose(enc[1], enc[2]))

    def test_values_are_bounded(self):
        enc = m.sinusoidal_positional_encoding(20, 32)
        self.assertTrue((enc >= -1.0001).all())
        self.assertTrue((enc <= 1.0001).all())


class TestGPTModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = m.GPTModel(vocab_size=50, embed_dim=32, num_layers=2,
                                 num_heads=4, max_seq_len=20)

    def test_lm_head_is_tied_to_token_embedding(self):
        self.assertIs(self.model.lm_head.weight, self.model.token_embedding.weight)

    def test_forward_output_shape(self):
        input_ids = torch.randint(0, 50, (2, 6))
        logits = self.model(input_ids)
        self.assertEqual(logits.shape, (2, 6, 50))

    def test_stacks_the_requested_number_of_blocks(self):
        self.assertEqual(len(self.model.blocks), 2)

    def test_causal_mask_blocks_future_tokens(self):
        # changing a later token must not change an earlier position's logits
        torch.manual_seed(1)
        input_ids = torch.randint(0, 50, (1, 6))
        logits_before = self.model(input_ids)

        modified = input_ids.clone()
        modified[0, -1] = (modified[0, -1] + 1) % 50
        logits_after = self.model(modified)

        self.assertTrue(torch.allclose(logits_before[0, 0], logits_after[0, 0], atol=1e-5))

    def test_different_batches_are_independent(self):
        torch.manual_seed(2)
        a = torch.randint(0, 50, (1, 5))
        b = torch.randint(0, 50, (1, 5))
        batched = torch.cat([a, b], dim=0)
        out_batched = self.model(batched)
        out_a = self.model(a)
        self.assertTrue(torch.allclose(out_batched[0], out_a[0], atol=1e-5))


class TestCrossAttentionDecoderBlock(unittest.TestCase):
    def test_output_shape_matches_target_sequence(self):
        torch.manual_seed(0)
        block = m.CrossAttentionDecoderBlock(embed_dim=32, num_heads=4)
        target = torch.randn(2, 5, 32)
        encoder_output = torch.randn(2, 7, 32)
        causal_mask = torch.triu(torch.full((5, 5), float("-inf")), diagonal=1)
        result = block(target, encoder_output, causal_mask)
        self.assertEqual(result.shape, (2, 5, 32))

    def test_encoder_and_target_lengths_may_differ(self):
        torch.manual_seed(1)
        block = m.CrossAttentionDecoderBlock(embed_dim=16, num_heads=2)
        target = torch.randn(1, 3, 16)
        encoder_output = torch.randn(1, 11, 16)
        causal_mask = torch.triu(torch.full((3, 3), float("-inf")), diagonal=1)
        result = block(target, encoder_output, causal_mask)
        self.assertEqual(result.shape, (1, 3, 16))

    def test_encoder_output_influences_the_result(self):
        torch.manual_seed(2)
        block = m.CrossAttentionDecoderBlock(embed_dim=16, num_heads=2)
        target = torch.randn(1, 3, 16)
        enc_a = torch.randn(1, 4, 16)
        enc_b = torch.randn(1, 4, 16)
        causal_mask = torch.triu(torch.full((3, 3), float("-inf")), diagonal=1)
        out_a = block(target, enc_a, causal_mask)
        out_b = block(target, enc_b, causal_mask)
        self.assertFalse(torch.allclose(out_a, out_b))


class TestCountTransformerParameters(unittest.TestCase):
    def test_matches_a_real_instantiated_model(self):
        torch.manual_seed(0)
        model = m.GPTModel(vocab_size=50, embed_dim=32, num_layers=2,
                            num_heads=4, max_seq_len=20)
        real_count = sum(p.numel() for p in model.parameters())
        formula_count = m.count_transformer_parameters(
            vocab_size=50, embed_dim=32, num_layers=2, max_seq_len=20)
        self.assertEqual(formula_count, real_count)

    def test_matches_a_larger_model(self):
        torch.manual_seed(1)
        model = m.GPTModel(vocab_size=200, embed_dim=64, num_layers=4,
                            num_heads=8, max_seq_len=128)
        real_count = sum(p.numel() for p in model.parameters())
        formula_count = m.count_transformer_parameters(
            vocab_size=200, embed_dim=64, num_layers=4, max_seq_len=128)
        self.assertEqual(formula_count, real_count)

    def test_more_layers_means_more_parameters(self):
        small = m.count_transformer_parameters(vocab_size=100, embed_dim=32, num_layers=2, max_seq_len=16)
        large = m.count_transformer_parameters(vocab_size=100, embed_dim=32, num_layers=4, max_seq_len=16)
        self.assertGreater(large, small)


class TestGenerateGreedy(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = m.GPTModel(vocab_size=50, embed_dim=32, num_layers=2,
                                 num_heads=4, max_seq_len=20)

    def test_extends_by_the_requested_number_of_tokens(self):
        input_ids = torch.randint(0, 50, (1, 3))
        result = m.generate_greedy(self.model, input_ids, max_new_tokens=4)
        self.assertEqual(result.shape, (1, 7))

    def test_preserves_the_original_prefix(self):
        input_ids = torch.randint(0, 50, (1, 3))
        result = m.generate_greedy(self.model, input_ids, max_new_tokens=2)
        self.assertTrue(torch.equal(result[:, :3], input_ids))

    def test_is_deterministic_given_the_same_model_and_input(self):
        input_ids = torch.randint(0, 50, (1, 4))
        result_a = m.generate_greedy(self.model, input_ids, max_new_tokens=3)
        result_b = m.generate_greedy(self.model, input_ids, max_new_tokens=3)
        self.assertTrue(torch.equal(result_a, result_b))

    def test_zero_new_tokens_returns_input_unchanged(self):
        input_ids = torch.randint(0, 50, (1, 5))
        result = m.generate_greedy(self.model, input_ids, max_new_tokens=0)
        self.assertTrue(torch.equal(result, input_ids))


if __name__ == "__main__":
    unittest.main()
