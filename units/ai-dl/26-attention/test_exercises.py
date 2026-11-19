"""Tests for 26 - attention and transformer blocks.

Where PyTorch ships an equivalent (F.scaled_dot_product_attention,
nn.MultiheadAttention) the exercise is checked against it, so the tests
pin real behaviour rather than my arithmetic.
"""

import importlib
import math
import os
import unittest

import torch
import torch.nn.functional as F
from torch import nn

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestScaledDotProductAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.q, self.k, self.v = (torch.randn(2, 3, 5, 8) for _ in range(3))

    def test_output_shape(self):
        out, weights = m.scaled_dot_product_attention(self.q, self.k, self.v)
        self.assertEqual(out.shape, (2, 3, 5, 8))
        self.assertEqual(weights.shape, (2, 3, 5, 5))

    def test_matches_pytorch(self):
        out, _ = m.scaled_dot_product_attention(self.q, self.k, self.v)
        torch.testing.assert_close(
            out, F.scaled_dot_product_attention(self.q, self.k, self.v),
            atol=1e-5, rtol=1e-4)

    def test_weights_are_a_distribution(self):
        _, weights = m.scaled_dot_product_attention(self.q, self.k, self.v)
        torch.testing.assert_close(weights.sum(-1),
                                   torch.ones(2, 3, 5), atol=1e-5, rtol=0)
        self.assertTrue((weights >= 0).all())

    def test_it_divides_by_sqrt_dk(self):
        """Without the scale the logits - and so the weights - are wrong."""
        q = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
        k = torch.tensor([[[4.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]])
        v = torch.eye(2)[None]
        _, weights = m.scaled_dot_product_attention(q, k, v)
        expected = torch.softmax(torch.tensor([4.0 / math.sqrt(4), 0.0]), 0)
        torch.testing.assert_close(weights[0, 0], expected)

    def test_causal_mask_matches_pytorch(self):
        mask = m.causal_mask(5)
        out, _ = m.scaled_dot_product_attention(self.q, self.k, self.v, mask)
        torch.testing.assert_close(
            out, F.scaled_dot_product_attention(self.q, self.k, self.v,
                                                is_causal=True),
            atol=1e-5, rtol=1e-4)

    def test_masked_positions_get_zero_weight(self):
        mask = m.causal_mask(5)
        _, weights = m.scaled_dot_product_attention(self.q, self.k, self.v,
                                                    mask)
        self.assertEqual(weights[..., 0, 1:].abs().max().item(), 0.0)

    def test_masked_rows_still_sum_to_one(self):
        mask = m.causal_mask(5)
        _, weights = m.scaled_dot_product_attention(self.q, self.k, self.v,
                                                    mask)
        torch.testing.assert_close(weights.sum(-1), torch.ones(2, 3, 5),
                                   atol=1e-5, rtol=0)

    def test_the_first_row_attends_only_to_itself(self):
        mask = m.causal_mask(5)
        out, _ = m.scaled_dot_product_attention(self.q, self.k, self.v, mask)
        torch.testing.assert_close(out[..., 0, :], self.v[..., 0, :],
                                   atol=1e-5, rtol=1e-4)

    def test_it_is_differentiable(self):
        q = self.q.clone().requires_grad_(True)
        out, _ = m.scaled_dot_product_attention(q, self.k, self.v)
        out.sum().backward()
        self.assertFalse(torch.isnan(q.grad).any())


class TestCausalMask(unittest.TestCase):
    def test_shape_and_dtype(self):
        mask = m.causal_mask(4)
        self.assertEqual(mask.shape, (4, 4))
        self.assertEqual(mask.dtype, torch.bool)

    def test_the_diagonal_is_included(self):
        """Off-by-one here stops a token attending to itself."""
        self.assertTrue(m.causal_mask(4).diagonal().all())

    def test_it_is_lower_triangular(self):
        torch.testing.assert_close(m.causal_mask(4),
                                   torch.ones(4, 4, dtype=torch.bool).tril())

    def test_the_future_is_blocked(self):
        mask = m.causal_mask(4)
        self.assertFalse(mask[0, 1:].any())

    def test_the_last_row_sees_everything(self):
        self.assertTrue(m.causal_mask(4)[-1].all())

    def test_length_one(self):
        torch.testing.assert_close(m.causal_mask(1),
                                   torch.ones(1, 1, dtype=torch.bool))


class TestPaddingMask(unittest.TestCase):
    def test_shape(self):
        self.assertEqual(m.padding_mask(torch.tensor([2, 3]), 5).shape, (2, 5))

    def test_values(self):
        mask = m.padding_mask(torch.tensor([2, 3]), 4)
        torch.testing.assert_close(
            mask, torch.tensor([[True, True, False, False],
                                [True, True, True, False]]))

    def test_max_length_defaults_to_the_longest(self):
        self.assertEqual(m.padding_mask(torch.tensor([2, 5])).shape, (2, 5))

    def test_a_full_length_row_is_all_true(self):
        self.assertTrue(m.padding_mask(torch.tensor([4]), 4).all())

    def test_a_zero_length_row_is_all_false(self):
        self.assertFalse(m.padding_mask(torch.tensor([0]), 4).any())

    def test_it_composes_with_attention(self):
        torch.manual_seed(0)
        q, k, v = (torch.randn(2, 4, 8) for _ in range(3))
        mask = m.padding_mask(torch.tensor([2, 4]), 4)[:, None, :]
        _, weights = m.scaled_dot_product_attention(q, k, v, mask)
        self.assertEqual(weights[0, :, 2:].abs().max().item(), 0.0)


class TestMultiHeadAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.mha = m.MultiHeadAttention(16, 4)
        self.x = torch.randn(2, 5, 16)

    def test_output_shape(self):
        out, weights = self.mha(self.x)
        self.assertEqual(out.shape, (2, 5, 16))
        self.assertEqual(weights.shape, (2, 4, 5, 5))

    def test_indivisible_dims_raise(self):
        with self.assertRaises(ValueError):
            m.MultiHeadAttention(10, 4)

    def test_parameter_count(self):
        """Four square projections: 4 * (D*D + D)."""
        total = sum(p.numel() for p in self.mha.parameters())
        self.assertEqual(total, 4 * (16 * 16 + 16))

    def test_one_head_matches_plain_attention(self):
        mha = m.MultiHeadAttention(8, 1)
        x = torch.randn(1, 4, 8)
        out, _ = mha(x)
        manual, _ = m.scaled_dot_product_attention(
            mha.q_proj(x), mha.k_proj(x), mha.v_proj(x))
        torch.testing.assert_close(out, mha.out_proj(manual),
                                   atol=1e-5, rtol=1e-4)

    def test_heads_are_split_not_interleaved(self):
        """A view instead of a transpose mixes positions across heads.

        With a value projection that copies the input, head h of position i
        must read only channels [h*d, (h+1)*d) of position i.
        """
        mha = m.MultiHeadAttention(8, 2, bias=False)
        with torch.no_grad():
            mha.v_proj.weight.copy_(torch.eye(8))
            mha.out_proj.weight.copy_(torch.eye(8))
            # Force every query onto its own position only.
            mha.q_proj.weight.zero_()
            mha.k_proj.weight.zero_()
        x = torch.arange(8, dtype=torch.float32).repeat(1, 3, 1)
        out, _ = mha(x)
        # Uniform attention over identical positions returns that position.
        torch.testing.assert_close(out, x, atol=1e-5, rtol=1e-4)

    def test_causal_mask_is_broadcast_over_heads(self):
        out, weights = self.mha(self.x, m.causal_mask(5))
        self.assertEqual(weights.shape, (2, 4, 5, 5))
        self.assertEqual(weights[:, :, 0, 1:].abs().max().item(), 0.0)

    def test_permuting_the_sequence_permutes_the_output(self):
        """Attention has no positional information of its own."""
        out, _ = self.mha(self.x)
        flipped, _ = self.mha(self.x.flip(1))
        torch.testing.assert_close(flipped, out.flip(1), atol=1e-4, rtol=1e-4)

    def test_gradients_reach_every_projection(self):
        self.mha(self.x)[0].sum().backward()
        for name, parameter in self.mha.named_parameters():
            self.assertIsNotNone(parameter.grad, f"{name} got no gradient")


class TestKVCache(unittest.TestCase):
    def test_starts_empty(self):
        self.assertEqual(len(m.KVCache()), 0)

    def test_first_append_returns_what_it_was_given(self):
        cache = m.KVCache()
        k, v = torch.randn(1, 2, 1, 4), torch.randn(1, 2, 1, 4)
        keys, values = cache.append(k, v)
        torch.testing.assert_close(keys, k)
        torch.testing.assert_close(values, v)

    def test_it_grows_along_the_sequence_axis(self):
        cache = m.KVCache()
        for _ in range(3):
            keys, _ = cache.append(torch.randn(1, 2, 1, 4),
                                   torch.randn(1, 2, 1, 4))
        self.assertEqual(keys.shape, (1, 2, 3, 4))
        self.assertEqual(len(cache), 3)

    def test_history_is_in_order(self):
        cache = m.KVCache()
        for step in range(3):
            keys, _ = cache.append(torch.full((1, 1, 1, 2), float(step)),
                                   torch.zeros(1, 1, 1, 2))
        torch.testing.assert_close(keys[0, 0, :, 0],
                                   torch.tensor([0.0, 1.0, 2.0]))

    def test_reset(self):
        cache = m.KVCache()
        cache.append(torch.randn(1, 1, 1, 2), torch.randn(1, 1, 1, 2))
        cache.reset()
        self.assertEqual(len(cache), 0)

    def test_cached_decoding_matches_a_full_forward(self):
        """The whole point: one token at a time must equal the full pass."""
        torch.manual_seed(0)
        mha = m.MultiHeadAttention(8, 2)
        x = torch.randn(1, 4, 8)
        full, _ = mha(x, m.causal_mask(4))

        cache = m.KVCache()
        outputs = []
        for step in range(4):
            token = x[:, step:step + 1]
            q = mha._split(mha.q_proj(token))
            keys, values = cache.append(mha._split(mha.k_proj(token)),
                                        mha._split(mha.v_proj(token)))
            out, _ = m.scaled_dot_product_attention(q, keys, values)
            outputs.append(mha.out_proj(
                out.transpose(1, 2).reshape(1, 1, 8)))
        torch.testing.assert_close(torch.cat(outputs, dim=1), full,
                                   atol=1e-5, rtol=1e-4)


class TestRoPE(unittest.TestCase):
    def setUp(self):
        self.cos, self.sin = m.rope_frequencies(8, 16)

    def test_table_shapes(self):
        self.assertEqual(self.cos.shape, (16, 4))
        self.assertEqual(self.sin.shape, (16, 4))

    def test_odd_head_dim_raises(self):
        with self.assertRaises(ValueError):
            m.rope_frequencies(7, 16)

    def test_position_zero_is_the_identity(self):
        """cos(0)=1, sin(0)=0, so nothing rotates."""
        x = torch.randn(1, 1, 1, 8)
        torch.testing.assert_close(m.apply_rope(x, self.cos, self.sin), x,
                                   atol=1e-6, rtol=0)

    def test_shape_is_unchanged(self):
        x = torch.randn(2, 3, 5, 8)
        self.assertEqual(m.apply_rope(x, self.cos, self.sin).shape,
                         (2, 3, 5, 8))

    def test_rotation_preserves_the_norm(self):
        x = torch.randn(1, 2, 6, 8)
        rotated = m.apply_rope(x, self.cos, self.sin)
        torch.testing.assert_close(rotated.norm(dim=-1), x.norm(dim=-1),
                                   atol=1e-5, rtol=1e-5)

    def test_the_dot_product_depends_only_on_relative_position(self):
        """The property that makes RoPE work at all."""
        a, b = torch.randn(1, 1, 8), torch.randn(1, 1, 8)
        close = (m.apply_rope(a, self.cos, self.sin, offset=1)
                 * m.apply_rope(b, self.cos, self.sin, offset=4)).sum()
        far = (m.apply_rope(a, self.cos, self.sin, offset=5)
               * m.apply_rope(b, self.cos, self.sin, offset=8)).sum()
        torch.testing.assert_close(close, far, atol=1e-4, rtol=1e-4)

    def test_offset_actually_shifts_the_position(self):
        """Decoding one token at a time needs this; forgetting it is a bug."""
        x = torch.randn(1, 1, 1, 8)
        at_zero = m.apply_rope(x, self.cos, self.sin, offset=0)
        at_three = m.apply_rope(x, self.cos, self.sin, offset=3)
        self.assertFalse(torch.allclose(at_zero, at_three))

    def test_offset_matches_slicing_a_longer_sequence(self):
        x = torch.randn(1, 1, 6, 8)
        full = m.apply_rope(x, self.cos, self.sin)
        piece = m.apply_rope(x[:, :, 3:], self.cos, self.sin, offset=3)
        torch.testing.assert_close(piece, full[:, :, 3:], atol=1e-5, rtol=1e-5)


class TestTransformerBlock(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.block = m.TransformerBlock(16, 4)
        self.x = torch.randn(2, 5, 16)

    def test_output_shape_matches_the_input(self):
        self.assertEqual(self.block(self.x).shape, self.x.shape)

    def test_it_is_residual(self):
        """Zero the two sublayer outputs and the block is the identity."""
        with torch.no_grad():
            self.block.attention.out_proj.weight.zero_()
            self.block.attention.out_proj.bias.zero_()
            self.block.mlp[2].weight.zero_()
            self.block.mlp[2].bias.zero_()
        torch.testing.assert_close(self.block(self.x), self.x)

    def test_it_is_pre_norm(self):
        """Post-norm would normalise the block OUTPUT to zero mean."""
        out = self.block(self.x)
        self.assertGreater(out.mean(-1).abs().max().item(), 1e-3)

    def test_mask_is_accepted(self):
        self.assertEqual(self.block(self.x, m.causal_mask(5)).shape,
                         self.x.shape)

    def test_causality_holds_end_to_end(self):
        """Changing a later token must not change an earlier output."""
        mask = m.causal_mask(5)
        self.block.eval()
        out = self.block(self.x, mask)
        changed = self.x.clone()
        changed[:, 4] += 10.0
        torch.testing.assert_close(self.block(changed, mask)[:, :4],
                                   out[:, :4], atol=1e-4, rtol=1e-4)

    def test_the_mlp_expands_by_the_ratio(self):
        block = m.TransformerBlock(16, 4, mlp_ratio=4)
        self.assertEqual(block.mlp[0].out_features, 64)

    def test_gradients_flow_to_the_input(self):
        x = self.x.clone().requires_grad_(True)
        self.block(x).sum().backward()
        self.assertFalse(torch.isnan(x.grad).any())
        self.assertGreater(x.grad.abs().sum().item(), 0.0)


class TestAttentionFlops(unittest.TestCase):
    def test_known_value(self):
        self.assertEqual(m.attention_flops(4, 2), 2 * 16 * 2)

    def test_it_is_quadratic_in_sequence_length(self):
        self.assertEqual(m.attention_flops(200, 64),
                         4 * m.attention_flops(100, 64))

    def test_it_is_linear_in_embed_dim(self):
        self.assertEqual(m.attention_flops(64, 200),
                         2 * m.attention_flops(64, 100))

    def test_head_count_does_not_change_the_total(self):
        self.assertEqual(m.attention_flops(128, 512, num_heads=8),
                         m.attention_flops(128, 512, num_heads=1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
