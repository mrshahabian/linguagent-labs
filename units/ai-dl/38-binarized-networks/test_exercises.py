"""Tests for 38 - binarized and quantized neural networks."""

import importlib
import os
import unittest

import torch

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestBinarize(unittest.TestCase):
    def test_positive_values_map_to_positive_one(self):
        x = torch.tensor([0.1, 3.0, 100.0])
        result = m.binarize(x)
        self.assertTrue(torch.equal(result, torch.ones(3)))

    def test_negative_values_map_to_negative_one(self):
        x = torch.tensor([-0.1, -3.0, -100.0])
        result = m.binarize(x)
        self.assertTrue(torch.equal(result, -torch.ones(3)))

    def test_zero_maps_to_a_defined_sign(self):
        x = torch.tensor([0.0])
        result = m.binarize(x)
        self.assertIn(result.item(), [-1.0, 1.0])

    def test_output_is_always_exactly_plus_or_minus_one(self):
        torch.manual_seed(0)
        x = torch.randn(100) * 10
        result = m.binarize(x)
        self.assertTrue(torch.all((result == 1.0) | (result == -1.0)))

    def test_gradient_passes_through_within_the_clip_range(self):
        x = torch.tensor([-0.5, 0.0, 0.3], requires_grad=True)
        y = m.binarize(x)
        y.sum().backward()
        self.assertTrue(torch.allclose(x.grad, torch.ones(3)))

    def test_gradient_is_zeroed_outside_the_clip_range(self):
        x = torch.tensor([-5.0, 0.2, 5.0], requires_grad=True)
        y = m.binarize(x)
        y.sum().backward()
        self.assertTrue(torch.allclose(x.grad, torch.tensor([0.0, 1.0, 0.0])))


class TestBinaryLinear(unittest.TestCase):
    def test_output_shape(self):
        torch.manual_seed(0)
        layer = m.BinaryLinear(4, 3)
        result = layer(torch.randn(2, 4))
        self.assertEqual(result.shape, (2, 3))

    def test_latent_weight_stays_real_valued(self):
        torch.manual_seed(1)
        layer = m.BinaryLinear(8, 4)
        layer(torch.randn(1, 8))
        self.assertTrue((layer.weight.abs() < 1).all())

    def test_weight_gradient_flows_through_ste(self):
        torch.manual_seed(2)
        layer = m.BinaryLinear(4, 2)
        result = layer(torch.randn(3, 4))
        result.sum().backward()
        self.assertIsNotNone(layer.weight.grad)
        self.assertFalse(torch.all(layer.weight.grad == 0))

    def test_forward_actually_uses_binarized_weights(self):
        torch.manual_seed(3)
        layer = m.BinaryLinear(4, 2)
        x = torch.randn(1, 4)
        expected = x @ m.binarize(layer.weight).T
        self.assertTrue(torch.allclose(layer(x), expected))


class TestXnorPopcountDot(unittest.TestCase):
    def test_matches_direct_dot_product(self):
        torch.manual_seed(0)
        a = (torch.randint(0, 2, (16,)).float() * 2 - 1)
        b = (torch.randint(0, 2, (16,)).float() * 2 - 1)
        direct = torch.dot(a, b).item()
        self.assertEqual(m.xnor_popcount_dot(a, b), direct)

    def test_identical_vectors_give_full_agreement(self):
        a = torch.tensor([1.0, -1.0, 1.0, -1.0])
        self.assertEqual(m.xnor_popcount_dot(a, a), 4)

    def test_opposite_vectors_give_full_disagreement(self):
        a = torch.tensor([1.0, -1.0, 1.0, -1.0])
        self.assertEqual(m.xnor_popcount_dot(a, -a), -4)

    def test_half_agreement_gives_zero(self):
        a = torch.tensor([1.0, 1.0, -1.0, -1.0])
        b = torch.tensor([1.0, -1.0, 1.0, -1.0])
        self.assertEqual(m.xnor_popcount_dot(a, b), 0)


class TestMemoryFootprintBytes(unittest.TestCase):
    def test_float32_is_four_bytes_per_parameter(self):
        float32_bytes, _ = m.memory_footprint_bytes(1000)
        self.assertEqual(float32_bytes, 4000)

    def test_binary_is_one_bit_per_parameter(self):
        _, binary_bytes = m.memory_footprint_bytes(1_000_000)
        self.assertEqual(binary_bytes, 125_000)

    def test_roughly_32x_compression(self):
        float32_bytes, binary_bytes = m.memory_footprint_bytes(8_000_000)
        self.assertAlmostEqual(float32_bytes / binary_bytes, 32.0, places=2)

    def test_rounds_up_to_a_whole_byte(self):
        _, binary_bytes = m.memory_footprint_bytes(1)
        self.assertEqual(binary_bytes, 1)


if __name__ == "__main__":
    unittest.main()
