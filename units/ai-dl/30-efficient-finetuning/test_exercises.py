"""Tests for 30 - efficient fine-tuning and inference."""

import importlib
import os
import unittest

import torch
from torch import nn

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestLoRALinear(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.base = nn.Linear(8, 4)
        self.x = torch.randn(3, 8)

    def test_matches_base_at_initialisation(self):
        # lora_B starts at zero, so the LoRA path contributes nothing yet.
        lora = m.LoRALinear(self.base, rank=2, alpha=4.0)
        self.assertTrue(torch.allclose(lora(self.x), self.base(self.x)))

    def test_base_parameters_are_frozen(self):
        lora = m.LoRALinear(self.base, rank=2, alpha=4.0)
        self.assertFalse(lora.base.weight.requires_grad)

    def test_lora_parameters_are_trainable(self):
        lora = m.LoRALinear(self.base, rank=2, alpha=4.0)
        self.assertTrue(lora.lora_A.requires_grad)
        self.assertTrue(lora.lora_B.requires_grad)

    def test_trainable_parameter_count_is_low_rank(self):
        lora = m.LoRALinear(self.base, rank=2, alpha=4.0)
        trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
        # rank(2) * in(8) + out(4) * rank(2) = 16 + 8 = 24, not 8*4=32 for full fine-tune
        self.assertEqual(trainable, 24)

    def test_output_changes_once_lora_b_is_nonzero(self):
        lora = m.LoRALinear(self.base, rank=2, alpha=4.0)
        with torch.no_grad():
            lora.lora_B.add_(0.5)
        self.assertFalse(torch.allclose(lora(self.x), self.base(self.x)))

    def test_merge_matches_unmerged_forward(self):
        lora = m.LoRALinear(self.base, rank=2, alpha=4.0)
        with torch.no_grad():
            lora.lora_B.add_(torch.randn_like(lora.lora_B) * 0.1)
        expected = lora(self.x)
        merged = lora.merge()
        self.assertTrue(torch.allclose(merged(self.x), expected, atol=1e-5))

    def test_merged_layer_is_plain_linear(self):
        lora = m.LoRALinear(self.base, rank=2, alpha=4.0)
        merged = lora.merge()
        self.assertIsInstance(merged, nn.Linear)
        self.assertFalse(hasattr(merged, "lora_A"))


class TestInt8Quantization(unittest.TestCase):
    def test_quantized_dtype_is_int8(self):
        t = torch.randn(50) * 3
        q, _ = m.quantize_int8(t)
        self.assertEqual(q.dtype, torch.int8)

    def test_values_stay_within_int8_range(self):
        t = torch.randn(200) * 100
        q, _ = m.quantize_int8(t)
        self.assertGreaterEqual(q.min().item(), -127)
        self.assertLessEqual(q.max().item(), 127)

    def test_max_magnitude_value_maps_near_127(self):
        t = torch.tensor([-10.0, 3.0, 10.0, -2.0])
        q, _ = m.quantize_int8(t)
        self.assertEqual(q.abs().max().item(), 127)

    def test_dequantize_is_close_to_original(self):
        torch.manual_seed(2)
        t = torch.randn(500) * 5
        q, scale = m.quantize_int8(t)
        dequantized = m.dequantize_int8(q, scale)
        # error should be bounded by roughly one quantization step (the scale)
        self.assertLess((t - dequantized).abs().max().item(), scale.item() * 1.01)

    def test_zero_stays_zero(self):
        t = torch.tensor([0.0, 5.0, -5.0])
        q, scale = m.quantize_int8(t)
        dequantized = m.dequantize_int8(q, scale)
        self.assertAlmostEqual(dequantized[0].item(), 0.0, places=5)


class TestCheckpoint(unittest.TestCase):
    def test_gradient_matches_direct_backward(self):
        torch.manual_seed(0)

        def fn(x):
            return (x * 2).sin().sum()

        x1 = torch.randn(6, requires_grad=True)
        x2 = x1.clone().detach().requires_grad_(True)

        fn(x1).backward()
        m.checkpoint(fn, x2).backward()

        self.assertTrue(torch.allclose(x1.grad, x2.grad, atol=1e-6))

    def test_fn_is_called_twice_forward_and_backward(self):
        calls = {"n": 0}

        def fn(x):
            calls["n"] += 1
            return (x ** 2).sum()

        x = torch.randn(4, requires_grad=True)
        y = m.checkpoint(fn, x)
        self.assertEqual(calls["n"], 1)   # just the forward pass so far
        y.backward()
        self.assertEqual(calls["n"], 2)   # backward recomputed it once more

    def test_forward_runs_without_grad_tracking(self):
        seen_grad_enabled = {}

        def fn(x):
            seen_grad_enabled["forward"] = torch.is_grad_enabled()
            return x.sum()

        x = torch.randn(3, requires_grad=True)
        m.checkpoint(fn, x)
        # inside the checkpointed forward, autograd tracking is off
        self.assertFalse(seen_grad_enabled["forward"])


class TestDynamicBatches(unittest.TestCase):
    def test_flushes_on_size_cap(self):
        arrivals = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
        batches = m.dynamic_batches(arrivals, max_batch_size=3, max_wait=100.0)
        self.assertEqual(batches, [[0, 1, 2], [3, 4, 5]])

    def test_flushes_on_wait_cap(self):
        # first batch: 0.0, 0.1 accumulate, then 5.0 blows the 1.0s wait
        # budget and flushes all three together; a fresh batch then starts
        # at 5.1, and the trailing 5.2 arrival joins it under the leftover
        # flush at the end of input.
        arrivals = [0.0, 0.1, 5.0, 5.1, 5.2]
        batches = m.dynamic_batches(arrivals, max_batch_size=10, max_wait=1.0)
        self.assertEqual(batches, [[0, 1, 2], [3, 4]])

    def test_leftover_partial_batch_is_still_returned(self):
        arrivals = [0.0, 0.1]
        batches = m.dynamic_batches(arrivals, max_batch_size=5, max_wait=100.0)
        self.assertEqual(batches, [[0, 1]])

    def test_empty_input(self):
        self.assertEqual(m.dynamic_batches([], max_batch_size=5, max_wait=1.0), [])

    def test_every_index_appears_exactly_once(self):
        arrivals = [0.0, 0.1, 0.3, 0.9, 1.5, 1.6, 1.65, 3.0]
        batches = m.dynamic_batches(arrivals, max_batch_size=3, max_wait=0.5)
        flat = [i for batch in batches for i in batch]
        self.assertEqual(flat, list(range(len(arrivals))))


if __name__ == "__main__":
    unittest.main()
