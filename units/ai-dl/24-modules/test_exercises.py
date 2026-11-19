"""Tests for 24 - nn.Module and model surgery."""

import importlib
import os
import unittest

import torch
from torch import nn

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestMLP(unittest.TestCase):
    def test_output_shape(self):
        model = m.MLP(4, [8, 16], 3)
        self.assertEqual(model(torch.randn(5, 4)).shape, (5, 3))

    def test_no_hidden_layers(self):
        model = m.MLP(4, [], 3)
        self.assertEqual(model(torch.randn(2, 4)).shape, (2, 3))

    def test_every_layer_is_registered(self):
        """A bare Python list of layers is invisible to .parameters()."""
        model = m.MLP(4, [8, 16], 3)
        expected = (4 * 8 + 8) + (8 * 16 + 16) + (16 * 3 + 3)
        self.assertEqual(sum(p.numel() for p in model.parameters()), expected)

    def test_submodules_appear_in_state_dict(self):
        model = m.MLP(4, [8], 3)
        self.assertEqual(len(model.state_dict()), 4)   # 2 layers x (w, b)

    def test_gradients_reach_the_first_layer(self):
        model = m.MLP(4, [8, 8], 2)
        model(torch.randn(3, 4)).sum().backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, f"{name} got no gradient")

    def test_moving_to_a_device_moves_every_layer(self):
        model = m.MLP(4, [8], 2).to(torch.float64)
        for parameter in model.parameters():
            self.assertEqual(parameter.dtype, torch.float64)

    def test_activation_is_applied_between_layers(self):
        """With ReLU and a negative-only input the hidden output is 0."""
        model = m.MLP(2, [4], 1)
        with torch.no_grad():
            for layer in model.layers:
                layer.weight.fill_(1.0)
                layer.bias.fill_(-100.0)
            model.head.weight.fill_(1.0)
            model.head.bias.zero_()
        self.assertEqual(model(torch.zeros(1, 2)).item(), 0.0)


class TestRunningMean(unittest.TestCase):
    def test_mean_is_a_buffer_not_a_parameter(self):
        module = m.RunningMean(3)
        names = [name for name, _ in module.named_parameters()]
        self.assertEqual(names, [])
        self.assertIn("mean", dict(module.named_buffers()))

    def test_buffers_are_in_the_state_dict(self):
        self.assertIn("mean", m.RunningMean(3).state_dict())

    def test_it_updates_in_training_mode(self):
        module = m.RunningMean(2, momentum=0.5)
        module.train()
        module(torch.tensor([[2.0, 4.0]]))
        torch.testing.assert_close(module.mean, torch.tensor([1.0, 2.0]))

    def test_it_does_not_update_in_eval_mode(self):
        module = m.RunningMean(2)
        module.eval()
        module(torch.tensor([[2.0, 4.0]]))
        torch.testing.assert_close(module.mean, torch.zeros(2))

    def test_forward_subtracts_the_running_mean(self):
        module = m.RunningMean(2)
        module.eval()
        with torch.no_grad():
            module.mean.copy_(torch.tensor([1.0, 1.0]))
        torch.testing.assert_close(module(torch.tensor([[3.0, 5.0]])),
                                   torch.tensor([[2.0, 4.0]]))

    def test_buffer_moves_with_the_module(self):
        module = m.RunningMean(2).to(torch.float64)
        self.assertEqual(module.mean.dtype, torch.float64)

    def test_the_buffer_does_not_collect_gradients(self):
        module = m.RunningMean(2)
        module.train()
        module(torch.randn(4, 2, requires_grad=True)).sum().backward()
        self.assertIsNone(module.mean.grad)


class TestFreeze(unittest.TestCase):
    def setUp(self):
        self.model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))

    def test_freezes_everything_by_default(self):
        m.freeze(self.model)
        self.assertFalse(any(p.requires_grad
                             for p in self.model.parameters()))

    def test_returns_the_count_of_frozen_scalars(self):
        count = m.freeze(self.model)
        self.assertEqual(count, (4 * 8 + 8) + (8 * 2 + 2))

    def test_prefix_freezes_only_that_subtree(self):
        m.freeze(self.model, prefix="0.")
        self.assertFalse(self.model[0].weight.requires_grad)
        self.assertTrue(self.model[1].weight.requires_grad)

    def test_prefix_count(self):
        self.assertEqual(m.freeze(self.model, prefix="0."), 4 * 8 + 8)

    def test_frozen_parameters_get_no_gradient(self):
        m.freeze(self.model, prefix="0.")
        self.model(torch.randn(2, 4)).sum().backward()
        self.assertIsNone(self.model[0].weight.grad)
        self.assertIsNotNone(self.model[1].weight.grad)

    def test_unmatched_prefix_freezes_nothing(self):
        self.assertEqual(m.freeze(self.model, prefix="nope"), 0)


class TestInitWeights(unittest.TestCase):
    def test_biases_are_zeroed(self):
        model = m.init_weights(nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2)))
        for layer in model:
            torch.testing.assert_close(layer.bias, torch.zeros_like(layer.bias))

    def test_weights_are_within_the_xavier_bound(self):
        torch.manual_seed(0)
        layer = nn.Linear(100, 100)
        m.init_weights(layer)
        bound = (6.0 / 200) ** 0.5
        self.assertLessEqual(layer.weight.abs().max().item(), bound + 1e-6)

    def test_weights_are_not_all_zero(self):
        layer = nn.Linear(10, 10)
        m.init_weights(layer)
        self.assertGreater(layer.weight.abs().sum().item(), 0.0)

    def test_it_recurses_into_nested_modules(self):
        model = nn.Sequential(nn.Sequential(nn.Linear(4, 4)), nn.Linear(4, 2))
        m.init_weights(model)
        torch.testing.assert_close(model[0][0].bias, torch.zeros(4))

    def test_non_linear_layers_are_left_alone(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
        before = model[1].weight.clone()
        m.init_weights(model)
        torch.testing.assert_close(model[1].weight, before)

    def test_gain_scales_the_bound(self):
        torch.manual_seed(0)
        small = nn.Linear(50, 50)
        m.init_weights(small, gain=0.01)
        self.assertLess(small.weight.abs().max().item(), 0.01)

    def test_a_layer_without_bias_does_not_crash(self):
        m.init_weights(nn.Linear(4, 4, bias=False))

    def test_returns_the_module(self):
        layer = nn.Linear(2, 2)
        self.assertIs(m.init_weights(layer), layer)


class TestTieWeights(unittest.TestCase):
    def test_the_tensors_are_the_same_object(self):
        embedding = nn.Embedding(100, 16)
        head = nn.Linear(16, 100, bias=False)
        m.tie_weights(embedding, head)
        self.assertIs(head.weight, embedding.weight)

    def test_updating_one_updates_the_other(self):
        embedding = nn.Embedding(10, 4)
        head = nn.Linear(4, 10, bias=False)
        m.tie_weights(embedding, head)
        with torch.no_grad():
            embedding.weight.add_(1.0)
        torch.testing.assert_close(head.weight, embedding.weight)

    def test_parameter_count_drops(self):
        embedding = nn.Embedding(10, 4)
        head = nn.Linear(4, 10, bias=False)
        model = nn.ModuleDict({"emb": embedding, "head": head})
        before = sum(p.numel() for p in model.parameters())
        m.tie_weights(embedding, head)
        after = sum(p.numel() for p in model.parameters())
        self.assertEqual(before - after, 40)

    def test_gradients_accumulate_from_both_uses(self):
        embedding = nn.Embedding(5, 3)
        head = nn.Linear(3, 5, bias=False)
        m.tie_weights(embedding, head)
        head(embedding(torch.tensor([0, 1]))).sum().backward()
        self.assertIsNotNone(embedding.weight.grad)

    def test_mismatched_shapes_raise(self):
        with self.assertRaises(ValueError):
            m.tie_weights(nn.Embedding(10, 4), nn.Linear(8, 10, bias=False))

    def test_returns_the_output_layer(self):
        embedding = nn.Embedding(4, 2)
        head = nn.Linear(2, 4, bias=False)
        self.assertIs(m.tie_weights(embedding, head), head)


class TestActivationStats(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

    def test_reports_every_leaf(self):
        stats = m.activation_stats(self.model, torch.randn(16, 4))
        self.assertEqual(sorted(stats), ["0", "1", "2"])

    def test_values_are_floats(self):
        stats = m.activation_stats(self.model, torch.randn(16, 4))
        self.assertTrue(all(isinstance(v, float) for v in stats.values()))

    def test_matches_a_manual_forward(self):
        x = torch.randn(16, 4)
        stats = m.activation_stats(self.model, x)
        with torch.no_grad():
            expected = self.model[0](x).std().item()
        self.assertAlmostEqual(stats["0"], expected, places=5)

    def test_hooks_are_removed_afterwards(self):
        """A leaked hook slows every later forward and holds memory."""
        m.activation_stats(self.model, torch.randn(4, 4))
        for layer in self.model.modules():
            self.assertEqual(len(layer._forward_hooks), 0)

    def test_hooks_are_removed_even_when_the_forward_raises(self):
        model = nn.Sequential(nn.Linear(4, 4))
        with self.assertRaises(RuntimeError):
            m.activation_stats(model, torch.randn(4, 99))
        for layer in model.modules():
            self.assertEqual(len(layer._forward_hooks), 0)

    def test_a_collapsing_network_shows_shrinking_stds(self):
        model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
        with torch.no_grad():
            for layer in model:
                layer.weight.mul_(0.01)
                layer.bias.zero_()
        stats = m.activation_stats(model, torch.randn(32, 8))
        self.assertLess(stats["1"], stats["0"])


class TestLoadPartial(unittest.TestCase):
    def test_matching_entries_are_loaded(self):
        model = nn.Linear(4, 2)
        source = {"weight": torch.ones(2, 4), "bias": torch.ones(2)}
        m.load_partial(model, source)
        torch.testing.assert_close(model.weight, torch.ones(2, 4))

    def test_mismatched_shapes_are_skipped_not_raised(self):
        model = nn.Linear(4, 2)
        before = model.weight.clone()
        skipped = m.load_partial(model, {"weight": torch.ones(10, 4)})
        self.assertEqual(skipped, ["weight"])
        torch.testing.assert_close(model.weight, before)

    def test_unknown_names_are_skipped(self):
        model = nn.Linear(4, 2)
        self.assertEqual(m.load_partial(model, {"nope": torch.ones(1)}),
                         ["nope"])

    def test_the_classic_new_head_case(self):
        """Backbone loads, the 1000-class head does not."""
        model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 10))
        source = {"0.weight": torch.ones(8, 4), "0.bias": torch.ones(8),
                  "1.weight": torch.ones(1000, 8), "1.bias": torch.ones(1000)}
        skipped = m.load_partial(model, source)
        self.assertEqual(skipped, ["1.bias", "1.weight"])
        torch.testing.assert_close(model[0].weight, torch.ones(8, 4))

    def test_nothing_skipped_returns_an_empty_list(self):
        model = nn.Linear(4, 2)
        self.assertEqual(m.load_partial(model, dict(model.state_dict())), [])

    def test_partially_loading_does_not_wipe_the_rest(self):
        model = nn.Linear(4, 2)
        bias_before = model.bias.clone()
        m.load_partial(model, {"weight": torch.ones(2, 4)})
        torch.testing.assert_close(model.bias, bias_before)


class TestModuleSummary(unittest.TestCase):
    def test_one_row_per_direct_child(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        self.assertEqual(len(m.module_summary(model)), 3)

    def test_names_and_types(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU())
        summary = m.module_summary(model)
        self.assertEqual(summary[0][:2], ("0", "Linear"))
        self.assertEqual(summary[1][:2], ("1", "ReLU"))

    def test_parameter_counts(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU())
        self.assertEqual(m.module_summary(model)[0][2], 4 * 8 + 8)
        self.assertEqual(m.module_summary(model)[1][2], 0)

    def test_counts_include_nested_parameters(self):
        model = nn.ModuleDict(
            {"block": nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))})
        self.assertEqual(m.module_summary(model)[0][2], 2 * (16 + 4))

    def test_does_not_recurse_into_grandchildren(self):
        model = nn.Sequential(nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4)))
        self.assertEqual(len(m.module_summary(model)), 1)

    def test_a_leaf_module_has_no_children(self):
        self.assertEqual(m.module_summary(nn.Linear(4, 2)), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
