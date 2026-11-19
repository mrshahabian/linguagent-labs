"""Tests for 23 - autograd.

Gradients are checked against closed forms or against a central-difference
approximation in float64 - never against a hard-coded number that would go
stale if the formula changed.
"""

import importlib
import os
import unittest

import torch

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestGradOf(unittest.TestCase):
    def test_known_derivative(self):
        """d/dx of x**3 is 3x**2."""
        x = torch.tensor([1.0, 2.0, 3.0])
        torch.testing.assert_close(m.grad_of(lambda t: t ** 3, x),
                                   3 * x ** 2)

    def test_of_a_sum_reduction(self):
        x = torch.tensor([1.0, 2.0])
        torch.testing.assert_close(m.grad_of(lambda t: t.sum(), x),
                                   torch.ones(2))

    def test_leaves_grad_alone(self):
        """A functional gradient must not touch .grad."""
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        m.grad_of(lambda t: (t ** 2).sum(), x)
        self.assertIsNone(x.grad)

    def test_works_on_a_tensor_that_does_not_require_grad(self):
        x = torch.tensor([2.0])
        torch.testing.assert_close(m.grad_of(lambda t: t ** 2, x),
                                   torch.tensor([4.0]))

    def test_multidimensional(self):
        x = torch.randn(3, 4)
        torch.testing.assert_close(m.grad_of(lambda t: (t ** 2).sum(), x),
                                   2 * x)


class TestAccumulation(unittest.TestCase):
    def test_two_backwards_double_the_gradient(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        torch.testing.assert_close(m.accumulate_twice(x), 2 * (2 * x))

    def test_it_really_is_accumulation_not_the_last_value(self):
        x = torch.tensor([5.0])
        self.assertEqual(m.accumulate_twice(x).item(), 20.0)

    def test_does_not_mutate_the_caller(self):
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        m.accumulate_twice(x)
        self.assertIsNone(x.grad)


class TestFrozenForward(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = torch.nn.Linear(4, 2)

    def test_output_matches_a_normal_forward(self):
        x = torch.randn(3, 4)
        torch.testing.assert_close(m.frozen_forward(self.model, x),
                                   self.model(x))

    def test_result_has_no_graph(self):
        out = m.frozen_forward(self.model, torch.randn(3, 4))
        self.assertFalse(out.requires_grad)
        self.assertIsNone(out.grad_fn)

    def test_parameters_are_not_modified(self):
        before = self.model.weight.clone()
        m.frozen_forward(self.model, torch.randn(3, 4))
        torch.testing.assert_close(self.model.weight, before)

    def test_does_not_leave_grad_disabled(self):
        """A bare torch.set_grad_enabled(False) would leak out of the call."""
        m.frozen_forward(self.model, torch.randn(3, 4))
        self.assertTrue(torch.is_grad_enabled())

    def test_does_not_flip_training_mode(self):
        """no_grad is not eval() - this function should not change the mode."""
        self.model.train()
        m.frozen_forward(self.model, torch.randn(3, 4))
        self.assertTrue(self.model.training)


class TestDetachVsClone(unittest.TestCase):
    def setUp(self):
        self.x = (torch.tensor([1.0, 2.0, 3.0], requires_grad=True) * 2)

    def test_detached_has_no_graph(self):
        detached, _ = m.detach_vs_clone(self.x)
        self.assertFalse(detached.requires_grad)

    def test_clone_keeps_the_graph(self):
        _, cloned = m.detach_vs_clone(self.x)
        self.assertTrue(cloned.requires_grad)

    def test_detached_shares_storage(self):
        detached, _ = m.detach_vs_clone(self.x)
        self.assertEqual(detached.data_ptr(), self.x.data_ptr())

    def test_clone_does_not_share_storage(self):
        _, cloned = m.detach_vs_clone(self.x)
        self.assertNotEqual(cloned.data_ptr(), self.x.data_ptr())

    def test_values_are_equal(self):
        detached, cloned = m.detach_vs_clone(self.x)
        torch.testing.assert_close(detached, self.x.detach())
        torch.testing.assert_close(cloned.detach(), self.x.detach())

    def test_gradient_flows_through_the_clone(self):
        leaf = torch.tensor([1.0, 2.0], requires_grad=True)
        _, cloned = m.detach_vs_clone(leaf * 3)
        cloned.sum().backward()
        torch.testing.assert_close(leaf.grad, torch.full((2,), 3.0))


class TestClipGradNorm(unittest.TestCase):
    def make(self, *values):
        params = []
        for value in values:
            p = torch.nn.Parameter(torch.zeros(len(value)))
            p.grad = torch.tensor(value)
            params.append(p)
        return params

    def test_returns_the_norm_before_clipping(self):
        params = self.make([3.0, 4.0])
        self.assertAlmostEqual(m.clip_grad_norm(params, 1.0).item(), 5.0,
                               places=5)

    def test_small_gradients_are_untouched(self):
        params = self.make([0.3, 0.4])
        m.clip_grad_norm(params, 10.0)
        torch.testing.assert_close(params[0].grad, torch.tensor([0.3, 0.4]))

    def test_large_gradients_are_scaled_to_max_norm(self):
        params = self.make([3.0, 4.0])
        m.clip_grad_norm(params, 1.0)
        self.assertAlmostEqual(params[0].grad.norm().item(), 1.0, places=4)

    def test_the_direction_is_preserved(self):
        params = self.make([3.0, 4.0])
        m.clip_grad_norm(params, 1.0)
        torch.testing.assert_close(params[0].grad,
                                   torch.tensor([0.6, 0.8]), atol=1e-4,
                                   rtol=0)

    def test_the_norm_is_global_across_parameters(self):
        """Two tensors of norm 3 and 4 have a GLOBAL norm of 5, not 4."""
        params = self.make([3.0], [4.0])
        self.assertAlmostEqual(m.clip_grad_norm(params, 100.0).item(), 5.0,
                               places=5)

    def test_all_parameters_share_one_scale_factor(self):
        params = self.make([3.0], [4.0])
        m.clip_grad_norm(params, 1.0)
        combined = torch.cat([p.grad for p in params])
        self.assertAlmostEqual(combined.norm().item(), 1.0, places=4)
        torch.testing.assert_close(combined, torch.tensor([0.6, 0.8]),
                                   atol=1e-4, rtol=0)

    def test_it_clips_in_place(self):
        params = self.make([3.0, 4.0])
        original = params[0].grad
        m.clip_grad_norm(params, 1.0)
        self.assertIs(params[0].grad, original)

    def test_parameters_without_gradients_are_skipped(self):
        params = self.make([3.0, 4.0])
        params.append(torch.nn.Parameter(torch.zeros(2)))   # grad is None
        self.assertAlmostEqual(m.clip_grad_norm(params, 10.0).item(), 5.0,
                               places=5)

    def test_matches_torch(self):
        params = self.make([3.0, 4.0], [1.0, 2.0])
        reference = [torch.nn.Parameter(p.detach().clone()) for p in params]
        for r, p in zip(reference, params):
            r.grad = p.grad.clone()
        mine = m.clip_grad_norm(params, 1.5)
        theirs = torch.nn.utils.clip_grad_norm_(reference, 1.5)
        self.assertAlmostEqual(mine.item(), theirs.item(), places=4)
        for r, p in zip(reference, params):
            torch.testing.assert_close(p.grad, r.grad, atol=1e-4, rtol=1e-4)


class TestSquareFunction(unittest.TestCase):
    def test_forward(self):
        x = torch.tensor([2.0, 3.0])
        torch.testing.assert_close(m.Square.apply(x), torch.tensor([4.0, 9.0]))

    def test_backward_matches_the_closed_form(self):
        x = torch.tensor([2.0, 3.0], requires_grad=True)
        m.Square.apply(x).sum().backward()
        torch.testing.assert_close(x.grad, torch.tensor([4.0, 6.0]))

    def test_matches_the_builtin_operator(self):
        x = torch.randn(5, requires_grad=True)
        reference = x.detach().clone().requires_grad_(True)
        m.Square.apply(x).sum().backward()
        (reference ** 2).sum().backward()
        torch.testing.assert_close(x.grad, reference.grad)

    def test_passes_gradcheck(self):
        x = torch.randn(4, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(m.Square.apply, (x,)))

    def test_chains_with_other_ops(self):
        x = torch.tensor([2.0], requires_grad=True)
        (m.Square.apply(x) * 3).sum().backward()
        torch.testing.assert_close(x.grad, torch.tensor([12.0]))


class TestStraightThrough(unittest.TestCase):
    def test_forward_rounds(self):
        x = torch.tensor([0.2, 0.7, -1.4])
        torch.testing.assert_close(m.StraightThroughRound.apply(x),
                                   torch.tensor([0.0, 1.0, -1.0]))

    def test_backward_is_the_identity(self):
        x = torch.tensor([0.2, 0.7], requires_grad=True)
        m.StraightThroughRound.apply(x).sum().backward()
        torch.testing.assert_close(x.grad, torch.ones(2))

    def test_a_plain_round_would_give_zero_gradient(self):
        """Establishes why the estimator is needed at all."""
        x = torch.tensor([0.2, 0.7], requires_grad=True)
        torch.round(x).sum().backward()
        torch.testing.assert_close(x.grad, torch.zeros(2))

    def test_gradient_is_scaled_by_the_chain(self):
        x = torch.tensor([0.7], requires_grad=True)
        (m.StraightThroughRound.apply(x) * 5).sum().backward()
        torch.testing.assert_close(x.grad, torch.tensor([5.0]))


class TestNumericGrad(unittest.TestCase):
    def test_matches_autograd_on_a_polynomial(self):
        def f(t):
            return (t ** 3).sum()

        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        torch.testing.assert_close(m.numeric_grad(f, x), 3 * x ** 2,
                                   atol=1e-5, rtol=1e-5)

    def test_matches_autograd_on_a_network(self):
        torch.manual_seed(0)
        model = torch.nn.Linear(3, 1).double()

        def f(t):
            return model(t).sum()

        x = torch.randn(2, 3, dtype=torch.float64)
        analytic = m.grad_of(f, x)
        torch.testing.assert_close(m.numeric_grad(f, x), analytic,
                                   atol=1e-5, rtol=1e-5)

    def test_shape_matches_the_input(self):
        x = torch.randn(2, 3, dtype=torch.float64)
        self.assertEqual(m.numeric_grad(lambda t: t.sum(), x).shape, (2, 3))

    def test_does_not_mutate_the_input(self):
        x = torch.tensor([1.0, 2.0], dtype=torch.float64)
        before = x.clone()
        m.numeric_grad(lambda t: (t ** 2).sum(), x)
        torch.testing.assert_close(x, before)


class TestGradientPenalty(unittest.TestCase):
    def test_known_value_for_a_linear_map(self):
        """For y = Wx the input-gradient is W, so the penalty is ||W||^2."""
        layer = torch.nn.Linear(3, 1, bias=False)
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([[1.0, 2.0, 2.0]]))
        penalty = m.gradient_penalty(layer, torch.randn(4, 3))
        self.assertAlmostEqual(penalty.item(), 9.0, places=4)

    def test_penalty_is_differentiable(self):
        """create_graph=True is what makes this possible at all."""
        layer = torch.nn.Linear(3, 1, bias=False)
        m.gradient_penalty(layer, torch.randn(2, 3)).backward()
        self.assertIsNotNone(layer.weight.grad)
        self.assertTrue((layer.weight.grad.abs() > 0).any())

    def test_gradient_wrt_weights_is_correct(self):
        """d/dW of ||W||^2 is 2W."""
        layer = torch.nn.Linear(3, 1, bias=False)
        weights = layer.weight.detach().clone()
        m.gradient_penalty(layer, torch.randn(2, 3)).backward()
        torch.testing.assert_close(layer.weight.grad, 2 * weights,
                                   atol=1e-5, rtol=1e-4)

    def test_returns_a_scalar(self):
        penalty = m.gradient_penalty(torch.nn.Linear(3, 1), torch.randn(5, 3))
        self.assertEqual(penalty.shape, ())

    def test_zero_for_a_constant_discriminator(self):
        class Constant(torch.nn.Module):
            def forward(self, x):
                return (x * 0).sum(-1, keepdim=True)

        penalty = m.gradient_penalty(Constant(), torch.randn(3, 4))
        self.assertAlmostEqual(penalty.item(), 0.0, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
