"""Tests for 29 - optimizers and learning-rate schedules."""

import importlib
import math
import os
import unittest

import torch

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


def quadratic_grad(param):
    """grad of 0.5 * sum(param**2) - the simplest well-behaved test loss."""
    return param


class TestSgdMomentumStep(unittest.TestCase):
    def test_matches_torch_optim_sgd(self):
        torch.manual_seed(0)
        p_torch = torch.tensor([3.0, -2.0], requires_grad=True)
        opt = torch.optim.SGD([p_torch], lr=0.1, momentum=0.9)

        param = torch.tensor([3.0, -2.0])
        velocity = torch.zeros_like(param)
        for _ in range(5):
            grad = quadratic_grad(param)
            param, velocity = m.sgd_momentum_step(param, grad, velocity, lr=0.1, momentum=0.9)

            opt.zero_grad()
            (0.5 * (p_torch ** 2).sum()).backward()
            opt.step()

        self.assertTrue(torch.allclose(param, p_torch.detach(), atol=1e-5))

    def test_zero_momentum_is_plain_gradient_descent(self):
        param = torch.tensor([4.0])
        velocity = torch.zeros_like(param)
        new_param, new_velocity = m.sgd_momentum_step(param, torch.tensor([2.0]), velocity, lr=0.5, momentum=0.0)
        self.assertTrue(torch.allclose(new_param, torch.tensor([3.0])))

    def test_does_not_mutate_inputs(self):
        param = torch.tensor([1.0])
        velocity = torch.tensor([0.5])
        param_copy, velocity_copy = param.clone(), velocity.clone()
        m.sgd_momentum_step(param, torch.tensor([0.1]), velocity, lr=0.1, momentum=0.9)
        self.assertTrue(torch.equal(param, param_copy))
        self.assertTrue(torch.equal(velocity, velocity_copy))

    def test_velocity_accumulates_across_steps(self):
        param = torch.tensor([0.0])
        velocity = torch.zeros_like(param)
        grad = torch.tensor([1.0])
        _, v1 = m.sgd_momentum_step(param, grad, velocity, lr=0.1, momentum=0.9)
        _, v2 = m.sgd_momentum_step(param, grad, v1, lr=0.1, momentum=0.9)
        self.assertGreater(v2.item(), v1.item())


class TestAdamStep(unittest.TestCase):
    def test_matches_torch_optim_adam(self):
        torch.manual_seed(0)
        p_torch = torch.tensor([3.0, -2.0, 0.5], requires_grad=True)
        opt = torch.optim.Adam([p_torch], lr=0.05, betas=(0.9, 0.999), eps=1e-8)

        param = torch.tensor([3.0, -2.0, 0.5])
        moment1 = torch.zeros_like(param)
        moment2 = torch.zeros_like(param)
        for step in range(1, 11):
            grad = quadratic_grad(param)
            param, moment1, moment2 = m.adam_step(param, grad, moment1, moment2, step, lr=0.05)

            opt.zero_grad()
            (0.5 * (p_torch ** 2).sum()).backward()
            opt.step()

        self.assertTrue(torch.allclose(param, p_torch.detach(), atol=1e-4))

    def test_first_step_is_bias_corrected_not_tiny(self):
        # With NO bias correction, the effective step at t=1 would be far
        # smaller than lr, because m and v start at (1 - beta) * grad and
        # (1 - beta2) * grad**2 respectively - dividing m by sqrt(v) alone
        # (uncorrected) still gives roughly the right direction and scale,
        # so this checks the corrected step lands close to a full lr step
        # in sign and rough magnitude, not the raw uncorrected estimate.
        param = torch.tensor([1.0])
        m0 = torch.zeros_like(param)
        v0 = torch.zeros_like(param)
        grad = torch.tensor([1.0])
        new_param, _, _ = m.adam_step(param, grad, m0, v0, t=1, lr=0.1)
        step_taken = (param - new_param).abs().item()
        self.assertAlmostEqual(step_taken, 0.1, places=3)


class TestAdamwStep(unittest.TestCase):
    def test_matches_torch_optim_adamw(self):
        torch.manual_seed(0)
        p_torch = torch.tensor([3.0, -2.0, 0.5], requires_grad=True)
        opt = torch.optim.AdamW([p_torch], lr=0.05, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)

        param = torch.tensor([3.0, -2.0, 0.5])
        moment1 = torch.zeros_like(param)
        moment2 = torch.zeros_like(param)
        for step in range(1, 11):
            grad = quadratic_grad(param)
            param, moment1, moment2 = m.adamw_step(param, grad, moment1, moment2, step, lr=0.05, weight_decay=0.1)

            opt.zero_grad()
            (0.5 * (p_torch ** 2).sum()).backward()
            opt.step()

        self.assertTrue(torch.allclose(param, p_torch.detach(), atol=1e-4))

    def test_decoupled_decay_differs_from_zero_decay(self):
        param = torch.tensor([2.0])
        m0, v0 = torch.zeros_like(param), torch.zeros_like(param)
        grad = torch.tensor([0.0])  # zero gradient - only decay should move the param
        new_param, _, _ = m.adamw_step(param, grad, m0, v0, t=1, lr=0.1, weight_decay=0.5)
        # with zero gradient, only the -lr * weight_decay * param term should fire
        self.assertAlmostEqual(new_param.item(), 2.0 - 0.1 * 0.5 * 2.0, places=5)

    def test_zero_weight_decay_matches_plain_adam(self):
        param = torch.tensor([3.0, -2.0])
        m0, v0 = torch.zeros_like(param), torch.zeros_like(param)
        grad = quadratic_grad(param)
        adam_param, _, _ = m.adam_step(param, grad, m0, v0, t=1, lr=0.05)
        adamw_param, _, _ = m.adamw_step(param, grad, m0, v0, t=1, lr=0.05, weight_decay=0.0)
        self.assertTrue(torch.allclose(adam_param, adamw_param, atol=1e-6))


class TestCosineWarmupLr(unittest.TestCase):
    def test_starts_at_zero(self):
        self.assertEqual(m.cosine_warmup_lr(0, 100, 10, base_lr=1.0), 0.0)

    def test_linear_during_warmup(self):
        self.assertAlmostEqual(m.cosine_warmup_lr(5, 100, 10, base_lr=1.0), 0.5)

    def test_reaches_base_lr_at_end_of_warmup(self):
        self.assertAlmostEqual(m.cosine_warmup_lr(10, 100, 10, base_lr=1.0), 1.0)

    def test_midpoint_of_cosine_decay(self):
        # halfway between warmup_steps and total_steps: cosine(pi/2 turn) = 0.5
        self.assertAlmostEqual(m.cosine_warmup_lr(55, 100, 10, base_lr=1.0), 0.5, places=5)

    def test_reaches_min_lr_at_total_steps(self):
        self.assertAlmostEqual(m.cosine_warmup_lr(100, 100, 10, base_lr=1.0, min_lr=0.1), 0.1)

    def test_stays_at_min_lr_past_total_steps(self):
        self.assertAlmostEqual(m.cosine_warmup_lr(500, 100, 10, base_lr=1.0, min_lr=0.1), 0.1)

    def test_no_warmup_starts_at_base_lr(self):
        self.assertAlmostEqual(m.cosine_warmup_lr(0, 100, 0, base_lr=1.0), 1.0)


class TestClipGradNorm(unittest.TestCase):
    def test_matches_torch_nn_utils_clip_grad_norm(self):
        torch.manual_seed(1)
        g1, g2 = torch.randn(4), torch.randn(3)
        grads = [g1.clone(), g2.clone()]
        total = m.clip_grad_norm_(grads, max_norm=1.0)

        p1 = torch.nn.Parameter(torch.zeros(4)); p1.grad = g1.clone()
        p2 = torch.nn.Parameter(torch.zeros(3)); p2.grad = g2.clone()
        torch_total = torch.nn.utils.clip_grad_norm_([p1, p2], max_norm=1.0)

        self.assertTrue(torch.allclose(total, torch_total, atol=1e-5))
        self.assertTrue(torch.allclose(grads[0], p1.grad, atol=1e-5))
        self.assertTrue(torch.allclose(grads[1], p2.grad, atol=1e-5))

    def test_clips_in_place(self):
        g = torch.tensor([3.0, 4.0])  # norm = 5
        grads = [g]
        m.clip_grad_norm_(grads, max_norm=1.0)
        self.assertTrue(torch.allclose(g, grads[0]))
        self.assertAlmostEqual(g.norm().item(), 1.0, places=4)

    def test_no_clipping_when_under_max_norm(self):
        g = torch.tensor([0.01, 0.01])
        original = g.clone()
        grads = [g]
        m.clip_grad_norm_(grads, max_norm=10.0)
        self.assertTrue(torch.allclose(grads[0], original))

    def test_returns_total_norm_before_clipping(self):
        g = torch.tensor([3.0, 4.0])  # norm = 5
        total = m.clip_grad_norm_([g], max_norm=1.0)
        self.assertAlmostEqual(total.item(), 5.0, places=4)


if __name__ == "__main__":
    unittest.main()
