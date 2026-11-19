"""Tests for 25 - data loading and the training loop.

Everything runs on tiny tensors for a handful of steps. Where a test is
about learning it asserts the loss went DOWN, never that it reached a
particular value - that would be a seed-dependent assertion pretending to
be a correctness one.
"""

import importlib
import os
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


def tiny_problem(n=32, d=4, classes=3, seed=0):
    torch.manual_seed(seed)
    return torch.randn(n, d), torch.randint(0, classes, (n,))


class TestArrayDataset(unittest.TestCase):
    def test_length(self):
        self.assertEqual(len(m.ArrayDataset(torch.zeros(7, 2),
                                            torch.zeros(7))), 7)

    def test_returns_a_single_example_not_a_batch(self):
        features, labels = tiny_problem(5)
        item, label = m.ArrayDataset(features, labels)[2]
        self.assertEqual(item.shape, (4,))
        self.assertEqual(label.shape, ())

    def test_values_match_the_source(self):
        features, labels = tiny_problem(5)
        item, label = m.ArrayDataset(features, labels)[3]
        torch.testing.assert_close(item, features[3])
        self.assertEqual(label.item(), labels[3].item())

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            m.ArrayDataset(torch.zeros(3, 2), torch.zeros(5))

    def test_works_with_a_dataloader(self):
        features, labels = tiny_problem(8)
        loader = torch.utils.data.DataLoader(
            m.ArrayDataset(features, labels), batch_size=4)
        batch_features, batch_labels = next(iter(loader))
        self.assertEqual(batch_features.shape, (4, 4))
        self.assertEqual(batch_labels.shape, (4,))


class TestPadCollate(unittest.TestCase):
    def test_pads_to_the_longest(self):
        padded, _ = m.pad_collate([[1, 2], [1, 2, 3], [1]])
        self.assertEqual(padded.shape, (3, 3))

    def test_returns_true_lengths(self):
        _, lengths = m.pad_collate([[1, 2], [1, 2, 3], [1]])
        torch.testing.assert_close(lengths, torch.tensor([2, 3, 1]))

    def test_padding_value(self):
        padded, _ = m.pad_collate([[1], [1, 2]], pad_value=-1)
        torch.testing.assert_close(padded, torch.tensor([[1, -1], [1, 2]]))

    def test_content_is_left_aligned(self):
        padded, _ = m.pad_collate([[5, 6, 7], [8]])
        torch.testing.assert_close(padded[1], torch.tensor([8, 0, 0]))

    def test_equal_lengths_need_no_padding(self):
        padded, lengths = m.pad_collate([[1, 2], [3, 4]])
        torch.testing.assert_close(padded, torch.tensor([[1, 2], [3, 4]]))
        torch.testing.assert_close(lengths, torch.tensor([2, 2]))

    def test_default_collate_would_fail_here(self):
        """Establishes why a custom collate_fn is needed at all."""
        with self.assertRaises(RuntimeError):
            torch.stack([torch.tensor([1, 2]), torch.tensor([1, 2, 3])])


class TestSeeding(unittest.TestCase):
    def test_same_seed_gives_the_same_torch_draw(self):
        m.seed_everything(123)
        first = torch.randn(4)
        m.seed_everything(123)
        torch.testing.assert_close(torch.randn(4), first)

    def test_same_seed_gives_the_same_numpy_draw(self):
        m.seed_everything(123)
        first = np.random.rand(4)
        m.seed_everything(123)
        np.testing.assert_allclose(np.random.rand(4), first)

    def test_same_seed_gives_the_same_python_draw(self):
        import random
        m.seed_everything(123)
        first = [random.random() for _ in range(4)]
        m.seed_everything(123)
        self.assertEqual([random.random() for _ in range(4)], first)

    def test_different_seeds_differ(self):
        m.seed_everything(1)
        first = torch.randn(8)
        m.seed_everything(2)
        self.assertFalse(torch.allclose(torch.randn(8), first))

    def test_worker_init_gives_workers_different_seeds(self):
        """Without this every worker produces identical augmentations."""
        torch.manual_seed(0)
        m.worker_init_fn(0)
        first = np.random.rand(3)
        torch.manual_seed(0)
        m.worker_init_fn(1)
        self.assertFalse(np.allclose(np.random.rand(3), first))


class TestTrainOneEpoch(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.features, self.labels = tiny_problem(16)
        self.loader = torch.utils.data.DataLoader(
            m.ArrayDataset(self.features, self.labels), batch_size=4)
        self.model = nn.Linear(4, 3)
        self.optimiser = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.loss_fn = nn.CrossEntropyLoss()

    def test_returns_a_finite_loss(self):
        loss = m.train_one_epoch(self.model, self.loader, self.optimiser,
                                 self.loss_fn)
        self.assertTrue(np.isfinite(loss))

    def test_parameters_actually_change(self):
        before = self.model.weight.detach().clone()
        m.train_one_epoch(self.model, self.loader, self.optimiser,
                          self.loss_fn)
        self.assertFalse(torch.allclose(self.model.weight, before))

    def test_repeated_epochs_reduce_the_loss(self):
        losses = [m.train_one_epoch(self.model, self.loader, self.optimiser,
                                    self.loss_fn) for _ in range(30)]
        self.assertLess(losses[-1], losses[0])

    def test_it_zeroes_the_gradients(self):
        """After the epoch, .grad must hold the LAST batch only.

        Without zero_grad it holds the sum over every batch, because .grad
        accumulates. lr=0 keeps the weights still so the two gradients are
        computed at the same point and the comparison is exact.
        """
        model = nn.Linear(4, 3)
        frozen = torch.optim.SGD(model.parameters(), lr=0.0)
        loader = torch.utils.data.DataLoader(
            m.ArrayDataset(self.features[:8], self.labels[:8]), batch_size=4)
        m.train_one_epoch(model, loader, frozen, self.loss_fn)
        after_epoch = model.weight.grad.detach().clone()

        model.zero_grad()
        self.loss_fn(model(self.features[4:8]), self.labels[4:8]).backward()
        last_batch_only = model.weight.grad.detach().clone()

        model.zero_grad()
        self.loss_fn(model(self.features[:4]), self.labels[:4]).backward()
        both_batches = last_batch_only + model.weight.grad

        torch.testing.assert_close(after_epoch, last_batch_only,
                                   atol=1e-5, rtol=1e-4)
        self.assertFalse(torch.allclose(after_epoch, both_batches, atol=1e-5),
                         "gradients accumulated across batches - "
                         "call optimiser.zero_grad()")

    def test_it_leaves_the_model_in_training_mode(self):
        self.model.eval()
        m.train_one_epoch(self.model, self.loader, self.optimiser,
                          self.loss_fn)
        self.assertTrue(self.model.training)

    def test_it_weights_by_sample_not_by_batch(self):
        """A short final batch must not count as much as a full one."""
        features, labels = tiny_problem(5)
        loader = torch.utils.data.DataLoader(
            m.ArrayDataset(features, labels), batch_size=4)
        model = nn.Linear(4, 3)
        loss_fn = nn.CrossEntropyLoss()
        # lr=0 so the model does not move and the arithmetic is checkable.
        reported = m.train_one_epoch(
            model, loader, torch.optim.SGD(model.parameters(), lr=0.0),
            loss_fn)
        with torch.no_grad():
            expected = loss_fn(model(features), labels).item()
        self.assertAlmostEqual(reported, expected, places=5)


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        features, labels = tiny_problem(8)
        self.loader = torch.utils.data.DataLoader(
            m.ArrayDataset(features, labels), batch_size=4)
        self.features, self.labels = features, labels
        self.model = nn.Linear(4, 3)
        self.loss_fn = nn.CrossEntropyLoss()

    def test_matches_a_manual_forward(self):
        with torch.no_grad():
            expected = self.loss_fn(self.model(self.features),
                                    self.labels).item()
        self.assertAlmostEqual(
            m.evaluate(self.model, self.loader, self.loss_fn), expected,
            places=5)

    def test_parameters_are_unchanged(self):
        before = self.model.weight.detach().clone()
        m.evaluate(self.model, self.loader, self.loss_fn)
        torch.testing.assert_close(self.model.weight, before)

    def test_no_gradients_are_produced(self):
        self.model.zero_grad(set_to_none=True)
        m.evaluate(self.model, self.loader, self.loss_fn)
        self.assertIsNone(self.model.weight.grad)

    def test_it_uses_eval_mode(self):
        """Dropout at p=1 outputs zeros in train mode and passes in eval."""
        model = nn.Sequential(nn.Dropout(p=1.0))
        loader = torch.utils.data.DataLoader(
            m.ArrayDataset(torch.ones(4, 2), torch.ones(4, 2)), batch_size=4)
        model.train()
        loss = m.evaluate(model, loader, nn.MSELoss())
        self.assertAlmostEqual(loss, 0.0, places=6)

    def test_it_restores_the_previous_mode(self):
        self.model.train()
        m.evaluate(self.model, self.loader, self.loss_fn)
        self.assertTrue(self.model.training)

    def test_it_leaves_an_eval_model_in_eval(self):
        self.model.eval()
        m.evaluate(self.model, self.loader, self.loss_fn)
        self.assertFalse(self.model.training)


class TestCosineWarmup(unittest.TestCase):
    def test_warmup_ramps_up(self):
        values = [m.cosine_warmup_lr(s, 1.0, 10, 100) for s in range(10)]
        self.assertEqual(values, sorted(values))
        self.assertLess(values[0], values[-1])

    def test_warmup_reaches_the_base_rate(self):
        self.assertAlmostEqual(m.cosine_warmup_lr(9, 1.0, 10, 100), 1.0,
                               places=6)

    def test_it_decays_after_warmup(self):
        values = [m.cosine_warmup_lr(s, 1.0, 10, 100) for s in range(10, 100)]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_it_ends_at_min_lr(self):
        self.assertAlmostEqual(m.cosine_warmup_lr(100, 1.0, 10, 100,
                                                  min_lr=0.01), 0.01,
                               places=6)

    def test_halfway_through_decay_is_the_midpoint(self):
        value = m.cosine_warmup_lr(50, 1.0, 0, 100)
        self.assertAlmostEqual(value, 0.5, places=6)

    def test_no_warmup_starts_at_the_base_rate(self):
        self.assertAlmostEqual(m.cosine_warmup_lr(0, 0.5, 0, 100), 0.5,
                               places=6)

    def test_it_does_not_go_negative_past_the_end(self):
        self.assertGreaterEqual(m.cosine_warmup_lr(500, 1.0, 10, 100), 0.0)


class TestCheckpoints(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = nn.Linear(4, 2)
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=0.01)

    def test_round_trip_restores_the_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ckpt.pt")
            m.save_checkpoint(path, self.model, self.optimiser, epoch=3)
            saved = self.model.weight.detach().clone()

            fresh = nn.Linear(4, 2)
            m.load_checkpoint(path, fresh)
            torch.testing.assert_close(fresh.weight, saved)

    def test_it_stores_the_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ckpt.pt")
            m.save_checkpoint(path, self.model, self.optimiser, epoch=7)
            self.assertEqual(m.load_checkpoint(path, self.model)["epoch"], 7)

    def test_extra_keys_survive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ckpt.pt")
            m.save_checkpoint(path, self.model, self.optimiser, epoch=1,
                              best_metric=0.92)
            payload = m.load_checkpoint(path, self.model)
            self.assertAlmostEqual(payload["best_metric"], 0.92)

    def test_optimiser_state_is_saved_and_restored(self):
        """Adam's moments must survive, or resuming spikes the loss."""
        self.model(torch.randn(4, 4)).sum().backward()
        self.optimiser.step()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ckpt.pt")
            m.save_checkpoint(path, self.model, self.optimiser, epoch=1)

            fresh_model = nn.Linear(4, 2)
            fresh_optimiser = torch.optim.Adam(fresh_model.parameters(),
                                               lr=0.01)
            m.load_checkpoint(path, fresh_model, fresh_optimiser)
            self.assertEqual(len(fresh_optimiser.state_dict()["state"]),
                             len(self.optimiser.state_dict()["state"]))

    def test_loading_without_an_optimiser_is_fine(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ckpt.pt")
            m.save_checkpoint(path, self.model, self.optimiser, epoch=1)
            m.load_checkpoint(path, nn.Linear(4, 2))


class TestOverfitOneBatch(unittest.TestCase):
    def test_the_loss_goes_down(self):
        torch.manual_seed(0)
        features, labels = tiny_problem(8)
        model = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 3))
        history = m.overfit_one_batch(model, features, labels,
                                      nn.CrossEntropyLoss())
        self.assertLess(history[-1], history[0])

    def test_it_reaches_near_zero(self):
        """A working model MUST be able to memorise one batch."""
        torch.manual_seed(0)
        features, labels = tiny_problem(8)
        model = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 3))
        history = m.overfit_one_batch(model, features, labels,
                                      nn.CrossEntropyLoss())
        self.assertLess(history[-1], 0.05)

    def test_history_length_matches_steps(self):
        features, labels = tiny_problem(4)
        history = m.overfit_one_batch(nn.Linear(4, 3), features, labels,
                                      nn.CrossEntropyLoss(), steps=17)
        self.assertEqual(len(history), 17)

    def test_a_detached_model_cannot_learn(self):
        """The diagnostic working as intended: a broken graph stays flat."""
        class Detached(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 3)

            def forward(self, x):
                return self.linear(x).detach() + 0 * self.linear(x).sum() * 0

        torch.manual_seed(0)
        features, labels = tiny_problem(8)
        history = m.overfit_one_batch(Detached(), features, labels,
                                      nn.CrossEntropyLoss(), steps=50)
        self.assertAlmostEqual(history[0], history[-1], places=5)


class TestClassWeights(unittest.TestCase):
    def test_rarer_classes_get_larger_weights(self):
        weights = m.class_weights(torch.tensor([0, 0, 0, 0, 1]), 2)
        self.assertGreater(weights[1].item(), weights[0].item())

    def test_balanced_data_gives_all_ones(self):
        weights = m.class_weights(torch.tensor([0, 0, 1, 1]), 2)
        torch.testing.assert_close(weights, torch.ones(2))

    def test_known_values(self):
        # N=5, C=2, counts=[4, 1] -> [5/8, 5/2]
        weights = m.class_weights(torch.tensor([0, 0, 0, 0, 1]), 2)
        torch.testing.assert_close(weights, torch.tensor([0.625, 2.5]))

    def test_absent_classes_are_zero_not_infinite(self):
        weights = m.class_weights(torch.tensor([0, 0, 1]), 4)
        self.assertTrue(torch.isfinite(weights).all())
        self.assertEqual(weights[2].item(), 0.0)

    def test_length_matches_num_classes(self):
        self.assertEqual(len(m.class_weights(torch.tensor([0, 1]), 7)), 7)

    def test_it_works_in_cross_entropy(self):
        labels = torch.tensor([0, 0, 0, 1])
        weights = m.class_weights(labels, 2)
        loss = nn.CrossEntropyLoss(weight=weights)
        self.assertTrue(np.isfinite(loss(torch.randn(4, 2), labels).item()))


class TestBuildLoader(unittest.TestCase):
    def setUp(self):
        features, labels = tiny_problem(16)
        self.dataset = m.ArrayDataset(features, labels)

    def test_batch_size(self):
        loader = m.build_loader(self.dataset, batch_size=4)
        self.assertEqual(next(iter(loader))[0].shape[0], 4)

    def test_the_same_seed_gives_the_same_order(self):
        first = [b[1] for b in m.build_loader(self.dataset, 4, seed=7)]
        second = [b[1] for b in m.build_loader(self.dataset, 4, seed=7)]
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b)

    def test_different_seeds_give_different_orders(self):
        first = torch.cat([b[1] for b in m.build_loader(self.dataset, 4,
                                                        seed=1)])
        second = torch.cat([b[1] for b in m.build_loader(self.dataset, 4,
                                                         seed=2)])
        self.assertFalse(torch.equal(first, second))

    def test_shuffling_off_preserves_dataset_order(self):
        loader = m.build_loader(self.dataset, 16, shuffle=False)
        torch.testing.assert_close(next(iter(loader))[1],
                                   self.dataset.labels)

    def test_every_sample_appears_exactly_once(self):
        seen = torch.cat([b[1] for b in m.build_loader(self.dataset, 4)])
        self.assertEqual(len(seen), 16)

    def test_extra_kwargs_are_forwarded(self):
        loader = m.build_loader(self.dataset, 5, drop_last=True)
        self.assertEqual(len(list(loader)), 3)   # 16 // 5


if __name__ == "__main__":
    unittest.main(verbosity=2)
