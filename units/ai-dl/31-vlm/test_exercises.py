"""Tests for 31 - vision-language models."""

import importlib
import os
import unittest

import torch
import torch.nn.functional as F

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestL2Normalize(unittest.TestCase):
    def test_result_has_unit_norm(self):
        torch.manual_seed(0)
        x = torch.randn(5, 8) * 10
        result = m.l2_normalize(x)
        self.assertTrue(torch.allclose(result.norm(dim=-1), torch.ones(5), atol=1e-5))

    def test_preserves_direction(self):
        x = torch.tensor([[3.0, 4.0]])
        result = m.l2_normalize(x)
        self.assertTrue(torch.allclose(result, torch.tensor([[0.6, 0.8]])))

    def test_does_not_divide_by_zero(self):
        x = torch.zeros(1, 4)
        result = m.l2_normalize(x)
        self.assertTrue(torch.isfinite(result).all())


class TestClipContrastiveLoss(unittest.TestCase):
    def test_matched_pairs_score_lower_than_shuffled(self):
        torch.manual_seed(1)
        image_embeds = torch.randn(8, 16)
        text_embeds = image_embeds + torch.randn(8, 16) * 0.01
        matched_loss = m.clip_contrastive_loss(image_embeds, text_embeds, logit_scale=14.0)

        shuffled_text = text_embeds[torch.randperm(8)]
        random_loss = m.clip_contrastive_loss(image_embeds, shuffled_text, logit_scale=14.0)

        self.assertLess(matched_loss.item(), random_loss.item())

    def test_perfect_alignment_gives_near_zero_loss(self):
        torch.manual_seed(2)
        embeds = F.normalize(torch.randn(4, 8), dim=-1)
        loss = m.clip_contrastive_loss(embeds, embeds, logit_scale=50.0)
        self.assertLess(loss.item(), 0.01)

    def test_loss_is_scalar(self):
        embeds = torch.randn(3, 4)
        loss = m.clip_contrastive_loss(embeds, embeds, logit_scale=10.0)
        self.assertEqual(loss.dim(), 0)

    def test_higher_logit_scale_sharpens_a_near_correct_loss(self):
        torch.manual_seed(3)
        image_embeds = torch.randn(6, 16)
        text_embeds = image_embeds + torch.randn(6, 16) * 0.05
        low_scale = m.clip_contrastive_loss(image_embeds, text_embeds, logit_scale=1.0)
        high_scale = m.clip_contrastive_loss(image_embeds, text_embeds, logit_scale=50.0)
        # with mostly-correct alignment, a sharper temperature drives loss down further
        self.assertLess(high_scale.item(), low_scale.item())


class TestZeroShotClassify(unittest.TestCase):
    def test_picks_the_matching_class(self):
        image_embed = torch.tensor([1.0, 0.0, 0.0])
        class_embeds = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        index, _ = m.zero_shot_classify(image_embed, class_embeds)
        self.assertEqual(index, 0)

    def test_picks_the_best_of_several_close_options(self):
        image_embed = torch.tensor([0.9, 0.1, 0.0])
        class_embeds = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        index, _ = m.zero_shot_classify(image_embed, class_embeds)
        self.assertEqual(index, 0)

    def test_similarities_are_bounded_cosine_range(self):
        image_embed = torch.randn(8)
        class_embeds = torch.randn(5, 8)
        _, similarities = m.zero_shot_classify(image_embed, class_embeds)
        self.assertTrue((similarities >= -1.0001).all())
        self.assertTrue((similarities <= 1.0001).all())

    def test_similarities_length_matches_number_of_classes(self):
        image_embed = torch.randn(4)
        class_embeds = torch.randn(7, 4)
        _, similarities = m.zero_shot_classify(image_embed, class_embeds)
        self.assertEqual(similarities.shape[0], 7)


class TestInsertImageTokens(unittest.TestCase):
    def test_replaces_placeholder_with_image_tokens(self):
        text_embeds = torch.tensor([[1., 1.], [2., 2.], [0., 0.], [3., 3.]])
        image_embeds = torch.tensor([[9., 9.], [8., 8.]])
        result = m.insert_image_tokens(text_embeds, image_embeds, image_token_position=2)
        expected = torch.tensor([[1., 1.], [2., 2.], [9., 9.], [8., 8.], [3., 3.]])
        self.assertTrue(torch.equal(result, expected))

    def test_sequence_length_grows_by_image_tokens_minus_one(self):
        text_embeds = torch.zeros(5, 4)
        image_embeds = torch.zeros(9, 4)
        result = m.insert_image_tokens(text_embeds, image_embeds, image_token_position=2)
        self.assertEqual(result.shape[0], 5 - 1 + 9)

    def test_placeholder_at_start(self):
        text_embeds = torch.tensor([[0., 0.], [1., 1.]])
        image_embeds = torch.tensor([[5., 5.]])
        result = m.insert_image_tokens(text_embeds, image_embeds, image_token_position=0)
        expected = torch.tensor([[5., 5.], [1., 1.]])
        self.assertTrue(torch.equal(result, expected))

    def test_placeholder_at_end(self):
        text_embeds = torch.tensor([[1., 1.], [0., 0.]])
        image_embeds = torch.tensor([[5., 5.]])
        result = m.insert_image_tokens(text_embeds, image_embeds, image_token_position=1)
        expected = torch.tensor([[1., 1.], [5., 5.]])
        self.assertTrue(torch.equal(result, expected))


class TestPatchGridPositions(unittest.TestCase):
    def test_correct_length(self):
        rows, cols = m.patch_grid_positions(4)
        self.assertEqual(len(rows), 16)
        self.assertEqual(len(cols), 16)

    def test_raster_order_3x3(self):
        rows, cols = m.patch_grid_positions(3)
        self.assertEqual(rows.tolist(), [0, 0, 0, 1, 1, 1, 2, 2, 2])
        self.assertEqual(cols.tolist(), [0, 1, 2, 0, 1, 2, 0, 1, 2])

    def test_max_row_and_col_are_n_minus_one(self):
        rows, cols = m.patch_grid_positions(5)
        self.assertEqual(rows.max().item(), 4)
        self.assertEqual(cols.max().item(), 4)


class TestCrossAttend(unittest.TestCase):
    def test_matches_manual_scaled_dot_product(self):
        torch.manual_seed(4)
        query = torch.randn(3, 8)
        key = torch.randn(6, 8)
        value = torch.randn(6, 8)
        result = m.cross_attend(query, key, value)

        scores = query @ key.T / (8 ** 0.5)
        weights = F.softmax(scores, dim=-1)
        expected = weights @ value
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))

    def test_output_shape_follows_query_length_and_value_dim(self):
        query = torch.randn(4, 16)
        key = torch.randn(10, 16)
        value = torch.randn(10, 16)
        result = m.cross_attend(query, key, value)
        self.assertEqual(result.shape, (4, 16))

    def test_query_and_kv_lengths_may_differ(self):
        # cross-attention with 2 queries attending over 20 keys/values
        query = torch.randn(2, 4)
        key = torch.randn(20, 4)
        value = torch.randn(20, 4)
        result = m.cross_attend(query, key, value)
        self.assertEqual(result.shape, (2, 4))

    def test_single_key_collapses_to_that_value(self):
        query = torch.randn(3, 4)
        key = torch.randn(1, 4)
        value = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        result = m.cross_attend(query, key, value)
        for row in result:
            self.assertTrue(torch.allclose(row, value[0]))


if __name__ == "__main__":
    unittest.main()
