"""Tests for 37 - classical image processing."""

import importlib
import os
import unittest

import torch
import torch.nn.functional as F

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestGaussianKernel(unittest.TestCase):
    def test_sums_to_one(self):
        kernel = m.gaussian_kernel(5, sigma=1.0)
        self.assertAlmostEqual(kernel.sum().item(), 1.0, places=5)

    def test_peak_is_at_the_center(self):
        kernel = m.gaussian_kernel(5, sigma=1.0)
        self.assertEqual(kernel[2, 2].item(), kernel.max().item())

    def test_symmetric(self):
        kernel = m.gaussian_kernel(5, sigma=1.5)
        self.assertTrue(torch.allclose(kernel, kernel.flip(0)))
        self.assertTrue(torch.allclose(kernel, kernel.flip(1)))

    def test_larger_sigma_is_flatter(self):
        tight = m.gaussian_kernel(5, sigma=0.5)
        wide = m.gaussian_kernel(5, sigma=5.0)
        self.assertGreater(tight[2, 2].item(), wide[2, 2].item())


class TestApplyKernel(unittest.TestCase):
    def test_identity_kernel_preserves_the_image(self):
        identity = torch.zeros(3, 3)
        identity[1, 1] = 1.0
        torch.manual_seed(0)
        image = torch.rand(8, 8)
        result = m.apply_kernel(image, identity)
        self.assertTrue(torch.allclose(result, image, atol=1e-5))

    def test_output_shape_matches_input(self):
        kernel = m.gaussian_kernel(5, 1.0)
        image = torch.rand(10, 12)
        result = m.apply_kernel(image, kernel)
        self.assertEqual(result.shape, image.shape)

    def test_box_blur_averages_a_uniform_region(self):
        box = torch.ones(3, 3) / 9
        image = torch.ones(10, 10) * 50.0
        result = m.apply_kernel(image, box)
        self.assertTrue(torch.allclose(result[1:-1, 1:-1], torch.full((8, 8), 50.0), atol=1e-4))


class TestSobelEdges(unittest.TestCase):
    def test_flat_image_has_no_interior_edges(self):
        flat = torch.ones(8, 8) * 100.0
        edges = m.sobel_edges(flat)
        self.assertTrue(torch.allclose(edges[1:-1, 1:-1], torch.zeros(6, 6), atol=1e-4))

    def test_step_edge_produces_a_strong_response(self):
        step = torch.zeros(8, 8)
        step[:, 4:] = 255.0
        edges = m.sobel_edges(step)
        # response is strongest in the columns straddling the step, away from the border
        self.assertGreater(edges[3:5, 3].max().item(), 200)

    def test_output_shape_matches_input(self):
        image = torch.rand(6, 9) * 255
        edges = m.sobel_edges(image)
        self.assertEqual(edges.shape, image.shape)

    def test_uniform_region_far_from_any_edge_is_near_zero(self):
        image = torch.zeros(10, 10)
        image[:, 8:] = 255.0
        edges = m.sobel_edges(image)
        self.assertTrue(torch.allclose(edges[1:-1, 1:4], torch.zeros(8, 3), atol=1e-4))


class TestHistogramEqualization(unittest.TestCase):
    def test_full_range_image_maps_extremes_to_extremes(self):
        image = torch.tensor([[0, 0], [255, 255]], dtype=torch.long)
        result = m.histogram_equalization(image)
        self.assertEqual(result[0, 0].item(), 0)
        self.assertEqual(result[1, 0].item(), 255)

    def test_output_shape_matches_input(self):
        image = torch.randint(0, 256, (5, 7))
        result = m.histogram_equalization(image)
        self.assertEqual(result.shape, image.shape)

    def test_low_contrast_image_gets_spread_out(self):
        # every value crammed into [100, 110] - a narrow, low-contrast band
        image = torch.randint(100, 111, (20, 20))
        result = m.histogram_equalization(image)
        self.assertGreater(int(result.max()) - int(result.min()), 110 - 100)

    def test_preserves_relative_order(self):
        image = torch.tensor([[10, 50], [100, 200]], dtype=torch.long)
        result = m.histogram_equalization(image)
        flat_in = image.flatten().tolist()
        flat_out = result.flatten().tolist()
        pairs_in_order = sorted(range(4), key=lambda i: flat_in[i])
        pairs_out_order = sorted(range(4), key=lambda i: flat_out[i])
        self.assertEqual(pairs_in_order, pairs_out_order)


class TestBilinearResize(unittest.TestCase):
    def test_matches_torch_interpolate_upsampling(self):
        torch.manual_seed(0)
        image = torch.rand(6, 8) * 255
        mine = m.bilinear_resize(image, 12, 16)
        reference = F.interpolate(
            image.unsqueeze(0).unsqueeze(0), size=(12, 16),
            mode="bilinear", align_corners=False
        ).squeeze()
        self.assertTrue(torch.allclose(mine, reference, atol=1e-3))

    def test_matches_torch_interpolate_downsampling(self):
        torch.manual_seed(1)
        image = torch.rand(10, 10) * 255
        mine = m.bilinear_resize(image, 4, 5)
        reference = F.interpolate(
            image.unsqueeze(0).unsqueeze(0), size=(4, 5),
            mode="bilinear", align_corners=False
        ).squeeze()
        self.assertTrue(torch.allclose(mine, reference, atol=1e-3))

    def test_output_shape(self):
        image = torch.rand(5, 5)
        result = m.bilinear_resize(image, 3, 7)
        self.assertEqual(result.shape, (3, 7))

    def test_same_size_is_approximately_identity(self):
        torch.manual_seed(2)
        image = torch.rand(6, 6) * 255
        result = m.bilinear_resize(image, 6, 6)
        self.assertTrue(torch.allclose(result, image, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
