"""37 - classical image processing

Implement each function so the tests in test_exercises.py pass. Read
README.md first; every exercise below is covered there.

    ../run 37            test your work
    ../run 37 --check    watch the reference solutions pass

The docstrings are the specification. Standard library only.
"""

import torch
import torch.nn.functional as F


def gaussian_kernel(size, sigma):
    """A (size, size) 2D Gaussian kernel, normalised to sum to 1.

    Separable in principle (a 2D Gaussian is the outer product of two 1D
    Gaussians, which is why real implementations convolve rows then
    columns instead of doing the full 2D kernel) - built directly here as
    a 2D grid for clarity, since the point of this exercise is the shape
    of the kernel, not the factorisation that makes it fast.

    Normalising to sum 1 is what keeps a blurred image's overall
    brightness unchanged - an unnormalised kernel would systematically
    darken or brighten the image depending on whether its sum is below
    or above 1.
    """
    raise NotImplementedError("your code here")


def apply_kernel(image, kernel):
    """Convolve a single-channel image with a 2D kernel, same output size.

    Padding by kernel_size // 2 on each side before F.conv2d's default
    "valid" convolution is what makes output size equal input size (a
    "same" convolution) - the same padding arithmetic as module 27's
    conv_output_size, applied here to make classical filters preserve
    image dimensions the way every real image-filtering function does.
    """
    raise NotImplementedError("your code here")


def sobel_edges(image):
    """Gradient magnitude via the Sobel operator - the classic edge detector.

    Two 3x3 kernels approximate the horizontal and vertical intensity
    gradient (Gx, Gy) at each pixel; combining them as
    sqrt(Gx^2 + Gy^2) gives the gradient MAGNITUDE, which is large at an
    edge (a sharp intensity change in either direction) regardless of the
    edge's orientation. This is the pre-deep-learning answer to "where
    are the edges in this image" - a fixed, hand-designed filter rather
    than a learned one.

    Note apply_kernel's zero-padding invents a fake intensity drop at the
    image border (real pixels next to padded zeros), so even a perfectly
    flat image shows a false "edge" exactly on its border - genuinely
    zero only in the interior. Every real edge detector has to either
    accept this border artifact or pad some other way (replicate the
    edge pixel, mirror the image) to avoid it.
    """
    raise NotImplementedError("your code here")


def histogram_equalization(image):
    """Redistribute a grayscale image's intensities to use the full 0-255 range.

    Built from the image's own cumulative histogram (CDF): a pixel's new
    value is proportional to what fraction of all pixels are at or below
    its current value. An image whose intensities are bunched into a
    narrow band (low contrast) gets that band stretched across the full
    range, because the CDF rises steeply there - the same amount of
    "probability mass" now spans more output values.

    Subtracting cdf_min (the CDF at the image's own darkest present
    value) before normalising is what forces the darkest pixel in the
    image to map to exactly 0, and the brightest to exactly 255 - the
    standard formula (matching OpenCV's equalizeHist), not just "roughly
    stretched." Without that subtraction the darkest pixel would map to
    some value above 0 instead, since its own count is already folded
    into cdf(v) at v itself.

    image: integer tensor, values in [0, 255].
    """
    raise NotImplementedError("your code here")


def bilinear_resize(image, out_height, out_width):
    """Resize a single-channel image using bilinear interpolation.

    Nearest-neighbour resizing just picks the closest source pixel,
    which produces visible blockiness when scaling up. Bilinear instead
    maps each output pixel back to a (fractional) source coordinate and
    blends the FOUR neighbouring source pixels, weighted by how close
    the fractional coordinate is to each one - the same shape of
    interpolation `F.interpolate(mode="bilinear")` does, built here from
    the coordinate mapping up.
    """
    raise NotImplementedError("your code here")
