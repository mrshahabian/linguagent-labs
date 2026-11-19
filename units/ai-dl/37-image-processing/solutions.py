"""Reference answers for 37 - classical image processing."""

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
    half = size // 2
    coords = torch.arange(size).float() - half
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(grid_x ** 2 + grid_y ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def apply_kernel(image, kernel):
    """Convolve a single-channel image with a 2D kernel, same output size.

    Padding by kernel_size // 2 on each side before F.conv2d's default
    "valid" convolution is what makes output size equal input size (a
    "same" convolution) - the same padding arithmetic as module 27's
    conv_output_size, applied here to make classical filters preserve
    image dimensions the way every real image-filtering function does.
    """
    kernel_size = kernel.shape[-1]
    padding = kernel_size // 2
    image_batched = image.unsqueeze(0).unsqueeze(0)
    kernel_batched = kernel.unsqueeze(0).unsqueeze(0)
    result = F.conv2d(image_batched, kernel_batched, padding=padding)
    return result.squeeze(0).squeeze(0)


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
    gx_kernel = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
    gy_kernel = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
    gx = apply_kernel(image, gx_kernel)
    gy = apply_kernel(image, gy_kernel)
    return torch.sqrt(gx ** 2 + gy ** 2)


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
    histogram = torch.bincount(image.flatten(), minlength=256).float()
    cdf = torch.cumsum(histogram, dim=0)
    cdf_min = cdf[cdf > 0][0]
    total = image.numel()
    cdf_normalized = (cdf - cdf_min) / (total - cdf_min)
    lookup_table = (cdf_normalized * 255).clamp(min=0).round().long()
    return lookup_table[image.flatten()].reshape(image.shape)


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
    in_height, in_width = image.shape
    image = image.float()

    scale_y = in_height / out_height
    scale_x = in_width / out_width

    output = torch.zeros(out_height, out_width)
    for out_y in range(out_height):
        src_y = (out_y + 0.5) * scale_y - 0.5
        src_y = max(0.0, min(src_y, in_height - 1))
        y0 = int(src_y)
        y1 = min(y0 + 1, in_height - 1)
        wy = src_y - y0

        for out_x in range(out_width):
            src_x = (out_x + 0.5) * scale_x - 0.5
            src_x = max(0.0, min(src_x, in_width - 1))
            x0 = int(src_x)
            x1 = min(x0 + 1, in_width - 1)
            wx = src_x - x0

            top = image[y0, x0] * (1 - wx) + image[y0, x1] * wx
            bottom = image[y1, x0] * (1 - wx) + image[y1, x1] * wx
            output[out_y, out_x] = top * (1 - wy) + bottom * wy

    return output
