# 37 — Classical image processing

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/ai-dl/37-image-processing/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/ai-dl/image-processing?utm_source=github&utm_medium=labs)

*~50 minutes · 19 tests*

Before a learned convolution ever sees a pixel, decades of hand-designed
filters already solved blurring, edge detection, contrast enhancement and
resizing — and those techniques still show up constantly, as
preprocessing, as augmentation, and as the vocabulary an interviewer uses
to ask about them. This module builds the classical toolkit: Gaussian
blur, the Sobel edge detector, histogram equalization, and bilinear
resizing — checked against PyTorch's own `F.conv2d` and `F.interpolate`
where an equivalent exists.

Work in `exercises.py`. Check your work with `python -m labkit check 37`, or open the notebook (`lab.ipynb`) in Colab.

Each section below ends with **Test it** — the exact command for just that
piece, run from `units/ai-dl/37-image-processing/`.

## What you'll build

| Name | You'll implement | Shows up in real code as |
|---|---|---|
| `gaussian_kernel` | the function | The blur kernel behind noise reduction and scale-space methods |
| `apply_kernel` | the function | "Same"-padding convolution — the mechanics under every classical filter |
| `sobel_edges` | the function | The pre-deep-learning edge detector, still used as a preprocessing step |
| `histogram_equalization` | the function | Contrast enhancement — the standard fix for a washed-out or dark image |
| `bilinear_resize` | the function | The interpolation behind every image resize that isn't nearest-neighbour |

---

## 1. A Gaussian kernel, and applying it

```python
kernel = torch.exp(-(grid_x ** 2 + grid_y ** 2) / (2 * sigma ** 2))
kernel = kernel / kernel.sum()
```

A 2D Gaussian kernel, normalised so its values sum to exactly 1 —
normalising is what keeps a blurred image's overall brightness
unchanged; an unnormalised kernel would systematically darken or
brighten the result depending on whether its sum is below or above 1.
(Real implementations exploit that a 2D Gaussian is *separable* — the
outer product of two 1D Gaussians — and convolve rows then columns
instead of building the full 2D kernel, purely for speed; this exercise
builds the 2D kernel directly since the shape is the point.)

```python
def apply_kernel(image, kernel):
    padding = kernel.shape[-1] // 2
    return F.conv2d(image[None, None], kernel[None, None], padding=padding).squeeze()
```

Padding by `kernel_size // 2` before convolving is what makes the output
the same size as the input — a "same" convolution, using the identical
padding arithmetic as `conv_output_size` back in module 27, applied here
so a filtered image doesn't shrink every time you apply one.

```python
>>> kernel = gaussian_kernel(5, sigma=1.0)
>>> kernel.sum()
tensor(1.0000)
>>> apply_kernel(image, kernel).shape == image.shape
True
```

**In `exercises.py`:** `gaussian_kernel`, `apply_kernel`.

**Test it:** `python3 -m unittest test_exercises.TestGaussianKernel test_exercises.TestApplyKernel -v`

## 2. Sobel: edges as a gradient magnitude

```python
gx_kernel = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
gy_kernel = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
magnitude = sqrt(gx ** 2 + gy ** 2)
```

Two fixed 3x3 kernels approximate the horizontal and vertical intensity
gradient at every pixel. Combining them as a gradient **magnitude**
(rather than looking at `Gx` or `Gy` alone) responds to an edge
regardless of its orientation — a horizontal edge, a vertical edge, and
a diagonal edge all produce a large magnitude, just from different
combinations of `Gx` and `Gy`. This is the pre-deep-learning answer to
"where are the edges in this image": a fixed, hand-designed filter,
where a CNN's early layers instead *learn* whatever edge-like filters
turn out to be useful for the task.

**A real gotcha worth knowing:** `apply_kernel`'s zero-padding invents a
false intensity drop right at the image border (a real pixel sitting
next to a padded zero), so even a perfectly flat image shows a nonzero
"edge" exactly on its border — genuinely zero only in the interior.
Every edge detector either accepts this border artifact or pads
differently (replicating the edge pixel, mirroring the image) to avoid
it.

```python
>>> flat = torch.ones(8, 8) * 100
>>> sobel_edges(flat)[1:-1, 1:-1]   # interior: exactly zero
tensor([[0., 0., ...
```

**In `exercises.py`:** `sobel_edges`.

**Test it:** `python3 -m unittest test_exercises.TestSobelEdges -v`

## 3. Histogram equalization: stretching a narrow band of intensities

```python
cdf = torch.cumsum(histogram, dim=0)
cdf_min = cdf[cdf > 0][0]
cdf_normalized = (cdf - cdf_min) / (total - cdf_min)
lookup_table = (cdf_normalized * 255).clamp(min=0).round().long()
```

The cumulative histogram (CDF) at intensity `v` is "what fraction of all
pixels are at or below `v`." Remapping each pixel through its own CDF
stretches a low-contrast image — one whose intensities are bunched into
a narrow band — across the *full* output range, because the CDF rises
steeply through that narrow band: the same amount of "probability mass"
now spans more output values.

**Subtracting `cdf_min`** — the CDF at the image's own darkest present
value — is what forces the darkest pixel to map to exactly 0 and the
brightest to exactly 255, matching the standard formula used by
OpenCV's `equalizeHist`. Skip that subtraction and the darkest pixel
maps to some value *above* zero, since its own count is already folded
into `cdf(v)` at `v` itself.

```python
>>> histogram_equalization(torch.tensor([[0, 0], [255, 255]]))
tensor([[  0,   0],
        [255, 255]])
```

**In `exercises.py`:** `histogram_equalization`.

**Test it:** `python3 -m unittest test_exercises.TestHistogramEqualization -v`

## 4. Bilinear resize: blending four neighbours, not picking one

```python
src_y = (out_y + 0.5) * scale_y - 0.5
y0, y1 = int(src_y), min(int(src_y) + 1, in_height - 1)
wy = src_y - y0
# blend the four (y0,x0), (y0,x1), (y1,x0), (y1,x1) neighbours by wx, wy
```

Nearest-neighbour resizing just picks whichever source pixel is
closest, which produces visible blockiness scaling up. Bilinear instead
maps each output pixel back to a **fractional** source coordinate and
blends the four neighbouring source pixels, weighted by how close that
fractional coordinate is to each one. The `+ 0.5` / `- 0.5` shift is
what aligns pixel *centers* rather than pixel corners between the two
grids — the same convention `F.interpolate(..., align_corners=False)`
uses, which is why this exercise is checked against it directly rather
than against a hand-derived expected array.

```python
>>> bilinear_resize(image, out_height=12, out_width=16)
# matches F.interpolate(image, size=(12, 16), mode="bilinear", align_corners=False)
```

**In `exercises.py`:** `bilinear_resize`.

**Test it:** `python3 -m unittest test_exercises.TestBilinearResize -v`

---

## The exercises

| Function | What it teaches |
|---|---|
| `gaussian_kernel` | Normalisation preserves brightness; the shape of a blur |
| `apply_kernel` | "Same"-padding convolution mechanics |
| `sobel_edges` | Gradient magnitude as orientation-independent edge strength |
| `histogram_equalization` | CDF-based contrast stretching, and why `cdf_min` matters |
| `bilinear_resize` | Fractional-coordinate interpolation, pixel-center alignment |

Every function in this table has a stub in `exercises.py` and test
coverage in `test_exercises.py` — a clean 1:1 mapping.

## Check your understanding

1. **Why does a Gaussian kernel need to be normalised to sum to 1
   before you convolve an image with it?**
   <details><summary>Answer</summary>Convolution's output at each pixel
   is a weighted sum of neighbouring pixel values using the kernel's
   entries as weights. If those weights don't sum to 1, the output is
   systematically scaled up or down relative to the input's brightness —
   normalising to 1 is what keeps a blurred image's overall brightness
   matching the original.</details>

2. **Why does Sobel combine `Gx` and `Gy` as `sqrt(Gx² + Gy²)` instead of
   just using `Gx` (or `Gy`) alone?**
   <details><summary>Answer</summary><code>Gx</code> alone only responds
   strongly to vertical edges (horizontal intensity change), and
   <code>Gy</code> alone only to horizontal edges. Combining them as a
   magnitude responds to an edge of ANY orientation — a diagonal edge
   produces moderate values in both <code>Gx</code> and <code>Gy</code>
   that combine into a large magnitude, even though neither alone would
   flag it strongly.</details>

3. **Why does a perfectly flat, constant-value image still show a
   nonzero Sobel response at its border?**
   <details><summary>Answer</summary>Convolving with padding adds zeros
   just outside the image, and a border pixel's neighbourhood includes
   those zeros right next to real (nonzero) pixel values — that contrast
   is a real intensity jump from the filter's point of view, even though
   it's an artifact of the padding, not a real edge in the
   image.</details>

4. **What does subtracting `cdf_min` in histogram equalization actually
   fix?**
   <details><summary>Answer</summary>Without it, the image's darkest
   present intensity would map to some value above 0, because its own
   pixel count is already included in the CDF at that intensity.
   Subtracting the CDF's value at the darkest present intensity resets
   that floor to exactly 0, so the full output range is genuinely used
   from 0 to 255, matching the standard formula (and OpenCV's
   `equalizeHist`).</details>

5. **What visual artifact does nearest-neighbour resizing produce that
   bilinear resizing avoids, and why?**
   <details><summary>Answer</summary>Blockiness — nearest-neighbour
   assigns each output pixel the value of whichever single source pixel
   is closest, so scaling up produces visible flat blocks where several
   output pixels copy the same source value. Bilinear blends four
   neighbouring source pixels by how close the mapped fractional
   coordinate is to each, producing a smooth gradient between source
   pixels instead of a hard step.</details>

## Interview questions this module answers

- *Why do we normalise a blur kernel before applying it?*
- *How does Sobel edge detection work, and why combine two kernels
  instead of one?*
- *An image looks washed out or too dark — how would you fix its
  contrast programmatically?*
- *What's the difference between nearest-neighbour and bilinear image
  resizing?*
