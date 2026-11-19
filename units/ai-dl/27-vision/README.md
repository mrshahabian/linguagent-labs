# 27 — Convolution, detection and video

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/ai-dl/27-vision/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/ai-dl/vision?utm_source=github&utm_medium=labs)

*~50 minutes · 72 tests*

Vision interviews open with arithmetic. *"You have a 224×224 input and this
stack of layers — what is the feature-map size? What is the receptive
field? How many parameters?"* You are expected to answer without a
calculator, because it is a proxy for whether you can reason about a model
you have not seen.

Then come the two implementations everyone is asked for — IoU and NMS — and
the shape questions from whichever sub-field the role is in: ViT
tokenisation, MobileNet-style efficiency, preprocessing, or video.

Work in `exercises.py`. Check your work with `python -m labkit check 27`, or open the notebook (`lab.ipynb`) in Colab.

Each section below ends with **Test it** — the exact command for just that
piece, run from `units/ai-dl/27-vision/`. No need to finish the whole file
before checking your work.

## What you'll build

10 functions in `exercises.py`, in this order. Keep this table open as you
go — each section below tells you which row it's building:

| Function | You'll implement with | Shows up in real code as |
|---|---|---|
| `conv_output_size` | the output-size formula, dilation, raising on a kernel that doesn't fit | Computing a feature-map shape while designing a backbone, before a shape mismatch surfaces three layers later |
| `receptive_field` | the jump-term recurrence over `[(kernel, stride), ...]` | Checking whether a backbone's deepest features can even see an object that size — the argument for dilated convolutions in segmentation |
| `conv2d_via_unfold` | `F.unfold` (im2col) plus one matmul | The GEMM every accelerator actually runs underneath `nn.Conv2d` — the same "reshape into one matmul" idea as FlashAttention |
| `box_iou` | `clamp(min=0)` on the overlap before dividing | The pairwise score behind mAP, and the first call inside every detector's post-processing |
| `nms` | sorting by score, iteratively dropping overlaps | The last stage of every object detector — the box-suppression step after Faster R-CNN, YOLO, or RetinaNet emits raw proposals |
| `patchify` | `F.unfold` with `stride == kernel` | The literal first layer of a Vision Transformer, turning an image into a token sequence |
| `depthwise_separable_params` | the standard-vs-separable parameter count | The MobileNet argument for "how would you make this model run on a phone" |
| `normalise_batch` | reshaping `(C,)` stats to `(1, C, 1, 1)` before broadcasting | Every image preprocessing pipeline — and the standard silent bug in a hand-written one |
| `sample_frames` | uniform temporal sampling with a midpoint offset | Feeding a fixed-length video model from clips of arbitrary length, without throwing away everything after the first second |
| `frame_difference` | temporal subtraction of consecutive frames | The cheap motion baseline behind a two-stream network's optical-flow branch |

---

## 1. The output-size formula

```
out = floor((in + 2·padding − dilation·(kernel−1) − 1) / stride) + 1
```

Memorise it. The corollaries worth having ready:

- **"Same" padding** for an odd kernel at stride 1 is `(kernel−1)//2`.
- **Stride s** divides the size by roughly s.
- **Dilation** enlarges the kernel's *reach* without adding parameters:
  effective kernel `dilation·(kernel−1)+1`. A dilated 3×3 with d=2 covers
  5×5 with 9 weights.

The reference raises `ValueError` when the kernel does not fit rather than
returning a negative size that fails much later inside a matmul:

```python
>>> conv_output_size(7, 3)                          # no padding, no stride
5
>>> conv_output_size(32, 3, stride=1, padding=1)    # "same" padding
32
>>> conv_output_size(224, 3, stride=2, padding=1)   # stride roughly halves it
112
>>> conv_output_size(9, 3, dilation=2)              # effective kernel = 2*(3-1)+1 = 5
5
>>> conv_output_size(3, 5)                          # kernel doesn't fit
ValueError: kernel 5 (effective 5) does not fit in 3 with padding 0
```

**In `exercises.py`:** `conv_output_size` is this section's target — the
formula above, plus the `ValueError` on a kernel that doesn't fit.

**Test it:** `python3 -m unittest test_exercises.TestConvOutputSize -v`

## 2. Receptive field

```
r₀ = 1;   rᵢ = rᵢ₋₁ + (kᵢ − 1) · jumpᵢ₋₁
jumpᵢ = jumpᵢ₋₁ · sᵢ
```

The **jump** — the cumulative stride — is the part people drop. A 3×3 after
two stride-2 layers moves 4 input pixels per step, so it grows the
receptive field by 8, not by 2.

Two facts that fall out:

```python
>>> receptive_field([(3, 1), (3, 1)])          # two stacked 3x3...
5
>>> receptive_field([(3, 2), (3, 2), (3, 1)])  # jump compounds: 1 -> 2 -> 4
15
```

- **Two stacked 3×3 = one 5×5; three = one 7×7.** With fewer parameters
  (2·9 vs 25) and an extra non-linearity. That is the entire VGG argument,
  and it is why 3×3 became the default kernel.
- **If your receptive field is smaller than the object you are detecting,
  no amount of training fixes it** — the network physically cannot see the
  whole thing. This is the argument for dilated convolutions in
  segmentation, and for the deep stages of an FPN.

**In `exercises.py`:** `receptive_field` is this section's target — the
recurrence above, folded left to right over the layer list.

**Test it:** `python3 -m unittest test_exercises.TestReceptiveField -v`

## 3. Convolution *is* a matrix multiply

```python
columns = F.unfold(x, (kh, kw), stride=stride, padding=padding)
out = weight.reshape(out_channels, -1) @ columns
```

`unfold` (im2col) extracts every sliding patch into a column, turning
convolution into one GEMM. **This is how it is actually implemented**:
matrix multiply is the one kernel every accelerator is fast at, so
everything gets reshaped into one.

```python
>>> x = torch.randn(2, 3, 8, 9)
>>> weight = torch.randn(4, 3, 3, 3)
>>> bias = torch.randn(4)
>>> out = conv2d_via_unfold(x, weight, bias, padding=1)
>>> out.shape
torch.Size([2, 4, 8, 9])
>>> torch.allclose(out, F.conv2d(x, weight, bias, padding=1), atol=1e-4)
True
```

The cost is memory — patches overlap, so the column matrix is `kh·kw` times
larger than the input. That trade-off is why implicit-GEMM (computing the
patches on the fly inside the kernel) and Winograd exist, and it is exactly
the same argument as FlashAttention's: fuse the operation so you never
materialise the intermediate.

Being able to say "convolution is a matmul with a data rearrangement" is a
better answer than reciting the sliding-window definition.

**In `exercises.py`:** `conv2d_via_unfold` is this section's target — build
`out_h`/`out_w` with §1's `conv_output_size`, unfold, matmul, reshape, add
the bias. It also has to stay differentiable (the tests check
`.backward()`), which falls out for free as long as you only use
differentiable tensor ops — no `.item()` or `.numpy()` along the way.

**Test it:** `python3 -m unittest test_exercises.TestConv2dViaUnfold -v`

## 4. IoU

```python
overlap = (min(br) - max(tl)).clamp(min=0)
iou = intersection / (area_a + area_b - intersection)
```

`clamp(min=0)` is the whole exercise. Without it, two disjoint boxes give a
negative width **times** a negative height — a *positive* "intersection",
and a plausible-looking IoU for boxes that do not touch. That is a
genuinely hard bug to see in an aggregate metric.

```python
>>> boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0],
...                       [5.0, 5.0, 15.0, 15.0],
...                       [20.0, 20.0, 30.0, 30.0]])
>>> box_iou(boxes[:1], boxes[1:2])   # 25 intersection over 175 union
tensor([[0.1429]])
>>> box_iou(boxes, boxes[:2]).shape  # (N, M): every pair, not just the diagonal
torch.Size([3, 2])
```

Conventions to state before you start writing: `(x1, y1, x2, y2)` vs
`(cx, cy, w, h)`, and whether boxes are inclusive of the far edge (the
`+1` in some older codebases). Ask; do not assume.

**In `exercises.py`:** `box_iou` is this section's target — pairwise IoU
between two sets of boxes, `(N, 4)` and `(M, 4)` in, `(N, M)` out.

**Test it:** `python3 -m unittest test_exercises.TestBoxIou -v`

## 5. NMS

Take the highest-scoring box, drop everything overlapping it above the
threshold, repeat.

```python
order = scores.argsort(descending=True)
while order.numel():
    best = order[0]; keep.append(best)
    ious = box_iou(boxes[best][None], boxes[order[1:]])[0]
    order = order[1:][ious <= threshold]
```

```python
>>> boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0],
...                       [1.0, 1.0, 11.0, 11.0],    # heavily overlaps box 0
...                       [50.0, 50.0, 60.0, 60.0]]) # far away from both
>>> scores = torch.tensor([0.9, 0.8, 0.7])
>>> nms(boxes, scores, iou_threshold=0.5)
tensor([0, 2])
```

Points that come up:

- **Sort by score, not by input order.** The test checks this explicitly.
- **The threshold convention** — is IoU exactly equal to it kept or
  dropped? Say which you chose. This reference keeps ties (`<=`, not `<`).
- **Class-wise NMS.** Real detectors run it per class, or add a large
  per-class offset to the coordinates so boxes of different classes can
  never overlap (`batched_nms`).
- **Soft-NMS** decays scores instead of deleting, which helps with heavily
  overlapping objects — a good "how would you improve this" answer.

**In `exercises.py`:** `nms` is this section's target. It calls §4's
`box_iou` internally, so `box_iou` needs to work first — the tests for
`nms` will fail for the wrong reason (a stubbed `box_iou`) if it isn't.

**Test it:** `python3 -m unittest test_exercises.TestNms -v`

## 6. ViT patches

```python
F.unfold(images, patch, stride=patch).transpose(1, 2)
```

`unfold` with `stride == kernel` is exactly "non-overlapping patches". Cut
the image up, flatten each patch into a token, and you have the input to a
transformer.

```python
>>> image = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
>>> patchify(image, 2)   # four 2x2 patches, row-major, each flattened
tensor([[[ 0.,  1.,  4.,  5.],
         [ 2.,  3.,  6.,  7.],
         [ 8.,  9., 12., 13.],
         [10., 11., 14., 15.]]])
>>> patchify(torch.randn(1, 3, 224, 224), 16).shape   # ViT-Base config
torch.Size([1, 196, 768])
```

224×224 with 16×16 patches → **196 tokens of 768 dims** (`3·16·16`). At
O(L²) attention, that number is why ViT input resolution is such a
constrained choice, and why every high-resolution variant (Swin,
hierarchical ViTs) exists to avoid paying it.

**In `exercises.py`:** `patchify` is this section's target — non-overlapping
patches via `F.unfold`, transposed so patches come before the flattened
pixel dimension, plus a `ValueError` when the image size isn't divisible by
the patch size.

**Test it:** `python3 -m unittest test_exercises.TestPatchify -v`

## 7. Depthwise-separable convolution

```
standard:  out · in · k²
separable: in · k²  (depthwise)  +  in · out  (pointwise 1×1)
```

```python
>>> depthwise_separable_params(64, 128, 3)
(73728, 8768)
>>> standard, separable = depthwise_separable_params(256, 256, 3)
>>> standard / separable
8.694339622641509
```

The saving is roughly `1/out + 1/k²` — about **9×** for a 3×3 with many
channels, matching the ratio above.

Depthwise applies one filter *per input channel* (`groups=in_channels`),
mixing space but not channels; the 1×1 then mixes channels but not space.
Together they approximate a full convolution at a fraction of the cost.

This one arithmetic fact is MobileNet, and it is the standard answer to
*"how would you make this run on a phone?"* — alongside quantisation,
pruning and distillation.

**In `exercises.py`:** `depthwise_separable_params` is this section's
target — return `(standard, separable)` parameter counts, excluding bias.

**Test it:** `python3 -m unittest test_exercises.TestDepthwiseSeparable -v`

## 8. The preprocessing bug

```python
images - mean          # mean is (C,) → broadcasts against WIDTH
```

A `(C,)` tensor aligns with the **last** axis. On a `(B, C, H, W)` batch
that is the width, not the channel:

```python
>>> images = torch.zeros(1, 3, 2, 5)   # (B, C, H, W) = (1, 3, 2, 5)
>>> mean = torch.tensor([1.0, 2.0, 3.0])
>>> images - mean
RuntimeError: The size of tensor a (5) must match the size of tensor b (3)
at non-singleton dimension 3
```

That at least crashes. On a **square** image it does not — it silently
normalises the wrong dimension instead:

```python
>>> square = torch.zeros(1, 3, 4, 3)   # H == C == 3: the shapes coincide
>>> (square - mean).shape              # no error, wrong axis subtracted
torch.Size([1, 3, 4, 3])
```

```python
mean.reshape(1, -1, 1, 1)          # correct
```

```python
>>> images = torch.zeros(1, 3, 2, 5)
>>> images[0, 0], images[0, 1], images[0, 2] = 1.0, 2.0, 3.0
>>> normalise_batch(images, [1.0, 2.0, 3.0], [1.0, 1.0, 1.0]).abs().max()
tensor(0.)
```

The tests here deliberately use a **non-square** image, because a square
one hides it. This is the most common bug in a hand-written preprocessing
pipeline.

**In `exercises.py`:** `normalise_batch` is this section's target —
reshape `mean`/`std` to `(1, C, 1, 1)` before subtracting and dividing,
accepting either a list or a tensor, without mutating `images`.

**Test it:** `python3 -m unittest test_exercises.TestNormaliseBatch -v`

## 9. Video

Two things a video role will ask about early — one about sampling, one
about motion. They are independent functions, tested separately.

### Temporal sampling

Models take a fixed number of frames; real clips have arbitrary lengths.
Sample **uniformly across the whole clip** rather than taking the first N,
which throws away everything after the first second:

```python
>>> sample_frames(100, 8)   # 8 frames spread across a 100-frame clip
tensor([ 6, 18, 31, 43, 56, 68, 81, 93])
>>> sample_frames(3, 8)     # clip shorter than the request -> frames repeat
tensor([0, 0, 0, 1, 1, 2, 2, 2])
```

The `+0.5` midpoint offset centres each sample in its segment instead of
biasing every one towards the start — that is why the first index above is
`6`, not `0`, for a clip split into 8 segments of ~12.5 frames each. Short
clips repeat frames rather than raising — a pipeline that crashes on one
short video is worse than one that pads.

**In `exercises.py`:** `sample_frames` is this subsection's target — indices
of `num_samples` frames spread evenly over `num_frames`, with the midpoint
offset, clamped so the last index never runs past `num_frames - 1`.

**Test it:** `python3 -m unittest test_exercises.TestSampleFrames -v`

### Motion

Frame differencing is the cheapest possible motion feature and a genuinely
useful baseline: a two-stream network's optical-flow branch is doing a more
expensive version of the same thing.

```python
>>> clip = torch.arange(4, dtype=torch.float32).reshape(4, 1, 1, 1)
>>> frame_difference(clip)
tensor([[[[1.]]],

        [[[1.]]],

        [[[1.]]]])
>>> clip5 = torch.arange(5, dtype=torch.float32).reshape(5, 1, 1, 1)
>>> frame_difference(clip5, stride=2)
tensor([[[[2.]]],

        [[[2.]]],

        [[[2.]]]])
```

Worth saying that a large part of what a video model learns from "motion"
is available from frame differences at almost no cost.

Beyond these two functions, the vocabulary to have: 3D convolutions
(expensive), (2+1)D factorised convolutions (spatial then temporal — the
same separability argument as MobileNet in §7), two-stream networks, and
video transformers with factorised space/time attention (again, to avoid
O((T·H·W)²)).

**In `exercises.py`:** `frame_difference` is this subsection's target —
`clip[stride:] - clip[:-stride]`, raising on a non-positive stride or a
clip too short for it.

**Test it:** `python3 -m unittest test_exercises.TestFrameDifference -v`

---

## The exercises

| Function | What it teaches |
|---|---|
| `conv_output_size` | The formula, dilation, failing loudly |
| `receptive_field` | The jump term; the VGG argument |
| `conv2d_via_unfold` | Convolution as GEMM; the memory trade-off |
| `box_iou` | `clamp(min=0)`; the false-positive overlap |
| `nms` | Score order; threshold convention |
| `patchify` | ViT tokenisation; where 196 comes from |
| `depthwise_separable_params` | The MobileNet arithmetic |
| `normalise_batch` | The `(1, C, 1, 1)` reshape |
| `sample_frames` | Uniform temporal sampling; short clips |
| `frame_difference` | Motion as a baseline feature |

## Check your understanding

1. **Why does stacking two 3×3 convolutions cover more receptive field
   growth per parameter than one 5×5 convolution?**
   <details><summary>Answer</summary>Two 3×3 layers give a 5×5
   effective receptive field (the "jump" compounds across layers) using
   2×(3×3) = 18 weights per channel pair, versus 25 for a single 5×5 —
   fewer parameters for the same receptive field, plus an extra
   nonlinearity in between.</details>

2. **What's the actual trade-off in implementing convolution via
   `unfold` (im2col) and a matrix multiply?**
   <details><summary>Answer</summary>It turns convolution into a single
   large, highly-optimized GEMM call — fast — at the cost of
   materializing every overlapping patch explicitly, which uses
   noticeably more memory than a direct sliding-window
   implementation.</details>

3. **Why does `box_iou` need `clamp(min=0)` on the intersection
   dimensions?**
   <details><summary>Answer</summary>When two boxes don't actually
   overlap, the naive intersection width/height computation can go
   negative — clamping to 0 turns "no overlap" into an intersection
   area of 0 instead of a nonsensical negative area.</details>

4. **In NMS, why does the order boxes are processed in matter?**
   <details><summary>Answer</summary>NMS greedily keeps the
   highest-scoring box first and suppresses lower-scoring boxes that
   overlap it too much — processing in score order (highest first) is
   what makes "keep the best, suppress worse duplicates" the actual
   behavior, rather than an arbitrary one.</details>

5. **A ViT patchifies a 224×224 image into 196 patches. Where does 196
   come from?**
   <details><summary>Answer</summary>With 16×16 patches, 224 / 16 = 14
   patches per side, and 14 × 14 = 196 patches total covering the whole
   image.</details>

6. **Why does `normalise_batch` reshape the per-channel mean/std to
   `(1, C, 1, 1)` before applying them to a `(N, C, H, W)` batch?**
   <details><summary>Answer</summary>That shape broadcasts correctly
   against `(N, C, H, W)` — matching on the channel dimension while
   the size-1 dimensions broadcast across batch, height and width, so
   every pixel in a channel is normalized by that channel's own
   statistics.</details>

## Interview questions this module answers

- *What size is the feature map after this stack?*
- *What is the receptive field, and why does it matter?*
- *How is convolution actually implemented on a GPU?*
- *Implement IoU. Now implement NMS.*
- *How does a ViT turn an image into tokens?*
- *How would you make this model run on a phone?*
- *How do you feed a variable-length video to a fixed-size model?*
- *Why do two 3×3 convolutions beat one 5×5?*
