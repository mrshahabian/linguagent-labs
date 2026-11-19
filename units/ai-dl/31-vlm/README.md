# 31 — Vision-language models

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/ai-dl/31-vlm/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/ai-dl/vlm?utm_source=github&utm_medium=labs)

*~55 minutes · 22 tests*

A VLM is two things bolted together: a way to make image and text
embeddings comparable (CLIP's contribution), and a way to feed image
information into a language model that only knows how to process token
sequences (LLaVA's and Flamingo's contribution, in two different shapes).
This module builds both halves from the primitives already in this
track — `scaled_dot_product_attention` from module 26, `patchify` from
module 27 — and one is enough to explain a large fraction of the VLM
questions in a modern LLM/multimodal interview.

Work in `exercises.py`. Check your work with `python -m labkit check 31`, or open the notebook (`lab.ipynb`) in Colab.

Each section below ends with **Test it** — the exact command for just that
piece, run from `units/ai-dl/31-vlm/`.

## What you'll build

| Name | You'll implement | Shows up in real code as |
|---|---|---|
| `l2_normalize` | `l2_normalize` | Turning a dot product into a cosine similarity |
| `clip_contrastive_loss` | `clip_contrastive_loss` | CLIP's training objective — symmetric InfoNCE over a batch |
| `zero_shot_classify` | `zero_shot_classify` | Classification with no classifier head — CLIP's headline trick |
| `insert_image_tokens` | `insert_image_tokens` | LLaVA-style fusion: image patches spliced into a text sequence |
| `patch_grid_positions` | `patch_grid_positions` | 2D position handling for a ViT-style patch grid |
| `cross_attend` | `cross_attend` | Flamingo/BLIP-style fusion: text queries attending over image K/V |

---

## 1. Comparable embeddings: `l2_normalize`

```python
x / x.norm(dim=dim, keepdim=True).clamp_min(eps)
```

Every exercise below assumes unit-norm embeddings, because that's what
turns a plain dot product into a **cosine similarity** — a measure of
direction alone. Without normalising first, a dot product conflates
"points the same way" with "has large magnitude," which is not what
"this image matches this caption" is supposed to mean. `clamp_min(eps)`
exists purely to avoid a division by exactly zero for a degenerate
all-zero vector; it does not meaningfully change any non-degenerate
result.

**In `exercises.py`:** `l2_normalize`.

**Test it:** `python3 -m unittest test_exercises.TestL2Normalize -v`

## 2. CLIP's training objective: symmetric InfoNCE

```python
image_embeds = l2_normalize(image_embeds)
text_embeds = l2_normalize(text_embeds)
logits_per_image = logit_scale * image_embeds @ text_embeds.T
logits_per_text = logits_per_image.T
labels = torch.arange(image_embeds.shape[0])
loss = (F.cross_entropy(logits_per_image, labels)
        + F.cross_entropy(logits_per_text, labels)) / 2
```

`image_embeds[i]` and `text_embeds[i]` are assumed to be a matched pair —
image `i` and its real caption. Everything else in the batch is treated
as a negative, for free — this is the "in-batch negatives" trick, and
it's why CLIP-style training wants a **large batch size**: a batch of 32
gives you 31 negatives per example; a batch of 8192 gives you 8191.

The similarity matrix's diagonal should score highest. Cross-entropy
against `labels = arange(N)` is exactly "row `i` should pick column `i`
as the answer" — computed once treating each **image** as the query
(rows) and once treating each **text** as the query (columns), because
nothing in the setup says one direction is more important than the
other. `logit_scale` is a learned temperature (real CLIP stores its
`log` and exponentiates before use, clamped to prevent runaway growth) —
sharper temperatures make the loss more sensitive to small alignment
errors once the model is mostly correct.

```python
>>> matched = clip_contrastive_loss(image_embeds, image_embeds + noise*0.01, logit_scale=14.0)
>>> shuffled = clip_contrastive_loss(image_embeds, shuffled_text, logit_scale=14.0)
>>> matched < shuffled
True
```

**In `exercises.py`:** `clip_contrastive_loss`.

**Test it:** `python3 -m unittest test_exercises.TestClipContrastiveLoss -v`

## 3. Zero-shot classification: no classifier head, ever

```python
similarities = l2_normalize(class_text_embeds) @ l2_normalize(image_embed)
predicted_index = int(similarities.argmax())
```

This is what a "zero-shot" claim concretely means: "class 3" is just
whichever class-name text embedding — the embedding of a string like
`"a photo of a dog"` — is most cosine-similar to the image embedding.
Nothing was ever trained to output "dog" as a class index. Adding a new
class means embedding one new string and appending it to
`class_text_embeds`; it does not mean retraining anything, which is the
entire practical appeal over a fixed-head classifier.

```python
>>> zero_shot_classify(dog_photo_embedding, [embed("a cat"), embed("a dog"), embed("a car")])
(1, tensor([0.18, 0.31, 0.05]))
```

**In `exercises.py`:** `zero_shot_classify`.

**Test it:** `python3 -m unittest test_exercises.TestZeroShotClassify -v`

## 4. LLaVA-style fusion: splice, don't merge

```python
before = text_embeds[:image_token_position]
after = text_embeds[image_token_position + 1:]
return torch.cat([before, image_embeds, after], dim=0)
```

The simplest way to give a language model access to an image: run the
image through a vision encoder and a small projection to get patch
embeddings in the same dimensionality as the text embeddings, then
**replace** a single placeholder token (conventionally `<image>`) in the
tokenized prompt with all of those patch embeddings, expanding the
sequence. After this splice, an ordinary decoder-only transformer
processes the whole thing as one sequence — it has no architectural
awareness that some of these vectors came from a vision encoder rather
than a text embedding table. All the actual multimodal reasoning is
delegated entirely to self-attention being able to look across the
image-token span the same way it looks across any other span. This is
LLaVA's approach, and it is why swapping in a stronger vision encoder or
language model is comparatively easy — the interface between them is
just "a sequence of vectors."

```python
>>> insert_image_tokens(text_embeds, image_patch_embeds, image_token_position=2)
# the sequence grows by (num_image_patches - 1)
```

**In `exercises.py`:** `insert_image_tokens`.

**Test it:** `python3 -m unittest test_exercises.TestInsertImageTokens -v`

## 5. Position handling: patches have two coordinates, tokens have one

```python
rows = torch.arange(n).repeat_interleave(n)
cols = torch.arange(n).repeat(n)
```

A text token's only natural coordinate is its index in a 1D sequence. An
image patch naturally has **two** — its row and column in the original
image — and a flattened 1D index alone loses that: patch 7 and patch 8
might be true neighbours, or might sit on opposite edges of the image,
entirely depending on the grid's width, which the index itself doesn't
encode. `patch_grid_positions` returns the `(row, col)` pair for every
patch in raster order — the same order `patchify` (module 27) produces
— which is what a 2D positional encoding scheme (2D RoPE, or a learned
row/column embedding table) needs as input. After `insert_image_tokens`
splices patches into a text sequence, the text tokens still use their
ordinary 1D sequence position; only the image span switches to this 2D
scheme.

```python
>>> patch_grid_positions(3)
(tensor([0, 0, 0, 1, 1, 1, 2, 2, 2]), tensor([0, 1, 2, 0, 1, 2, 0, 1, 2]))
```

**In `exercises.py`:** `patch_grid_positions`.

**Test it:** `python3 -m unittest test_exercises.TestPatchGridPositions -v`

## 6. Flamingo/BLIP-style fusion: cross-attention instead of concatenation

```python
scores = query @ key.T / math.sqrt(d)
weights = F.softmax(scores, dim=-1)
return weights @ value
```

This is the **identical formula** to `scaled_dot_product_attention`
from module 26 — nothing about the mechanism changes when `query` comes
from the text side and `key`/`value` come from the image side instead of
all three coming from the same sequence. "Cross-attention" names *where
the inputs come from*, not a different algorithm. This is the second
major VLM fusion pattern, used by Flamingo (gated cross-attention layers
interleaved through a frozen language model) and BLIP-2's Q-Former: text
queries attend directly over image key/value pairs, rather than image
patches being spliced into the token sequence the way `insert_image_tokens`
does it. The practical difference: cross-attention doesn't grow the
text sequence length at all, and `query` and `key`/`value` are free to
have completely different lengths — cross-attention has no requirement
that they match, unlike ordinary self-attention where Q, K and V all
come from (and are the same length as) one sequence.

```python
>>> cross_attend(text_queries, image_patch_keys, image_patch_values).shape
torch.Size([num_text_queries, embed_dim])
```

**In `exercises.py`:** `cross_attend`.

**Test it:** `python3 -m unittest test_exercises.TestCrossAttend -v`

---

## The exercises

| Function | What it teaches |
|---|---|
| `l2_normalize` | Cosine similarity requires unit-norm vectors first |
| `clip_contrastive_loss` | Symmetric InfoNCE; in-batch negatives |
| `zero_shot_classify` | Classification via text-embedding similarity, no head |
| `insert_image_tokens` | LLaVA-style splice fusion |
| `patch_grid_positions` | 2D coordinates for a patch grid |
| `cross_attend` | Flamingo/BLIP-style cross-modal attention |

Every function in this table has a stub in `exercises.py` and test
coverage in `test_exercises.py` — a clean 1:1 mapping.

## Check your understanding

1. **Why does CLIP's loss need to be computed in both directions —
   image-as-query and text-as-query — rather than just one?**
   <details><summary>Answer</summary>Nothing about the setup privileges
   one modality over the other; "this image matches this caption" and
   "this caption matches this image" are two separate retrieval
   directions that a single training signal doesn't automatically
   optimize for equally. Averaging both cross-entropy terms trains the
   embedding space to work for both image-to-text and text-to-image
   retrieval.</details>

2. **Why does CLIP-style contrastive training benefit from a large
   batch size specifically, more than most other training objectives
   do?**
   <details><summary>Answer</summary>Every other example in the batch
   is used as a free negative — a batch of size N gives N-1 negatives
   per example with no extra sampling step. A bigger batch means more
   negatives per example, which makes the contrastive signal harder
   and more informative; this is a more direct, structural dependence
   on batch size than most training objectives have.</details>

3. **What does "zero-shot" concretely mean in `zero_shot_classify` — what
   was never trained?**
   <details><summary>Answer</summary>No classification head mapping
   embeddings to a fixed set of class indices was ever trained. The
   "classifier" is entirely improvised at inference time by embedding
   each class's name as text and picking whichever is most
   cosine-similar to the image embedding — adding a class means adding
   a string, not retraining anything.</details>

4. **What's the structural difference between LLaVA's fusion approach
   (`insert_image_tokens`) and Flamingo's (`cross_attend`)?**
   <details><summary>Answer</summary>LLaVA splices image patch
   embeddings directly into the text token sequence, so the sequence
   length grows and an ordinary self-attention transformer processes
   everything as one sequence. Flamingo's cross-attention keeps text and
   image as two separate sequences and lets text queries attend over
   image keys/values without ever concatenating them — the text
   sequence length doesn't change, and image and text tokens are never
   mixed into one sequence at all.</details>

5. **Why can't a flattened 1D sequence position alone capture an image
   patch's spatial relationship to its neighbours the way it does for a
   text token?**
   <details><summary>Answer</summary>A text token genuinely only has one
   natural coordinate — its position in the sequence. An image patch has
   two — row and column in the source image — and flattening a 2D grid
   into a 1D index loses one of those dimensions: two patches with
   adjacent 1D indices might be true horizontal neighbours, or might sit
   on opposite ends of a row depending on the grid's width, which the
   flat index alone doesn't distinguish.</details>

## Interview questions this module answers

- *How does CLIP learn to align image and text embeddings with no
  labelled classification data?*
- *What does "zero-shot" mean here, precisely?*
- *Design a VLM architecture: how does the image get into the language
  model?*
- *What's the difference between how LLaVA and Flamingo fuse image and
  text information?*
- *Why doesn't a flattened patch index alone work as a position for a
  vision transformer?*
