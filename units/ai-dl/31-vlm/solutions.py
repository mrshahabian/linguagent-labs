"""Reference answers for 31 - vision-language models."""

import math

import torch
import torch.nn.functional as F


def l2_normalize(x, dim=-1, eps=1e-8):
    """Normalise vectors along dim to unit length, without dividing by zero.

    Every embedding-comparison exercise below assumes unit-norm vectors,
    because that is what turns a dot product into a cosine similarity -
    a plain dot product on un-normalised embeddings conflates "aligned
    direction" with "large magnitude", which is not what CLIP-style
    retrieval means by "similar".
    """
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def clip_contrastive_loss(image_embeds, text_embeds, logit_scale):
    """The CLIP training objective: symmetric InfoNCE over a batch.

    image_embeds[i] and text_embeds[i] are assumed to be a matched pair
    (the i-th image and its caption) - everything else in the batch is
    treated as a negative. The similarity matrix's diagonal should score
    highest; the loss is ordinary cross-entropy against that diagonal,
    computed once treating each image as the query (rows) and once
    treating each text as the query (columns), then averaged. Only the
    "in-batch negatives" trick makes this cheap: no separate negative
    sampling is needed, just a bigger batch.

    logit_scale is a learned temperature (CLIP itself stores its log and
    exponentiates it before use, clamped to prevent the logits from
    blowing up) - passed here already exponentiated for simplicity.
    """
    image_embeds = l2_normalize(image_embeds)
    text_embeds = l2_normalize(text_embeds)
    logits_per_image = logit_scale * image_embeds @ text_embeds.T
    logits_per_text = logits_per_image.T
    labels = torch.arange(image_embeds.shape[0], device=image_embeds.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2


def zero_shot_classify(image_embed, class_text_embeds):
    """Classify one image embedding against a set of class-text embeddings.

    This is CLIP's headline trick: no classification head was ever
    trained. "Class 3" is just whichever class-name text embedding
    (e.g. the embedding of "a photo of a dog") is most cosine-similar to
    the image embedding. Adding a new class means embedding a new string,
    not retraining anything.

    Returns (predicted_index, similarities) - the similarities are
    plain cosine similarity, not passed through a temperature, since
    nothing here is being optimised.
    """
    image_embed = l2_normalize(image_embed, dim=-1)
    class_text_embeds = l2_normalize(class_text_embeds, dim=-1)
    similarities = class_text_embeds @ image_embed
    predicted_index = int(similarities.argmax())
    return predicted_index, similarities


def insert_image_tokens(text_embeds, image_embeds, image_token_position):
    """Splice image patch embeddings into a text embedding sequence.

    LLaVA-style multimodal fusion: a single placeholder position in the
    text sequence (conventionally an <image> token) is REPLACED by every
    image patch embedding, expanding the sequence length by
    (num_image_patches - 1). After this splice, a plain decoder-only
    transformer processes the whole thing as one sequence - it has no
    idea some of these "tokens" came from a vision encoder rather than
    an embedding table. All the multimodal reasoning happens because
    attention can look across the image-token span exactly like it looks
    across any other tokens.
    """
    before = text_embeds[:image_token_position]
    after = text_embeds[image_token_position + 1:]
    return torch.cat([before, image_embeds, after], dim=0)


def patch_grid_positions(num_patches_per_side):
    """2D (row, col) grid coordinates for a square image-patch grid.

    Text tokens have one natural coordinate: their position in a 1D
    sequence. Image patches naturally have TWO - row and column in the
    original image - and a vision transformer's positional encoding
    should reflect that a patch is close to its neighbours in both
    directions, which a single flattened 1D index alone does not
    capture (patch 7 and patch 8 might be adjacent, or on opposite
    edges of the image, depending only on the grid width). Returns two
    1D tensors of length num_patches_per_side**2, in row-major
    (raster) order matching how patchify() flattens patches.
    """
    rows = torch.arange(num_patches_per_side).repeat_interleave(num_patches_per_side)
    cols = torch.arange(num_patches_per_side).repeat(num_patches_per_side)
    return rows, cols


def cross_attend(query, key, value):
    """Cross-attention: queries from one modality, keys/values from another.

    The formula is EXACTLY scaled dot-product attention - softmax(QK^T /
    sqrt(d))V - nothing about the mechanism itself changes when Q comes
    from text tokens and K, V come from image patch embeddings instead
    of from the same sequence as Q. "Cross-attention" names where the
    inputs come from, not a different algorithm; this is the fusion
    style behind Flamingo and BLIP's Q-Former, in contrast to
    insert_image_tokens's approach of just concatenating everything into
    one sequence and letting ordinary self-attention do the mixing.

    query: (num_queries, d). key, value: (num_kv, d). num_queries and
    num_kv may differ - cross-attention does not require equal lengths
    the way self-attention naturally has them equal.
    """
    d = query.shape[-1]
    scores = query @ key.T / math.sqrt(d)
    weights = F.softmax(scores, dim=-1)
    return weights @ value
