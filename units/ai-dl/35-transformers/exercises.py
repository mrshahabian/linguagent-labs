"""35 - assembling transformers end to end

Implement each function so the tests in test_exercises.py pass. Read
README.md first; every exercise below is covered there.

    ../run 35            test your work
    ../run 35 --check    watch the reference solutions pass

The docstrings are the specification. Standard library only.
"""

import math
import torch
import torch.nn.functional as F
from torch import nn


def sinusoidal_positional_encoding(seq_len, embed_dim):
    """The original Transformer paper's fixed (non-learned) position encoding.

    Every even dimension gets sin(pos / 10000^(2i/d)), every odd dimension
    gets cos of the same argument. Two properties make this a reasonable
    choice instead of a learned embedding table: it needs no parameters, and
    it extrapolates to sequence lengths never seen in training (a learned
    absolute-position table has no row for position 10001 if it only ever
    saw sequences up to length 10000).

    Contrast with module 26 (attention): RoPE rotates Q/K INSIDE attention
    based on relative offset; this instead produces a fixed per-position
    vector ADDED to the token embedding before any attention happens. Both
    solve the same problem - attention itself has no notion of order - but
    RoPE encodes relative position implicitly in the dot product, while this
    encodes absolute position explicitly in the input.
    """
    raise NotImplementedError("your code here")


class TransformerBlock(nn.Module):
    """Pre-norm decoder block: x + self_attn(norm(x)), then x + mlp(norm(x)).

    Identical shape to module 26's TransformerBlock - repeated here because
    each module in this repo is self-contained, not because it's a new
    idea. GPTModel below stacks N of these.
    """

    def __init__(self, embed_dim, num_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
        )

    def forward(self, x, causal_mask):
        raise NotImplementedError("your code here")


class GPTModel(nn.Module):
    """A minimal decoder-only language model: embed, stack blocks, un-embed.

    Three pieces the individual attention primitives in module 26 don't
    show on their own: a LEARNED positional embedding added to the token
    embedding, N transformer blocks actually stacked in an nn.ModuleList
    (not just one), and WEIGHT TYING between the input embedding and the
    output projection - self.lm_head.weight IS self.token_embedding.weight,
    the same tensor, not a copy. Tying works because both have shape
    (vocab_size, embed_dim): the embedding looks a row up by index, the
    untied head would learn a separate (vocab_size, embed_dim) matrix to
    project back to vocab logits - tying assumes the same matrix is a
    reasonable answer for both directions, and roughly halves the
    parameters spent on the vocabulary.
    """

    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, max_seq_len, mlp_ratio=4):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids):
        raise NotImplementedError("your code here")


class CrossAttentionDecoderBlock(nn.Module):
    """A seq2seq decoder block: self-attention, THEN cross-attention, THEN an MLP.

    Three residual sublayers instead of GPTModel's two - the middle one is
    new. Self-attention (causal, over the target sequence built so far)
    lets the decoder look at its own previous outputs; cross-attention
    (queries from the target sequence, keys/values from the ENCODER's
    output) is how information about the source sequence enters the
    decoder at all - exactly the same mechanic as module 31's VLM
    cross-attention, applied here between a source and target sequence in
    the same modality (e.g. translation) instead of between text and image
    patches.
    """

    def __init__(self, embed_dim, num_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
        )

    def forward(self, x, encoder_output, causal_mask):
        raise NotImplementedError("your code here")


def count_transformer_parameters(vocab_size, embed_dim, num_layers, max_seq_len, mlp_ratio=4):
    """Closed-form parameter count matching GPTModel exactly, for the same sizes.

    Per block: attention is 4 projections of size (embed_dim, embed_dim)
    plus biases (Q, K, V, output) = 4 * (embed_dim^2 + embed_dim) - this
    total does NOT depend on num_heads, since splitting into heads only
    reshapes the same weight matrices, it doesn't add parameters. The MLP
    is two linears expanding by mlp_ratio, plus their biases; two
    LayerNorms contribute 2 * embed_dim each (weight and bias).

    The token embedding and the tied output head together cost ONE
    (vocab_size * embed_dim) matrix, not two - that's the point of tying.
    The position embedding is untied and separate, costing
    (max_seq_len * embed_dim) on its own.

    This exists to answer "how big is this model" without instantiating
    it - the same estimate a serving engineer makes before ever loading
    weights, to check it fits in memory.
    """
    raise NotImplementedError("your code here")


@torch.no_grad()
def generate_greedy(model, input_ids, max_new_tokens):
    """Autoregressively extend input_ids by max_new_tokens, greedy at each step.

    No KV cache here (module 26 covers that trade-off on its own) - this
    is the simplest correct version: run the whole sequence through the
    model every step, take the argmax over the last position's logits,
    append it, repeat. Simplicity is the point of this exercise; a serving
    system would add the KV cache from module 26 on top of exactly this
    loop, not replace it.
    """
    raise NotImplementedError("your code here")
