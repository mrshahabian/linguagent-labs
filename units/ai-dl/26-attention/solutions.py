"""Reference answers for 26 - attention and transformer blocks."""

import math

import torch
from torch import nn


def scaled_dot_product_attention(q, k, v, mask=None):
    """Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V.

    q, k, v: (..., L, D) -> (..., L, D), plus the (..., L, S) weights.

    Why sqrt(d_k): q.k is a sum of d_k products of unit-variance terms, so
    its variance grows like d_k. Without the scale the logits are large,
    softmax saturates towards one-hot, and its gradient vanishes - the
    model stops learning at exactly the depth where you need it to.

    `mask` is True where attention is ALLOWED. Excluded logits go to -inf
    BEFORE the softmax, never zeroed after it.
    """
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights


def causal_mask(length, device=None):
    """(L, L) boolean mask, True where a query may attend to a key.

    Lower triangular INCLUDING the diagonal: position i sees 0..i. Dropping
    the diagonal (`diagonal=-1`) stops a token attending to itself and is
    the classic off-by-one here - training still works, slightly worse, so
    nothing ever crashes to tell you.
    """
    return torch.ones(length, length, dtype=torch.bool,
                      device=device).tril()


def padding_mask(lengths, max_length=None):
    """(B, L) boolean mask, True at real positions and False at padding.

    Built by comparing a row of positions against a column of lengths -
    the same broadcasting trick as one-hot. Without this, padding tokens
    contribute to every attention sum and the model learns from them.
    """
    max_length = int(max_length or lengths.max())
    positions = torch.arange(max_length, device=lengths.device)
    return positions[None, :] < lengths[:, None]


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention.

    The whole idea: instead of one attention over D dimensions, run H
    attentions over D/H each. Each head can specialise (syntax, position,
    coreference) and they are concatenated back.

    The shape dance is the part interviewers watch for:
        (B, L, D) -> (B, L, H, D/H) -> transpose -> (B, H, L, D/H)
    You must `transpose(1, 2)`, not `view(B, H, L, d)`. A reshape would
    interleave the heads with the sequence and mix positions together -
    silently, since the shapes are identical either way.
    """

    def __init__(self, embed_dim, num_heads, bias=True):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim {embed_dim} not divisible by num_heads {num_heads}")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _split(self, x):
        batch, length, _ = x.shape
        return x.view(batch, length, self.num_heads,
                      self.head_dim).transpose(1, 2)

    def forward(self, x, mask=None):
        batch, length, _ = x.shape
        q = self._split(self.q_proj(x))
        k = self._split(self.k_proj(x))
        v = self._split(self.v_proj(x))
        if mask is not None and mask.dim() == 2:
            mask = mask[None, None, :, :]   # broadcast over batch and heads
        out, weights = scaled_dot_product_attention(q, k, v, mask)
        out = out.transpose(1, 2).reshape(batch, length, self.embed_dim)
        return self.out_proj(out), weights


class KVCache:
    """Append-only cache of past keys and values for autoregressive decoding.

    Why it exists: generating token N re-attends to all N-1 previous
    tokens. Recomputing their keys and values every step makes generation
    O(N^2) in compute; caching them makes each step O(N) and is the single
    biggest win in LLM inference.

    The cost is memory: 2 * layers * heads * head_dim * seq_len * batch *
    bytes. That is why serving an LLM is a memory problem before it is a
    compute problem, and why paged attention exists.
    """

    def __init__(self):
        self.keys = None
        self.values = None

    def append(self, k, v):
        """Add this step's k/v and return the full history."""
        if self.keys is None:
            self.keys, self.values = k, v
        else:
            self.keys = torch.cat([self.keys, k], dim=-2)
            self.values = torch.cat([self.values, v], dim=-2)
        return self.keys, self.values

    def __len__(self):
        return 0 if self.keys is None else self.keys.shape[-2]

    def reset(self):
        self.keys = self.values = None


def rope_frequencies(head_dim, max_length, base=10000.0, device=None):
    """Precompute (cos, sin) tables for rotary position embeddings.

    RoPE encodes position by ROTATING each pair of channels by an angle
    proportional to the position. Because a rotation is orthogonal, the dot
    product between two rotated vectors depends only on their RELATIVE
    offset - which is what attention actually needs, and why RoPE
    extrapolates to lengths it never saw in training.

    Used by Llama, Mistral, Qwen and most current open models.
    """
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")
    inverse_freq = 1.0 / (base ** (
        torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(max_length, device=device).float()
    angles = torch.outer(positions, inverse_freq)     # (L, head_dim/2)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin, offset=0):
    """Rotate x by the precomputed angles. x: (..., L, head_dim).

    `offset` is what makes cached decoding correct: at step N the single
    new token is at position N, not position 0. Forgetting it is a real
    and very confusing bug - the model generates fluent text that slowly
    loses track of where it is.
    """
    length = x.shape[-2]
    cos = cos[offset:offset + length]
    sin = sin[offset:offset + length]
    first, second = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack([first * cos - second * sin,
                           first * sin + second * cos], dim=-1)
    return rotated.flatten(-2)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: x + attn(norm(x)), then x + mlp(norm(x)).

    PRE-norm, not post-norm. The original paper put the norm after the
    residual; pre-norm keeps a clean identity path from input to output, so
    gradients reach the early layers without a warm-up schedule and deep
    stacks train stably. Every modern LLM uses it.

    The MLP expands by 4x by convention - most of the parameters in a
    transformer live here, not in attention.
    """

    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        attended, _ = self.attention(self.norm1(x), mask)
        x = x + attended
        return x + self.mlp(self.norm2(x))


def attention_flops(seq_len, embed_dim, num_heads=1):
    """Multiply-accumulates in one self-attention forward pass.

    Two matmuls dominate: QK^T is L*L*D and (weights @ V) is L*L*D, so the
    total is 2 * L^2 * D. Splitting into heads does NOT change it - each
    head is L*L*(D/H) and there are H of them.

    The point is the L^2: doubling the context quadruples attention cost.
    That single fact motivates FlashAttention, sliding-window attention,
    and every long-context paper of the last few years.
    """
    return 2 * seq_len * seq_len * embed_dim
