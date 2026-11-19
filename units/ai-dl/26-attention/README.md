# 26 — Attention and transformer blocks

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/ai-dl/26-attention/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/ai-dl/attention?utm_source=github&utm_medium=labs)

*~60 minutes · 54 tests*

*"Implement attention from scratch"* is the single most-asked question in
LLM and VLM interviews. Not because it is hard — it is four lines — but
because it is a precise test of whether you can hold shapes in your head,
whether you know why the scale is there, and whether you understand what a
KV cache is actually caching.

Everything here is written out by hand and checked against PyTorch's own
fused implementation, so the exercise is pinned to real behaviour rather
than to my arithmetic.

Work in `exercises.py`. Check your work with `python -m labkit check 26`, or open the notebook (`lab.ipynb`) in Colab.

Each section below ends with **Test it** — the exact command for just that
piece, run from `units/ai-dl/26-attention/`. No need to finish the whole
file before checking your work.

## What you'll build

9 classes/functions in `exercises.py`, in this order. Keep this table open
as you go — each section below tells you which row it's building:

| Function / class | You'll implement | Shows up in real code as |
|---|---|---|
| `scaled_dot_product_attention` | the four lines: scores, mask, softmax, weighted sum | The core primitive behind every transformer layer — what `F.scaled_dot_product_attention` and FlashAttention implement as a fused kernel |
| `causal_mask` | one `tril()` call | Enforcing autoregressive generation — every GPT-style decoder, at both training and inference time |
| `padding_mask` | a broadcast comparison | Batching variable-length sequences without letting padding tokens leak into the softmax — any production training run over real, unequal-length text |
| `MultiHeadAttention` | `_split`, `forward` | The actual attention layer inside every transformer block — GPT, BERT, ViT, and everything descended from them |
| `KVCache` | `append`, `__len__`, `reset` | What makes streaming, token-by-token LLM generation (the way ChatGPT-style APIs respond) fast instead of quadratic |
| `rope_frequencies` | precomputing the `(cos, sin)` tables | The position tables Llama, Mistral and Qwen precompute once and reuse for every forward pass |
| `apply_rope` | rotating the query/key channels | Injecting position at both training and (offset) cached-decode time in those same models |
| `TransformerBlock` | `forward` | The repeated unit stacked N times to build a full transformer — GPT, ViT, and nearly everything else in this space |
| `attention_flops` | the L² formula | Capacity planning — estimating serving cost when sizing a model's context window |

---

## 1. The four lines

```python
scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
scores = scores.masked_fill(~mask, finfo.min)
weights = torch.softmax(scores, dim=-1)
output = weights @ v
```

Shapes: `q, k, v` are `(..., L, D)`; `scores` is `(..., L, S)`; the output
is back to `(..., L, D)`.

The intuition worth being able to give in one sentence: **each query asks
every key how relevant it is, and takes a weighted average of the
corresponding values.**

```python
>>> q = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
>>> k = torch.tensor([[[4.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]])
>>> v = torch.eye(2)[None]
>>> out, weights = scaled_dot_product_attention(q, k, v)
>>> weights
tensor([[[0.8808, 0.1192]]])
>>> out               # weighted average of v's two rows, by those weights
tensor([[[0.8808, 0.1192]]])
```

### Why `sqrt(d_k)`

`q·k` is a sum of `d_k` products of roughly unit-variance terms, so its
variance grows like `d_k`. Without the scale, the logits are large, softmax
saturates towards one-hot, and **its gradient vanishes** — the model stops
learning exactly where you need depth.

That gradient argument is the answer. "It normalises the variance" is only
half of it; the interviewer is listening for what the saturation does to
training. The same `q`, `k` from above, with and without the scale, makes
the saturation visible directly:

```python
>>> _, w_scaled = scaled_dot_product_attention(q, k, v)
>>> w_scaled
tensor([[[0.8808, 0.1192]]])
>>> w_unscaled = torch.softmax(q @ k.transpose(-2, -1), dim=-1)  # no / sqrt(d_k)
>>> w_unscaled
tensor([[[0.9820, 0.0180]]])
```

Dropping the scale pushes the same logits from `0.88/0.12` to `0.98/0.02` —
closer to one-hot, and closer to the flat part of softmax's gradient.

**In `exercises.py`:** `scaled_dot_product_attention` is this section's (and
this subsection's) target — the `sqrt(d_k)` division is one line inside it,
not a separate method.

**Test it:** `python3 -m unittest test_exercises.TestScaledDotProductAttention -v`

This class also covers masking `scaled_dot_product_attention` itself (the
`test_causal_mask*`/`test_masked_*` cases) — that's still this function's
material, since masking is one of its four lines. Building the masks
themselves (`causal_mask`, `padding_mask`) is §2, and their own tests live
in separate classes.

## 2. Masks are boolean and applied before the softmax

**Causal mask** — `torch.ones(L, L).tril()`. Lower triangular *including*
the diagonal, so position i sees 0..i. Dropping the diagonal is the classic
off-by-one: training still works, slightly worse, and nothing ever crashes
to tell you.

**Padding mask** — `positions[None, :] < lengths[:, None]`, the same
broadcasting trick as one-hot. Without it, padding tokens contribute to
every attention sum.

Both combine by `&`. Both go in **before** the exponential, as
`finfo(dtype).min` rather than literal `-inf` (module 01 explains why).

```python
>>> causal_mask(4)
tensor([[ True, False, False, False],
        [ True,  True, False, False],
        [ True,  True,  True, False],
        [ True,  True,  True,  True]])
>>> padding_mask(torch.tensor([2, 4]), 4)
tensor([[ True,  True, False, False],
        [ True,  True,  True,  True]])
```

**In `exercises.py`:** `causal_mask` and `padding_mask`.

**Test it:** `python3 -m unittest test_exercises.TestCausalMask test_exercises.TestPaddingMask -v`

`TestPaddingMask.test_it_composes_with_attention` calls
`scaled_dot_product_attention` too, so it needs §1 working as well as
`padding_mask` — everything else in these two classes tests the mask
functions in isolation.

## 3. Multi-head: the shape dance

Instead of one attention over D dims, run H attentions over D/H each. Each
head can specialise — induction, positional, syntactic — and they are
concatenated back.

```python
(B, L, D) → view(B, L, H, D/H) → transpose(1, 2) → (B, H, L, D/H)
```

**You must `transpose(1, 2)`, not `view(B, H, L, d)`.** Both produce a
tensor of identical shape. Only one is correct. The reshape would
interleave the head axis with the sequence axis, mixing positions
together — silently, with no error, producing a model that trains to a
mediocre loss and never works properly.

If you remember one thing from this module, make it that.

Coming back out: `transpose(1, 2).reshape(B, L, D)` — `reshape`, not
`view`, because the transpose left it non-contiguous.

Parameters: four square projections, `4 × (D² + D)`. Note the head count
does **not** appear — splitting into heads is free.

```python
>>> mha = MultiHeadAttention(embed_dim=8, num_heads=2)
>>> x = torch.randn(1, 3, 8)
>>> out, weights = mha(x)
>>> out.shape
torch.Size([1, 3, 8])
>>> weights.shape          # (batch, heads, L, S)
torch.Size([1, 2, 3, 3])
>>> sum(p.numel() for p in mha.parameters())
288
>>> 4 * (8 * 8 + 8)        # 4 * (D^2 + D); head count doesn't appear
288
```

**In `exercises.py`:** `MultiHeadAttention._split` and
`MultiHeadAttention.forward`. Both build directly on §1's
`scaled_dot_product_attention`, which `forward` calls per head.

**Test it:** `python3 -m unittest test_exercises.TestMultiHeadAttention -v`

## 4. The KV cache

Generating token N re-attends to all N−1 previous tokens. Recomputing their
keys and values every step makes generation O(N²) in compute; caching them
makes each step O(N).

```python
self.keys = torch.cat([self.keys, k], dim=-2)
```

```python
>>> cache = KVCache()
>>> k1, v1 = torch.randn(1, 1, 1, 2), torch.randn(1, 1, 1, 2)
>>> keys, values = cache.append(k1, v1)
>>> keys.shape, len(cache)
(torch.Size([1, 1, 1, 2]), 1)
>>> keys, values = cache.append(torch.randn(1, 1, 1, 2), torch.randn(1, 1, 1, 2))
>>> keys.shape, len(cache)     # grew along the sequence axis
(torch.Size([1, 1, 2, 2]), 2)
>>> cache.reset()
>>> len(cache)
0
```

This is the single biggest win in LLM inference, and its cost is memory:

```
2 × layers × heads × head_dim × seq_len × batch × bytes
```

For a 7B model at 4k context and batch 32 that is tens of gigabytes — often
**more than the weights**. That one calculation is why serving an LLM is a
memory problem before it is a compute problem, and it is what motivates:

- **PagedAttention** (vLLM) — allocate the cache in fixed-size blocks so it
  does not fragment.
- **Multi-query / grouped-query attention** — share K and V across heads,
  cutting the cache by the head count.
- **Quantising the cache** to int8 or fp8.

The test in this module decodes one token at a time and asserts the result
**equals the full masked forward**. That is the only check that actually
catches an off-by-one in cached decoding, and it is worth writing whenever
you implement one.

**In `exercises.py`:** `KVCache.append`, `KVCache.__len__` and
`KVCache.reset`.

**Test it:** `python3 -m unittest test_exercises.TestKVCache -v`

`test_cached_decoding_matches_a_full_forward` in that class also exercises
`MultiHeadAttention` (§3) and `causal_mask` (§2) end to end — the whole
point of the test is checking `KVCache` against a full forward pass, so
that coupling is intentional, not an isolation gap.

## 5. RoPE — rotary position embeddings

Attention itself has no notion of order: permute the input and the output
permutes with it (there is a test for exactly this). Position has to be
injected.

RoPE does it by **rotating** each pair of channels by an angle proportional
to the position:

```
θ_i = position / base^(2i/d)
```

Because a rotation is orthogonal, the dot product between two rotated
vectors depends only on their **relative** offset:

```
⟨R(p)q, R(p')k⟩  depends only on  p − p'
```

which is what attention actually needs, and why RoPE extrapolates to
lengths it never saw in training. That property has its own test here —
the same pair of vectors at offsets (1, 4) and (5, 8) must give the same
dot product.

```python
>>> cos, sin = rope_frequencies(head_dim=8, max_length=16)
>>> cos.shape, sin.shape
(torch.Size([16, 4]), torch.Size([16, 4]))
>>> x = torch.randn(1, 1, 1, 8)
>>> torch.equal(apply_rope(x, cos, sin, offset=0), x)   # cos(0)=1, sin(0)=0
True
>>> a, b = torch.randn(1, 1, 8), torch.randn(1, 1, 8)
>>> close = (apply_rope(a, cos, sin, offset=1) * apply_rope(b, cos, sin, offset=4)).sum()
>>> far   = (apply_rope(a, cos, sin, offset=5) * apply_rope(b, cos, sin, offset=8)).sum()
>>> close.item(), far.item()   # same relative offset (3) -> same dot product
(-1.1781532764434814, -1.1781535148620605)
```

Used by Llama, Mistral, Qwen and most current open models. Alternatives to
be able to name: learned absolute embeddings (BERT, GPT-2 — no
extrapolation), sinusoidal (the original paper), and ALiBi (a linear bias
on the attention scores).

**`offset` matters.** At decode step N the single new token is at position
N, not 0. Forgetting it produces a model that generates fluent text and
slowly loses track of where it is — a genuinely confusing bug, and one the
tests here pin down.

**In `exercises.py`:** `rope_frequencies` and `apply_rope`.

**Test it:** `python3 -m unittest test_exercises.TestRoPE -v`

## 6. Pre-norm blocks

```python
x = x + attention(norm(x))
x = x + mlp(norm(x))
```

**Pre**-norm, not post-norm. The original paper normalised *after* the
residual; pre-norm keeps a clean identity path from input to output, so
gradients reach the early layers without a warm-up schedule and deep stacks
train stably. Every modern LLM uses it.

A quick way to tell them apart, which the tests use: post-norm normalises
the block **output**, so its mean per position would be ~0. Pre-norm's is
not:

```python
>>> block = TransformerBlock(embed_dim=16, num_heads=4)
>>> x = torch.randn(1, 5, 16)
>>> out = block(x)
>>> out.mean(-1)          # per-position mean, far from 0 -> pre-norm
tensor([[-0.4533,  0.0581,  0.5772,  0.0629,  0.2951]], grad_fn=<MeanBackward1>)
>>> block.mlp[0].out_features   # 4x expansion by convention
64
```

The MLP expands by 4× by convention. **Most of the parameters in a
transformer live here, not in attention** — roughly 2/3 of them — which
surprises people and is a good thing to know when asked where to cut.

Modern variants: RMSNorm instead of LayerNorm (no mean subtraction,
cheaper), SwiGLU instead of GELU, and QK-norm for stability at scale.

**In `exercises.py`:** `TransformerBlock.forward`. It's built from pieces
finished in earlier sections — `norm1`/`norm2`/`mlp` are already wired up
in `__init__`; you're composing them with §3's `MultiHeadAttention` via the
two residual lines above.

**Test it:** `python3 -m unittest test_exercises.TestTransformerBlock -v`

## 7. The L² problem

```
FLOPs ≈ 2 · L² · D
```

Two matmuls dominate — `QK^T` and `weights @ V`, each `L·L·D`. Splitting
into heads does not change it: each head is `L·L·(D/H)` and there are H.

```python
>>> attention_flops(seq_len=4, embed_dim=2)
64
>>> attention_flops(200, 64) / attention_flops(100, 64)   # doubling L -> 4x
4.0
>>> attention_flops(128, 512, num_heads=8) == attention_flops(128, 512, num_heads=1)
True
```

**Doubling the context quadruples attention cost.** That single fact
motivates FlashAttention (same maths, tiled so it never materialises the
L×L matrix — an IO win, not a FLOP win), sliding-window attention, linear
attention, and every long-context paper of the last few years.

Note that at short sequence lengths the MLP actually dominates; attention
only takes over once L is comparable to D. Being able to say where the
crossover is marks the difference between reciting and understanding.

**In `exercises.py`:** `attention_flops`.

**Test it:** `python3 -m unittest test_exercises.TestAttentionFlops -v`

---

## The exercises

| Function | What it teaches |
|---|---|
| `scaled_dot_product_attention` | The four lines; why `sqrt(d_k)` |
| `causal_mask` | Lower triangular *including* the diagonal |
| `padding_mask` | Broadcasting lengths into a mask |
| `MultiHeadAttention` | The transpose-not-view split; parameter count |
| `KVCache` | What caching buys and what it costs |
| `rope_frequencies` / `apply_rope` | Relative position by rotation; `offset` |
| `TransformerBlock` | Pre-norm; residual paths; the 4× MLP |
| `attention_flops` | The L² scaling argument |

## Check your understanding

1. **Why does scaled dot-product attention divide by `sqrt(d_k)`
   before the softmax?**
   <details><summary>Answer</summary>Without scaling, dot products grow
   with the dimension `d_k`, pushing the softmax into a region with
   near-zero gradients (it saturates toward one-hot). Dividing by
   `sqrt(d_k)` keeps the pre-softmax values in a well-behaved
   range.</details>

2. **Why does `causal_mask` include the diagonal, rather than strictly
   masking everything at or after the current position?**
   <details><summary>Answer</summary>A token is always allowed to
   attend to itself — only *future* tokens must be masked out. The
   diagonal represents "attend to the current position," which is
   required, not forbidden.</details>

3. **`MultiHeadAttention`'s head split uses a transpose, not just a
   reshape. Why can't a plain `view` do the same job?**
   <details><summary>Answer</summary>Splitting `(batch, seq, d_model)`
   into per-head tensors requires moving the head dimension before the
   sequence dimension for the batched matmul to work per-head — that
   reordering is a transpose, and `view` can't express a change in
   dimension order, only a reshape of contiguous memory.</details>

4. **What does a KV cache actually save on, and what does it cost?**
   <details><summary>Answer</summary>It saves recomputing the key and
   value projections for every previously-seen token at each new
   decoding step — turning per-step cost from growing with the whole
   sequence to just the new token. The cost is memory, which grows with
   sequence length and must be stored for the whole generation.</details>

5. **Why does RoPE encode *relative* position through rotation, instead
   of adding an absolute position embedding?**
   <details><summary>Answer</summary>Rotating query and key vectors by
   an angle proportional to position means the dot product between two
   rotated vectors depends only on their *relative* offset, not their
   absolute positions — this generalizes better to sequence lengths
   not seen during training.</details>

6. **Why is attention's compute cost quadratic in sequence length, and
   what does that mean practically for long contexts?**
   <details><summary>Answer</summary>Every token attends to every other
   token, so the attention matrix itself is `(seq_len, seq_len)` —
   doubling the context length roughly quadruples the compute and
   memory for attention specifically, which is why long-context serving
   is a genuinely different engineering problem, not just "the same
   thing but bigger."</details>

## Interview questions this module answers

- *Implement scaled dot-product attention.*
- *Why divide by the square root of the head dimension?*
- *What exactly does multi-head buy you over one big attention?*
- *What does a KV cache store, and how big does it get?*
- *How does RoPE encode position, and why does it extrapolate?*
- *Why pre-norm instead of post-norm?*
- *Attention is O(L²) — what do you do about long contexts?*
- *Where do the parameters in a transformer actually live?*
