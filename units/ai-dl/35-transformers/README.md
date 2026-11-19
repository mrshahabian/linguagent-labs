# 35 — Assembling transformers end to end

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/ai-dl/35-transformers/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/ai-dl/transformers?utm_source=github&utm_medium=labs)

*~55 minutes · 19 tests*

Module 26 builds the primitives — scaled dot-product attention, causal
masks, multi-head projections, RoPE, one pre-norm block. This module
assembles them into something you can actually run: a stack of blocks with
learned position embeddings and a tied output head that produces logits
over a vocabulary, plus the encoder-decoder shape used by seq2seq models
like translation systems, which needs a third sublayer module 26 never
introduces — cross-attention between two different sequences.

Work in `exercises.py`. Check your work with `python -m labkit check 35`, or open the notebook (`lab.ipynb`) in Colab.

Each section below ends with **Test it** — the exact command for just that
piece, run from `units/ai-dl/35-transformers/`.

## What you'll build

| Name | You'll implement | Shows up in real code as |
|---|---|---|
| `sinusoidal_positional_encoding` | the function | The original Transformer paper's fixed position signal |
| `TransformerBlock` | `forward` | One decoder layer — repeated from module 26, stacked here |
| `GPTModel` | `__init__` weight tying, `forward` | A minimal, actually-runnable decoder-only language model |
| `CrossAttentionDecoderBlock` | `forward` | The seq2seq decoder layer — self-attn, cross-attn, MLP |
| `count_transformer_parameters` | the function | Estimating model size before you ever load weights |
| `generate_greedy` | the function | The autoregressive loop every `model.generate()` runs |

---

## 1. Fixed vs. learned vs. relative position signals

```python
position = torch.arange(seq_len).unsqueeze(1).float()
div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
encoding[:, 0::2] = torch.sin(position * div_term)
encoding[:, 1::2] = torch.cos(position * div_term)
```

Three ways to give attention a sense of order, and this module uses all
three across its pieces:

- **Sinusoidal** (here): a fixed, parameter-free function of position,
  added to the token embedding before the first block. Extrapolates to
  sequence lengths never seen in training, since it's a formula, not a
  lookup table.
- **Learned absolute** (`GPTModel.position_embedding` below): a plain
  `nn.Embedding` indexed by position. Simpler, but has no row for a
  position past whatever `max_seq_len` it was created with.
- **RoPE** (module 26): rotates Q/K *inside* attention based on relative
  offset between tokens, rather than adding anything to the input at all.

```python
>>> enc = sinusoidal_positional_encoding(10, 16)
>>> enc[0, 0::2]     # position 0: sin(0) = 0 for every even dimension
tensor([0., 0., 0., 0., 0., 0., 0., 0.])
>>> enc[0, 1::2]     # position 0: cos(0) = 1 for every odd dimension
tensor([1., 1., 1., 1., 1., 1., 1., 1.])
```

**In `exercises.py`:** `sinusoidal_positional_encoding`.

**Test it:** `python3 -m unittest test_exercises.TestSinusoidalPositionalEncoding -v`

## 2. Stacking blocks, and tying the output head

```python
self.token_embedding = nn.Embedding(vocab_size, embed_dim)
self.blocks = nn.ModuleList([TransformerBlock(...) for _ in range(num_layers)])
self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
self.lm_head.weight = self.token_embedding.weight
```

`GPTModel` is genuinely just: embed tokens, add learned positions, run
through `num_layers` stacked blocks (an `nn.ModuleList`, not a Python
list — module 24 covers why that distinction matters for parameter
registration), normalize, project back to vocabulary logits.

**Weight tying** is the detail worth stopping on. The token embedding
table and the output projection both have shape `(vocab_size,
embed_dim)` — one looks a row up by index, the other computes a dot
product against every row to produce logits. Setting
`self.lm_head.weight = self.token_embedding.weight` makes them the
*same tensor* (verified in the tests with `assertIs`, not just checking
the values match) — a gradient through either path updates the shared
matrix. This roughly halves the parameters spent on the vocabulary in a
model where the embedding table is often the single largest matrix,
and it's standard in GPT-2 and most open LLM architectures since.

```python
>>> model = GPTModel(vocab_size=50, embed_dim=32, num_layers=2, num_heads=4, max_seq_len=20)
>>> model.lm_head.weight is model.token_embedding.weight
True
>>> model(torch.randint(0, 50, (2, 6))).shape
torch.Size([2, 6, 50])
```

**In `exercises.py`:** `TransformerBlock.forward`, `GPTModel.__init__`
(the tying line), `GPTModel.forward`.

**Test it:** `python3 -m unittest test_exercises.TestGPTModel -v`

## 3. The third sublayer: cross-attention

```python
self_attended, _ = self.self_attention(normed, normed, normed, attn_mask=causal_mask)
x = x + self_attended
normed = self.norm2(x)
cross_attended, _ = self.cross_attention(normed, encoder_output, encoder_output)
x = x + cross_attended
```

A translation model's decoder needs to look at **two** things: the
target sequence it's building (via causal self-attention, same as
`GPTModel`) and the **source** sequence's encoded representation (via
cross-attention). This is a third residual sublayer between self-attention
and the MLP — queries come from the target sequence, keys and values come
from the encoder's output, and the two sequences are free to have
different lengths, exactly the same cross-attention mechanic as module
31's VLM fusion, just applied between two sequences of the same modality
(source and target text) instead of text and image patches.

```python
>>> block = CrossAttentionDecoderBlock(embed_dim=32, num_heads=4)
>>> block(target_embeddings, encoder_output, causal_mask).shape
torch.Size([2, 5, 32])   # target's shape - encoder_output can be a different length
```

**In `exercises.py`:** `CrossAttentionDecoderBlock.forward`.

**Test it:** `python3 -m unittest test_exercises.TestCrossAttentionDecoderBlock -v`

## 4. Estimating model size without loading it

```python
attention_params = 4 * (embed_dim * embed_dim + embed_dim)   # Q, K, V, output
mlp_params = 2 * embed_dim * (embed_dim * mlp_ratio) + embed_dim * mlp_ratio + embed_dim
```

Attention's parameter count does **not** depend on the number of heads —
splitting into heads reshapes the same `(embed_dim, embed_dim)` weight
matrices, it doesn't add more of them. The MLP dominates: at the
conventional `mlp_ratio=4`, its two linear layers hold roughly 8x
`embed_dim²` versus attention's roughly 4x, which is why "most of a
transformer's parameters live in the MLP, not attention" is a true
statement worth having ready. This formula matches a real instantiated
`GPTModel` parameter-for-parameter — the test asserts exact equality, not
an approximation.

```python
>>> count_transformer_parameters(vocab_size=50257, embed_dim=768, num_layers=12, max_seq_len=1024)
124439808   # GPT-2 small, roughly
```

**In `exercises.py`:** `count_transformer_parameters`.

**Test it:** `python3 -m unittest test_exercises.TestCountTransformerParameters -v`

## 5. The generation loop underneath `model.generate()`

```python
for _ in range(max_new_tokens):
    logits = model(input_ids)
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    input_ids = torch.cat([input_ids, next_token], dim=1)
```

No KV cache here — module 26 covers that optimisation on its own, and
this loop is exactly what it optimises. Every step reruns the **entire**
sequence through the model, reads the last position's logits, greedily
picks the highest-probability next token (module 28 covers the
alternatives — temperature, top-k, nucleus, beam search), and appends
it. This is the shape every decoding strategy is built around;
understanding this loop plainly is what makes the fancier strategies in
module 28 legible as *modifications to one step* of it, not a different
algorithm.

```python
>>> generate_greedy(model, torch.randint(0, 50, (1, 3)), max_new_tokens=4).shape
torch.Size([1, 7])
```

**In `exercises.py`:** `generate_greedy`.

**Test it:** `python3 -m unittest test_exercises.TestGenerateGreedy -v`

---

## The exercises

| Function / class | What it teaches |
|---|---|
| `sinusoidal_positional_encoding` | Fixed position signal; extrapolates past training lengths |
| `GPTModel` | Stacking blocks in an `nn.ModuleList`; weight tying |
| `CrossAttentionDecoderBlock` | The third sublayer a seq2seq decoder needs |
| `count_transformer_parameters` | Why the MLP, not attention, dominates parameter count |
| `generate_greedy` | The loop every decoding strategy modifies one step of |

Every function in this table has a stub in `exercises.py` and test
coverage in `test_exercises.py` — a clean 1:1 mapping.

## Check your understanding

1. **Why does sinusoidal positional encoding extrapolate to sequence
   lengths longer than anything seen in training, while a learned
   absolute position embedding does not?**
   <details><summary>Answer</summary>Sinusoidal encoding is computed from
   a formula for any position, so position 10001 is just as well-defined
   as position 5 even if training only ever saw sequences up to length
   1000. A learned embedding table has one row per position up to
   whatever <code>max_seq_len</code> it was created with — there is no
   row to look up for a position beyond that, and no formula to fall
   back on.</details>

2. **Why does `self.lm_head.weight = self.token_embedding.weight` work
   — what makes those two matrices interchangeable?**
   <details><summary>Answer</summary>Both have exactly the same shape,
   <code>(vocab_size, embed_dim)</code> — the embedding looks up a row
   by token index, and the output head computes a dot product against
   every row to score each vocabulary entry. Tying assumes a token's
   input representation and its "does this look like the right output"
   representation can reasonably be the same vector, which halves the
   parameters spent on the vocabulary and is standard practice in most
   open decoder-only models.</details>

3. **What's the one structural difference between `GPTModel`'s block and
   `CrossAttentionDecoderBlock`?**
   <details><summary>Answer</summary>A third residual sublayer —
   cross-attention — sits between the causal self-attention and the MLP.
   Its queries come from the target sequence being decoded, but its keys
   and values come from a separate encoder's output, which is how
   information about a source sequence (e.g. the sentence being
   translated) enters the decoder at all.</details>

4. **Why does attention's parameter count not depend on the number of
   heads, while its computational cost (module 26) very much does
   depend on sequence length?**
   <details><summary>Answer</summary>Splitting embed_dim into multiple
   heads reshapes the same <code>Q</code>/<code>K</code>/<code>V</code>
   projection matrices into smaller chunks — it doesn't allocate new
   parameters, so the total parameter count stays
   <code>4 * (embed_dim² + embed_dim)</code> regardless of head count.
   Compute cost is a separate question entirely, governed by the
   <code>L²</code> term from the sequence length, which is exactly why
   module 26 treats them as different concerns.</details>

5. **What does `generate_greedy` leave out that a production serving
   system would add, and why doesn't that change correctness?**
   <details><summary>Answer</summary>A KV cache — instead of recomputing
   attention over the whole sequence at every step, a cache reuses the
   previous steps' key/value projections and only computes the new
   token's contribution. It's purely a performance optimisation: the
   output token sequence is identical either way, since the cache stores
   exactly the same values the recomputation would have produced.</details>

## Interview questions this module answers

- *Why do we tie the input embedding and output projection weights?*
- *Design the decoder for a translation model — what's different from a
  GPT-style decoder?*
- *Estimate the parameter count of this model without loading it —
  where do the parameters actually go?*
- *Walk me through what `model.generate()` does, one token at a time.*
- *Sinusoidal encoding vs. a learned position embedding vs. RoPE — when
  would you pick each?*
