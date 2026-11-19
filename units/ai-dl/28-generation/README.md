# 28 — Decoding, sampling and evaluation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/ai-dl/28-generation/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/ai-dl/generation?utm_source=github&utm_medium=labs)

*~50 minutes · 74 tests*

A language model outputs a distribution over the vocabulary. Turning that
into text is a separate algorithm, with its own parameters, its own bugs,
and far more influence on perceived output quality than most people expect.

This is also where LLM interviews get practical: *"the model repeats
itself — what knobs do you have?"*, *"what does temperature 0.7 actually
do?"*, *"why is greedy not the most likely sentence?"*

Work in `exercises.py`. Check your work with `python -m labkit check 28`, or open the notebook (`lab.ipynb`) in Colab.

Each section below ends with **Test it** — the exact command for just that
piece, run from `units/ai-dl/28-generation/`. No need to finish the whole
file before checking your work.

## What you'll build

9 functions in `exercises.py`, in this order. Keep this table open as you
go — each section below tells you which row it's building:

| Function | You'll implement | Shows up in real code as |
|---|---|---|
| `apply_temperature` | dividing logits by T; T=0 as a special case | The `temperature` parameter on every inference API — the cheapest knob for making a chatbot feel more or less deterministic |
| `top_k_filter` | keeping the k highest logits, `-inf` elsewhere | The `top_k` argument to `generate()` — the guard against a model occasionally sampling a genuinely bizarre token |
| `top_p_filter` | cumulative-mass cutoff, keeping the crossing token | The `top_p`/nucleus argument most inference servers default to over top-k, because it adapts to how confident the model is at each step |
| `repetition_penalty` | the CTRL sign branch | The knob you reach for first when a chatbot gets stuck repeating the same phrase |
| `sample_next_token` | composing temperature → top-k → top-p → sample | The one function an autoregressive decoding loop calls once per generated token — the guts of `model.generate()` |
| `greedy_decode` | argmax at every step, stopping at eos | The default decode wherever determinism matters more than diversity — function calling, code generation, debugging a generation bug |
| `beam_search` | log-space scores, length penalty | Standard in translation and summarisation, where there's roughly one right answer and diversity would hurt |
| `perplexity` | `exp(mean cross-entropy)`, respecting `ignore_index` | The first sanity-check metric run on a new checkpoint, and the number reported in most language-model papers |
| `shift_for_causal_lm` | building `(inputs, labels)` from one sequence | The line at the top of every causal-LM training loop; get it wrong and the model silently learns to copy its input |

---

## 1. Temperature

```python
logits / temperature
```

Applied to **logits**, before the softmax. Dividing probabilities instead
does not renormalise into anything meaningful.

- **T < 1** sharpens — more deterministic, more repetitive.
- **T > 1** flattens — more diverse, more incoherent.
- **T → 0** is greedy. Handle it as a special case; the division overflows
  to inf/NaN.

Temperature never changes the **ranking**, only the gaps. That is why it
cannot fix a model that puts its mass on the wrong token.

```python
>>> logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
>>> apply_temperature(logits, 2.0)          # T > 1: flattens
tensor([[0.5000, 1.0000, 1.5000, 2.0000]])
>>> apply_temperature(logits, 0.0).softmax(-1)   # T -> 0: greedy, no NaN
tensor([[0., 0., 0., 1.]])
```

**In `exercises.py`:** `apply_temperature` is this section's target — the
division, plus the `T == 0` special case and a `ValueError` on negative T.

**Test it:** `python3 -m unittest test_exercises.TestTemperature -v`

## 2. Top-k

Keep the k highest logits, set the rest to `-inf`.

Why truncate at all: over a 50k vocabulary the tail holds a large share of
the probability mass *by count*, even when each individual token is
absurd. Sample 200 tokens and you will eventually draw one.

The weakness: **k is fixed**. Where the model is confident, k=50 is far too
loose and lets in noise; where it is genuinely uncertain, k=50 is too tight
and cuts off real options.

```python
>>> logits = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])
>>> top_k_filter(logits, 2)                  # only indices 1 and 4 survive
tensor([[-3.4028e+38,  5.0000e+00, -3.4028e+38, -3.4028e+38,  4.0000e+00]])
>>> top_k_filter(logits, 2).softmax(-1)      # renormalises over just those two
tensor([[0.0000, 0.7311, 0.0000, 0.0000, 0.2689]])
```

**In `exercises.py`:** `top_k_filter` is this section's target — mask every
logit below the k-th largest, and treat `k >= vocab size` as a no-op rather
than an error.

**Test it:** `python3 -m unittest test_exercises.TestTopK -v`

## 3. Top-p (nucleus sampling)

Keep the smallest set of tokens whose cumulative probability reaches p.

This is the fix for top-k's weakness: the cut **adapts to the
distribution**. Confident step → the nucleus is one or two tokens.
Uncertain step → it widens. The tests here assert both behaviours
explicitly.

Two implementation details that are easy to get wrong:

- **The crossing token is included.** You want the smallest set with mass
  *≥* p, so the token that takes you over the line is kept. Shifting the
  cumulative sum right by one is what does that.
- **The most likely token is always kept**, even when it alone exceeds p.
  Otherwise a very confident step leaves nothing to sample from.

Typical settings: `temperature 0.7–0.9` with `top_p 0.9–0.95`. Top-k and
top-p compose — most inference servers apply both.

```python
>>> logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))
>>> top_p_filter(logits, 0.8)                # 0.5 + 0.3 = 0.8 crosses the line
tensor([[-6.9315e-01, -1.2040e+00, -3.4028e+38, -3.4028e+38]])
>>> top_p_filter(logits, 0.8).softmax(-1)    # renormalised over the nucleus
tensor([[0.6250, 0.3750, 0.0000, 0.0000]])
```

**In `exercises.py`:** `top_p_filter` is this section's target — sort
descending, mask everything after the token that crosses p, and always keep
the most likely token.

**Test it:** `python3 -m unittest test_exercises.TestTopP -v`

## 4. Repetition penalty

The CTRL formulation:

```python
logits[token] = logit / penalty  if logit > 0 else logit * penalty
```

**The sign branch is the point.** Dividing a *negative* logit by 1.1 makes
it larger — rewarding exactly the repetition you were trying to suppress.
This is a real bug in published code, and asking about it is a good way to
tell whether someone has implemented sampling or only used it.

Alternatives: frequency and presence penalties (subtract a constant, or
something proportional to the count — the OpenAI API's version), and
no-repeat-n-gram blocking (hard-ban any n-gram already produced).

```python
>>> repetition_penalty(torch.tensor([[2.0, 1.0]]), [[0]], 2.0)   # positive logit: divided
tensor([[1., 1.]])
>>> repetition_penalty(torch.tensor([[-2.0, 1.0]]), [[0]], 2.0)  # negative logit: multiplied
tensor([[-4.,  1.]])
```

**In `exercises.py`:** `repetition_penalty` is this section's target — the
sign-dependent divide/multiply, applied per row to the tokens already seen
by that row.

**Test it:** `python3 -m unittest test_exercises.TestRepetitionPenalty -v`

## 5. Order of operations

```
temperature → top-k → top-p → softmax → sample
```

Temperature **first**, because it changes the shape the filters see.
Filtering before scaling gives a different — and wrong — distribution. At
T=100 the logits are nearly uniform so `top_p=0.5` keeps about half the
vocabulary; filter first and it keeps one token. There is a test for
exactly that.

```python
>>> gen = torch.Generator().manual_seed(0)
>>> logits = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])
>>> sample_next_token(logits, generator=gen)
tensor([1])
>>> sample_next_token(logits, top_k=1)   # only one token survives -> deterministic
tensor([1])
```

**In `exercises.py`:** `sample_next_token` is this section's target — it
composes §1–§3's `apply_temperature`, `top_k_filter` and `top_p_filter` in
that order, then samples with `torch.multinomial`. Those three need to work
first; a stubbed filter makes this section's tests fail for the wrong
reason.

**Test it:** `python3 -m unittest test_exercises.TestSampleNextToken -v`

## 6. Greedy vs beam search

### Greedy decoding

**Greedy** takes the argmax at every step. It is deterministic, which makes
it the only decode you can unit-test exactly — reach for it first when
debugging a generation bug.

Greedy is **not** the most likely *sequence*. Picking the best token now
can walk into a dead end:

```
0 → 1 (0.6) → eos (0.5)  =  0.30      greedy takes this
0 → 2 (0.4) → eos (0.9)  =  0.36      beam finds this
```

That is exactly the toy model the tests use.

```python
>>> greedy_decode(ToyModel(), [0], max_new_tokens=10, eos_token=3)
[0, 1, 3]
```

**In `exercises.py`:** `greedy_decode` is this subsection's target — argmax
each step via `step_fn(tokens)`, append, stop at `eos_token` if given.

**Test it:** `python3 -m unittest test_exercises.TestGreedyDecode -v`

### Beam search

**Beam search** keeps the `beam_width` highest-scoring prefixes at each
step. Two details:

- **Sum log-probabilities, do not multiply probabilities.** Multiplying 50
  numbers below 1 underflows float32 long before the sequence ends.
- **Length penalty.** Every extra token adds another negative
  log-probability, so beam search has a well-known bias towards short
  outputs. Dividing by `len^alpha` before comparing corrects it; α≈0.6–1.0
  is typical.

Beam search is standard in translation and summarisation, where there is
roughly one right answer. It is **not** used for open-ended chat — it
produces bland, repetitive text, because the highest-probability sequence
is usually the most generic one. Knowing *when not* to use it is the
senior-level half of the answer.

```python
>>> greedy_decode(TrapModel(), [0], 2, eos_token=3)                  # takes the trap
[0, 1, 3]
>>> beam_search(TrapModel(), [0], 2, beam_width=3, eos_token=3)      # finds the better path
[0, 2, 3]
```

**In `exercises.py`:** `beam_search` is this subsection's target — expand
every live beam by its top `beam_width` continuations, sum log-probs, keep
the best `beam_width` by length-penalised score, repeat. It calls the same
`step_fn` protocol as `greedy_decode`, and two of its tests
(`test_width_one_matches_greedy`, `test_it_beats_greedy_on_the_trap`)
compare directly against `greedy_decode`'s output, so that needs to be
working too.

**Test it:** `python3 -m unittest test_exercises.TestBeamSearch -v`

## 7. Perplexity

```
perplexity = exp(mean cross-entropy)
```

"How many tokens is the model effectively choosing between." A uniform
model over V tokens has perplexity exactly V — the sanity check to run
first, and one of the tests here.

`ignore_index` matters more than it looks: counting padded positions makes
perplexity look far better than it is, because padding is trivially
predictable. There is a test showing precisely that inflation.

Perplexity is only comparable across models with the **same tokenizer**.
A model with a larger vocabulary needs fewer tokens per sentence and gets a
different number for the same text. That caveat comes up whenever someone
compares perplexities from two papers.

```python
>>> logits = torch.zeros(1, 4, 7)
>>> targets = torch.zeros(1, 4, dtype=torch.long)
>>> perplexity(logits, targets)                       # uniform over 7 tokens
tensor(7.0000)
>>> perplexity(logits, torch.tensor([[0, 0, -100, -100]]))  # padding ignored
tensor(7.0000)
```

**In `exercises.py`:** `perplexity` is this section's target —
`F.cross_entropy(..., ignore_index=ignore_index).exp()` over the flattened
logits and targets.

**Test it:** `python3 -m unittest test_exercises.TestPerplexity -v`

## 8. The label shift

```python
inputs = ids[..., :-1]
labels = ids[..., 1:]
```

Position i predicts token i+1. The last position has nothing to predict.

**This off-by-one is the most expensive bug in language modelling**,
because it does not crash. If labels equal inputs, the model learns to copy
its input, the loss drops beautifully, every metric looks excellent, and
generation produces fluent nonsense. You lose a training run to it.

`.clone()` the labels — a view into the same storage means masking a label
position corrupts the input.

Note that HuggingFace models shift internally when you pass `labels=`, so
doing it yourself as well shifts twice. Know which convention your code is
in.

```python
>>> ids = torch.tensor([[1, 2, 3, 4]])
>>> inputs, labels = shift_for_causal_lm(ids)
>>> inputs
tensor([[1, 2, 3]])
>>> labels
tensor([[2, 3, 4]])
```

**In `exercises.py`:** `shift_for_causal_lm` is this section's target —
slice `inputs`/`labels` as above, `.clone()` the labels, and raise
`ValueError` on a sequence shorter than 2 tokens. One test
(`test_it_composes_with_perplexity`) feeds this section's output straight
into §7's `perplexity`, so that needs to be working too, though
`shift_for_causal_lm` itself does not depend on it.

**Test it:** `python3 -m unittest test_exercises.TestShiftForCausalLm -v`

---

## The exercises

| Function | What it teaches |
|---|---|
| `apply_temperature` | Scaling logits; T=0 as a special case |
| `top_k_filter` | Truncating the tail; why k is inflexible |
| `top_p_filter` | Adaptive nucleus; keeping the crossing token |
| `repetition_penalty` | The sign branch |
| `sample_next_token` | The order of operations |
| `greedy_decode` | Deterministic decoding; eos handling |
| `beam_search` | Log-space scores; length penalty; the dead end |
| `perplexity` | exp(cross-entropy); `ignore_index` |
| `shift_for_causal_lm` | The off-by-one that does not crash |

## Check your understanding

1. **Why does temperature `T=0` need to be handled as a special case
   instead of just dividing logits by it?**
   <details><summary>Answer</summary>Dividing by 0 is undefined —
   <code>T=0</code> means "always pick the single highest-probability
   token," which is exactly greedy decoding, and has to be implemented
   as that directly rather than as a division.</details>

2. **Why is a fixed `top_k` sometimes the wrong choice compared to
   `top_p` (nucleus sampling)?**
   <details><summary>Answer</summary>A fixed k doesn't adapt to how
   peaked or flat the distribution is — on a confident distribution, k
   candidates might include implausible tail tokens; on a flat one, k
   might exclude reasonable options. `top_p` adapts by keeping however
   many tokens are needed to reach a probability mass.</details>

3. **In `top_p_filter`, why does the token that crosses the cumulative
   probability threshold get kept, not dropped?**
   <details><summary>Answer</summary>Dropping it could leave the
   remaining probability mass below `p`, technically violating the
   nucleus definition — keeping the crossing token guarantees the kept
   set's cumulative probability is at least `p`.</details>

4. **Why must `repetition_penalty` branch on the sign of the logit
   instead of applying the same formula uniformly?**
   <details><summary>Answer</summary>Dividing a positive logit by the
   penalty reduces it, but dividing a *negative* logit by the same
   penalty makes it more negative — i.e. increases its magnitude, doing
   the opposite of what's intended. The sign has to flip which operation
   (multiply vs divide) is applied.</details>

5. **What does length penalty in beam search actually correct for, and
   why is it needed?**
   <details><summary>Answer</summary>Beam search scores use summed
   log-probabilities, which are always negative and get more negative
   with every additional token — without a length penalty, shorter
   sequences are unfairly favored regardless of quality, since they
   accumulate fewer negative terms.</details>

6. **Why does `shift_for_causal_lm`'s off-by-one *not* crash if it's
   done wrong?**
   <details><summary>Answer</summary>Misaligning input and target by one
   position still produces tensors of matching, valid shapes — the model
   trains and loss goes down, just on the wrong objective (predicting
   the current token instead of the next one), so nothing errors, it
   just silently learns the wrong thing.</details>

## Interview questions this module answers

- *What does temperature actually do?*
- *Top-k or top-p — what is the difference and when does it matter?*
- *The model keeps repeating itself. What are your options?*
- *Why is greedy decoding not the most likely sentence?*
- *When would you use beam search, and when would you not?*
- *What is perplexity, and when can you compare two of them?*
- *How do you build the labels for causal language modelling?*
- *In what order do the sampling parameters get applied?*
