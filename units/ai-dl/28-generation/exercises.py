"""28 - decoding, sampling and evaluation

Implement each function so the tests in test_exercises.py pass. Read
README.md first; every exercise below is covered there.

    ../run 28            test your work
    ../run 28 --check    watch the reference solutions pass

The docstrings are the specification. Standard library only.
"""

import torch
import torch.nn.functional as F


def apply_temperature(logits, temperature):
    """Divide logits by temperature before the softmax.

    T < 1 sharpens the distribution (more deterministic), T > 1 flattens it
    (more diverse). The reason it is applied to LOGITS and not to
    probabilities is that dividing probabilities does not renormalise into
    anything meaningful.

    T -> 0 is greedy decoding, and is handled as a special case because the
    division would overflow to inf/NaN.
    """
    raise NotImplementedError("your code here")


def top_k_filter(logits, k):
    """Keep the k highest logits, set the rest to -inf.

    Truncating the tail is what stops a model occasionally sampling a
    genuinely absurd token: over a 50k vocabulary the tail holds most of
    the probability mass by count, even when each token is individually
    unlikely.

    k >= vocab size is a no-op rather than an error, which is what callers
    sweeping k expect.
    """
    raise NotImplementedError("your code here")


def top_p_filter(logits, p):
    """Nucleus sampling: keep the smallest set of tokens with mass >= p.

    Better than top-k because the cut adapts to the distribution. Where the
    model is confident the nucleus is one or two tokens; where it is
    genuinely uncertain it widens. Top-k cannot do that - a fixed k is
    either too tight in the first case or too loose in the second.

    The most-likely token is ALWAYS kept, even when it alone exceeds p.
    Dropping it would leave nothing to sample from.
    """
    raise NotImplementedError("your code here")


def repetition_penalty(logits, generated, penalty=1.1):
    """Divide the logits of already-generated tokens (the CTRL formulation).

    The subtlety is the sign: for a POSITIVE logit you divide, for a
    NEGATIVE one you multiply. Dividing a negative logit by 1.1 makes it
    larger, which rewards the repetition you were trying to suppress.
    Getting this backwards is a real bug in published code.
    """
    raise NotImplementedError("your code here")


def sample_next_token(logits, temperature=1.0, top_k=None, top_p=None,
                      generator=None):
    """Apply the filters in the standard order, then sample one token.

    Order matters and is the thing to state out loud: temperature FIRST
    (it changes the shape the filters see), then top-k, then top-p, then
    softmax and sample. Filtering before scaling gives a different, and
    wrong, distribution.
    """
    raise NotImplementedError("your code here")


def greedy_decode(step_fn, prompt, max_new_tokens, eos_token=None):
    """Decode greedily, stopping at eos_token.

    `step_fn(tokens) -> logits` for the NEXT token, so this works against a
    real model or a stub. Greedy is deterministic and therefore the only
    decode you can unit-test exactly - which is why it is the one to reach
    for when debugging a generation bug.

    Greedy is not the highest-probability SEQUENCE: picking the best token
    at each step can walk into a dead end that beam search avoids. That
    distinction is a standard interview question.
    """
    raise NotImplementedError("your code here")


def beam_search(step_fn, prompt, max_new_tokens, beam_width=3,
                eos_token=None, length_penalty=1.0):
    """Keep the `beam_width` highest-scoring sequences at each step.

    Scores are summed LOG-probabilities, not multiplied probabilities:
    multiplying 50 numbers below 1 underflows float32 long before the
    sequence ends.

    `length_penalty` divides by len**alpha before comparing. Without it,
    beam search has a well-known bias towards short sequences, because
    every extra token adds another negative log-probability.
    """
    raise NotImplementedError("your code here")


def perplexity(logits, targets, ignore_index=-100):
    """exp(mean cross-entropy) over the non-ignored positions.

    logits: (B, L, V), targets: (B, L).

    Perplexity is "how many tokens is the model effectively choosing
    between". A uniform model over V tokens has perplexity exactly V, which
    is the sanity check to run first - and the reason `ignore_index`
    matters: counting padded positions as correct makes perplexity look far
    better than it is.
    """
    raise NotImplementedError("your code here")


def shift_for_causal_lm(input_ids, ignore_index=-100):
    """Build (inputs, labels) for next-token prediction.

    Position i must predict token i+1, so the labels are the inputs shifted
    left by one. The final position has nothing to predict and is masked
    out with ignore_index.

    Off-by-one here is the most expensive bug in language modelling,
    because it does not crash: the model learns to copy its input, the loss
    drops beautifully, and generation produces fluent nonsense.
    """
    raise NotImplementedError("your code here")
