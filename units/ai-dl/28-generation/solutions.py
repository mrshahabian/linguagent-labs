"""Reference answers for 28 - decoding, sampling and evaluation."""

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
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if temperature == 0:
        out = torch.full_like(logits, torch.finfo(logits.dtype).min)
        out.scatter_(-1, logits.argmax(-1, keepdim=True), 0.0)
        return out
    return logits / temperature


def top_k_filter(logits, k):
    """Keep the k highest logits, set the rest to -inf.

    Truncating the tail is what stops a model occasionally sampling a
    genuinely absurd token: over a 50k vocabulary the tail holds most of
    the probability mass by count, even when each token is individually
    unlikely.

    k >= vocab size is a no-op rather than an error, which is what callers
    sweeping k expect.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    k = min(k, logits.shape[-1])
    threshold = logits.topk(k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold,
                              torch.finfo(logits.dtype).min)


def top_p_filter(logits, p):
    """Nucleus sampling: keep the smallest set of tokens with mass >= p.

    Better than top-k because the cut adapts to the distribution. Where the
    model is confident the nucleus is one or two tokens; where it is
    genuinely uncertain it widens. Top-k cannot do that - a fixed k is
    either too tight in the first case or too loose in the second.

    The most-likely token is ALWAYS kept, even when it alone exceeds p.
    Dropping it would leave nothing to sample from.
    """
    if not 0 < p <= 1:
        raise ValueError("p must be in (0, 1]")
    ordered, indices = logits.sort(dim=-1, descending=True)
    cumulative = ordered.softmax(-1).cumsum(-1)
    # Remove everything strictly after the token that crosses p. Shifting
    # right is what keeps the crossing token itself.
    remove = cumulative - ordered.softmax(-1) >= p
    remove[..., 0] = False
    remove = remove.scatter(-1, indices, remove)
    return logits.masked_fill(remove, torch.finfo(logits.dtype).min)


def repetition_penalty(logits, generated, penalty=1.1):
    """Divide the logits of already-generated tokens (the CTRL formulation).

    The subtlety is the sign: for a POSITIVE logit you divide, for a
    NEGATIVE one you multiply. Dividing a negative logit by 1.1 makes it
    larger, which rewards the repetition you were trying to suppress.
    Getting this backwards is a real bug in published code.
    """
    if penalty <= 0:
        raise ValueError("penalty must be positive")
    out = logits.clone()
    for row, tokens in enumerate(generated):
        if len(tokens) == 0:
            continue
        unique = torch.as_tensor(tokens, device=logits.device,
                                 dtype=torch.long).unique()
        selected = out[row, unique]
        out[row, unique] = torch.where(selected > 0, selected / penalty,
                                       selected * penalty)
    return out


def sample_next_token(logits, temperature=1.0, top_k=None, top_p=None,
                      generator=None):
    """Apply the filters in the standard order, then sample one token.

    Order matters and is the thing to state out loud: temperature FIRST
    (it changes the shape the filters see), then top-k, then top-p, then
    softmax and sample. Filtering before scaling gives a different, and
    wrong, distribution.
    """
    logits = apply_temperature(logits, temperature)
    if top_k is not None:
        logits = top_k_filter(logits, top_k)
    if top_p is not None:
        logits = top_p_filter(logits, top_p)
    probabilities = logits.softmax(-1)
    return torch.multinomial(probabilities, num_samples=1,
                             generator=generator).squeeze(-1)


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
    tokens = list(prompt)
    for _ in range(max_new_tokens):
        logits = step_fn(tokens)
        next_token = int(logits.argmax(-1))
        tokens.append(next_token)
        if eos_token is not None and next_token == eos_token:
            break
    return tokens


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
    beams = [(0.0, list(prompt), False)]
    for _ in range(max_new_tokens):
        if all(finished for _, _, finished in beams):
            break
        candidates = []
        for score, tokens, finished in beams:
            if finished:
                candidates.append((score, tokens, True))
                continue
            log_probabilities = F.log_softmax(step_fn(tokens).float(), dim=-1)
            top = log_probabilities.topk(beam_width)
            for value, index in zip(top.values.tolist(),
                                    top.indices.tolist()):
                candidates.append((score + value, tokens + [index],
                                   index == eos_token))
        candidates.sort(
            key=lambda c: c[0] / (len(c[1]) ** length_penalty), reverse=True)
        beams = candidates[:beam_width]
    return max(beams,
               key=lambda c: c[0] / (len(c[1]) ** length_penalty))[1]


def perplexity(logits, targets, ignore_index=-100):
    """exp(mean cross-entropy) over the non-ignored positions.

    logits: (B, L, V), targets: (B, L).

    Perplexity is "how many tokens is the model effectively choosing
    between". A uniform model over V tokens has perplexity exactly V, which
    is the sanity check to run first - and the reason `ignore_index`
    matters: counting padded positions as correct makes perplexity look far
    better than it is.
    """
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                           targets.reshape(-1), ignore_index=ignore_index)
    return loss.exp()


def shift_for_causal_lm(input_ids, ignore_index=-100):
    """Build (inputs, labels) for next-token prediction.

    Position i must predict token i+1, so the labels are the inputs shifted
    left by one. The final position has nothing to predict and is masked
    out with ignore_index.

    Off-by-one here is the most expensive bug in language modelling,
    because it does not crash: the model learns to copy its input, the loss
    drops beautifully, and generation produces fluent nonsense.
    """
    if input_ids.shape[-1] < 2:
        raise ValueError("need at least two tokens to build a shifted target")
    inputs = input_ids[..., :-1]
    labels = input_ids[..., 1:].clone()
    return inputs, labels
