# Domain: what prefill, decode and the KV cache actually are

You can solve every level treating this as an abstract resource puzzle. But
the rules stop feeling arbitrary once you know what they model, and "why is it
like that?" is a normal interview follow-up. This chapter is the theory.

## Generation is one token at a time

A language model does not produce an answer. It produces *one next token*,
given everything before it. To produce a sentence you run it repeatedly,
appending each token to the input and running again.

```
prompt: "The capital of France is"
step 1 -> " Paris"
step 2 -> "."
step 3 -> <end>
```

That loop is inherently **sequential**: token *n* cannot be computed before
token *n-1* exists. This one fact generates almost every rule in the exercise.

## Why there are two kinds of work

Naively, generating token *n* means re-running the model over all *n-1*
previous tokens. That would be quadratic and hopeless.

The fix is caching. For each token the model computes two vectors — a **key**
and a **value** — used by the attention mechanism. Crucially, a token's key
and value never change once computed: they do not depend on what comes after.
So you compute them once and keep them. That store is the **KV cache**.

This splits the work into two phases with completely different shapes:

**Prefill** — process the whole prompt, filling the KV cache for every prompt
token, and produce the first output token.

- The prompt tokens are all known up front, so they can be processed **in
  parallel**. On a GPU that is one big matrix multiply.
- Cost scales with prompt length: a 600-token prompt is roughly 600 units of
  work. Compute-bound.

**Decode** — produce each subsequent token.

- Reuses the cached keys and values, so only the *one* new token is processed.
- Cost is ~1 unit regardless of how long the conversation already is.
- Memory-bandwidth-bound: it does very little arithmetic per byte of weights
  it reads.

That is the model the exercise uses exactly:

| | cost | can be batched | parallel within one request |
|---|---|---|---|
| prefill of *n* tokens | *n* units | yes | yes — one step |
| decode | 1 unit | yes, across requests | no — strictly sequential |

**And this is why prefill produces the first token.** Prefill has to run the
model over the prompt anyway; the output of that run *is* the first token's
probability distribution. Getting it is free. So a request with
`max_tokens = n` needs one prefill plus `n - 1` decodes — the single most
commonly missed detail in the whole exercise.

Real systems report these as two separate metrics: **TTFT** (time to first
token — your prefill latency) and **TPOT/ITL** (time per output token — your
decode latency). Users feel them differently, which is why a scheduler is
allowed to trade one against the other.

## Why `max_tokens` is an upper bound, not a prediction

A request stops when the model emits an end-of-sequence token, or when it hits
the caller's `max_tokens` limit. **You cannot know which will happen, or when.**
Predicting output length is an open research problem.

So a scheduler has two options, and levels 3a and 3b are exactly those two:

- **Reserve for the worst case** (`max_tokens`). Safe: an admitted request can
  always run to completion. Wasteful: you hold memory for tokens that may
  never be generated. → **Level 3a.**
- **Reserve for what is needed right now.** Efficient, but you can be wrong,
  and when you are you must take memory back from someone. → **Level 3b.**

Real systems do the second, because the first leaves most of the GPU idle.

## Why memory is the binding constraint

The KV cache is big. For a 13B-parameter model it is on the order of ~1 MB per
token. A single 2,000-token conversation is ~2 GB — on an 80 GB card that is
40 conversations, before the ~26 GB of model weights. And it *grows by one
slot per token generated*, for every request, simultaneously.

So the interesting resource is not FLOPs, it is slots. Hence `kv_capacity`,
hence the `+ 1` in `r.kv + 1`: every decoding request needs one more slot this
step than it holds right now.

The "slots" model is not a simplification invented for the exercise — it is
**PagedAttention**, the core idea of vLLM: manage the KV cache in fixed-size
blocks like operating-system virtual-memory pages, so that memory can be
allocated per block instead of per worst-case-conversation.

## Why preemption exists, and why the victim loses its tokens

If you admit optimistically, you will sometimes reach a step where every
admitted request needs one more slot and there are not enough. Something has
to give. The scheduler **preempts** a victim: drops its KV cache and returns
it to the queue.

Two consequences, both modelled in the exercise:

- **Its computed keys and values are gone.** When it is re-admitted the work
  must be redone. The tokens it already produced are kept as *text*, but they
  are now part of the context that must be re-prefilled — which is why a
  victim's `prefill_remaining` becomes `prompt_len + tokens_generated`, and why
  in the Level 3b trace `B` comes back needing `B:5` rather than `B:4`.
  (This is vLLM's "recompute" preemption. The alternative, "swap", copies the
  cache to CPU memory instead; this exercise models recompute.)
- **It rejoins at its original queue position, not the back.** Otherwise
  eviction would starve exactly the requests that are already suffering, and a
  request could be evicted and re-queued forever.

Victim choice — *most recently admitted* — is also principled: the newest
request has done the least work, so preempting it discards the least. In 3c
priority overrides this, and recency only breaks ties within a tier.

## Why the queue is FIFO and you must stop rather than skip

"Stop at the first request that doesn't fit, don't skip ahead to a smaller,
later one" is a fairness rule. If you always packed the budget greedily, a
long prompt at the head of the queue could be passed over indefinitely by a
stream of short ones behind it — unbounded starvation. Stopping bounds a
request's wait by the queue ahead of it.

The cost is **head-of-line blocking**: throughput you gave up for a latency
guarantee. That trade-off is the entire subject of this exercise, and it is
worth being able to say out loud:

- Level 1 → simple, fair, and deadlocks on a prompt longer than `max_work`.
- Level 2 → chunked prefill fixes the deadlock and stops one huge prefill from
  freezing everyone else's decodes. (Real name: **chunked prefill**, and the
  decode-first ordering is what production servers call
  **decode-priority scheduling** — it protects inter-token latency for users
  who are already streaming.)
- Level 3a → correct memory accounting, poor utilisation.
- Level 3b → good utilisation, at the cost of occasional wasted work.
- Level 3c → service tiers, so the waste lands on the requests that matter least.

## Continuous batching, in one paragraph

The reason a *step* exists at all: GPUs are efficient on big batches, so you
want many requests decoding together. Static batching would make everyone wait
for the slowest request in the batch to finish. **Continuous batching**
(a.k.a. in-flight batching) instead re-decides the batch membership *every
step* — finished requests leave immediately, waiting ones join immediately.
That per-step re-decision is precisely your `plan_step`, called once per step,
with a fresh view.

## Check yourself

1. Why can prefill be done in parallel but decode cannot?
2. A request has `prompt_len=8`, `max_tokens=5`. How many steps of work does
   it need, and how many KV slots at its peak?
3. Why does a preempted request need to re-process more tokens than it did the
   first time?
4. Why is "stop, don't skip" better than greedily packing the budget?

<details>
<summary>Answers</summary>

1. A prompt's tokens all exist before you start, so their keys and values can
   be computed in one parallel pass. Output token *n* does not exist until
   token *n-1* has been generated, so decode is a strictly sequential loop.
2. One prefill (8 units) plus 4 decodes = 5 steps minimum, 12 units of work.
   Peak slots = `8 + 5 - 1 = 12`: 8 context tokens plus the 4 output tokens
   that already exist when the 5th is generated.
3. Its KV cache was discarded, but the tokens it already emitted are part of
   the conversation now. It must re-prefill `prompt_len + tokens_generated`
   tokens to rebuild the cache it lost.
4. It bounds waiting time. Greedy packing gets more throughput but lets a
   large request at the head of the queue be starved indefinitely by smaller
   later arrivals.

</details>
