"""Level 3a - KV-aware admission.

Memory enters the picture. Each admitted request holds KV-cache slots: one per
prefilled context token, plus one per output token it has already decoded.

    peak = prompt_len + max_tokens - 1

(prompt_len slots of context, plus max_tokens - 1 slots for the output tokens
that exist before the final one is generated - the peak is reached on the step
that generates the last token.)

New logic:
  * Decodes first, as before.
  * Prefill the oldest request as before, but admit it only if the *peak* usage
    of every admitted request plus the candidate - each counted at its peak,
    not its current usage - sums to at most kv_capacity.
  * For fairness, stop at the oldest request that doesn't fit; never skip ahead.
  * No chunking here: each prefill completes in one step, as in Level 1.

Reserving at peak is conservative - it wastes throughput, which is exactly the
problem Level 3b solves - but it guarantees every admitted request can run to
completion without ever running out of memory.
"""

from scheduler.api import KvEngineView, PrefillOp, StepPlan


def peak_slots(request) -> int:
    """The most KV slots this request will ever hold, on its very last step."""
    return request.prompt_len + request.max_tokens - 1


def plan_step(view: KvEngineView) -> StepPlan:
    plan = StepPlan()
    work_budget = view.max_work

    for request in view.admitted:
        plan.decode.append(request.id)
        work_budget -= 1

    # Everything already admitted was itself admitted at peak, so the pool is
    # reserved at peak too.
    reserved = sum(peak_slots(r) for r in view.admitted)

    for request in view.waiting:
        cost = request.prefill_remaining
        if cost > work_budget or reserved + peak_slots(request) > view.kv_capacity:
            break
        plan.prefill.append(PrefillOp(id=request.id, tokens=cost))
        work_budget -= cost
        reserved += peak_slots(request)

    return plan
