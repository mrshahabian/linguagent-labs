"""Level 3b - Preemption.

Level 3a's peak reservation guaranteed progress and fairness, but at the cost
of throughput: a request that will *eventually* need 5 slots holds all 5 from
the moment it is admitted, even on its first step when it only uses 2.

So now we admit against *current* memory. That is optimistic: a step can arrive
where the admitted requests cannot all grow. When that happens we preempt.

Preempting a victim makes the engine:
  * discard its KV cache, immediately freeing its memory (usable this step),
  * save its already-generated output tokens - they count towards
    prefill_remaining, so a re-admitted victim must re-process them as context,
  * return it to `waiting` at a position determined by its original arrival,
    not at the back.

The algorithm:
  * While there aren't enough memory slots for all decodes to make progress,
    preempt the most recently admitted request.
  * Decode all remaining requests (now guaranteed to fit).
  * Then admit waiting prefills, oldest first, while the work budget and the
    free slots allow - counting the slot each scheduled decode grows into this
    step - and stop at the first that doesn't fit.

The `+ 1` in `r.kv + 1` is the whole level: a decode grows its request by one
slot, so the memory a decoding request needs this step is one more than it
holds right now.
"""

from scheduler.api import KvEngineView, PrefillOp, StepPlan


def plan_step(view: KvEngineView) -> StepPlan:
    plan = StepPlan()

    # 1. Preempt for growth. view.admitted is in admission order, so the last
    #    element is the most recently admitted - the victim.
    survivors = list(view.admitted)
    while survivors and sum(r.kv + 1 for r in survivors) > view.kv_capacity:
        victim = survivors.pop()
        plan.preempt.append(victim.id)

    # 2. Decode the survivors. Each costs 1 unit of work and grows by 1 slot.
    work_budget = view.max_work
    kv_used = 0
    for request in survivors:
        plan.decode.append(request.id)
        work_budget -= 1
        kv_used += request.kv + 1

    # 3. Admit waiting prefills against what is actually left.
    for request in view.waiting:
        cost = request.prefill_remaining
        if cost > work_budget or kv_used + cost > view.kv_capacity:
            break
        plan.prefill.append(PrefillOp(id=request.id, tokens=cost))
        work_budget -= cost
        kv_used += cost

    return plan
