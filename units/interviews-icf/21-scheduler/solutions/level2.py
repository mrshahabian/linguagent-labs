"""Level 2 - Chunked prefill.

Level 1 could not handle a prompt longer than max_work: the request never fit,
so it never started. Now prefill work may be split across steps.

Priority order within a step:
  1. Decodes (still oldest first; this avoids having a large prefill stall
     other requests for many steps).
  2. Already-started prefills (we prefer to complete a partial prefill over
     starting a new one).
  3. Then start waiting prefills, oldest first, while the budget allows. A
     prefill too big for the remaining budget is chunked to fill it exactly.

The one new idea: `min(budget, request.prefill_remaining)` - take as much of
the prefill as the budget allows instead of skipping it.
"""

from scheduler.api import EngineView, PrefillOp, StepPlan


def plan_step(view: EngineView) -> StepPlan:
    plan = StepPlan()
    budget = view.max_work

    # An admitted request is either decoding (prefill done) or still being
    # prefilled in chunks. Level 1 never had to tell these apart.
    ready = [r for r in view.admitted if r.prefill_remaining == 0]
    started = [r for r in view.admitted if r.prefill_remaining > 0]

    for request in ready:
        plan.decode.append(request.id)
        budget -= 1

    for request in started:
        if budget <= 0:
            break
        chunk = min(budget, request.prefill_remaining)
        plan.prefill.append(PrefillOp(id=request.id, tokens=chunk))
        budget -= chunk

    for request in view.waiting:
        if budget <= 0:
            break
        chunk = min(budget, request.prefill_remaining)
        plan.prefill.append(PrefillOp(id=request.id, tokens=chunk))
        budget -= chunk

    return plan
