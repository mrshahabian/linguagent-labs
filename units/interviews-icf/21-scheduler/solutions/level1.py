"""Level 1 - Basic scheduling.

Rules:
  * Schedule a decode for every admitted request, oldest first (the order of
    view.admitted).
  * Then, in arrival order, schedule each waiting prefill while the total work
    this step stays at most view.max_work units.
  * Stop at the first waiting prefill that doesn't fit - don't skip ahead to a
    smaller, later one.

Costs: a prefill of n tokens costs n units; a decode costs 1 unit.
"""

from scheduler.api import EngineView, PrefillOp, StepPlan


def plan_step(view: EngineView) -> StepPlan:
    plan = StepPlan()
    work = 0

    # Decode every admitted request. view.admitted is already oldest-first,
    # so no sorting is needed.
    for request in view.admitted:
        plan.decode.append(request.id)
        work += 1

    # Then start waiting prefills in arrival order.
    for request in view.waiting:
        cost = request.prefill_remaining
        if work + cost > view.max_work:
            # `break`, not `continue`: fairness says stop at the first one that
            # doesn't fit rather than skipping ahead to a smaller, later one.
            break
        plan.prefill.append(PrefillOp(id=request.id, tokens=cost))
        work += cost

    return plan
