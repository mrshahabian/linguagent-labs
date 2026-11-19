"""Level 3c - Priority tiers.

Now that eviction is implemented, we can handle requests of differing
`priority`, where a larger number = more urgent.

  * Victim selection: always the lowest-priority request, ties broken by most
    recently admitted, as before.
  * Preemption for growth: as in Level 3b, if there aren't enough free slots
    for every admitted request to decode this step, preempt (in the victim
    order above) until they fit - before decoding.
  * Decodes: unchanged - in the order of view.admitted. (That is admission
    order, which under priority admission is no longer always arrival order.)
  * Admission order: highest priority first, tiebreaker oldest. view.waiting is
    already in arrival order, so a *stable* sort by descending priority keeps
    oldest-first within a tier. As before, stop at the first candidate - in
    this order - that can't be admitted.
  * Preemption for admission: if the candidate's prefill fits the remaining
    budget but there isn't enough free memory, evict admitted requests of
    *strictly lower* priority than the candidate (victim order above) as needed
    to make room. Only a memory shortfall triggers eviction; a candidate that
    doesn't fit the budget stops admission as usual. Only do this if it
    actually suffices: if evicting every strictly-lower-priority request still
    wouldn't free enough memory, evict no one and stop admitting.

A victim does no decode this step - a plan that both preempts and decodes the
same request is rejected - so each eviction frees its kv slots *plus* the one
slot its decode would have grown into, and refunds the one unit of work budget
its decode would have used (which may let further candidates be admitted this
step).

Append to plan.preempt in the order you evict.
"""

from scheduler.api import KvEngineView, PrefillOp, StepPlan


def plan_step(view: KvEngineView) -> StepPlan:
    plan = StepPlan()

    # Admission rank: position in view.admitted. Higher = more recently
    # admitted = evicted first within a priority tier.
    rank = {request.id: index for index, request in enumerate(view.admitted)}

    def victim_key(request):
        return (request.priority, -rank[request.id])

    # 1. Preempt for growth, lowest priority first.
    #    `active` holds the requests that will still be here after this loop;
    #    the invariant from here on is that every member of `active` has a
    #    decode scheduled, so evicting one frees kv + 1 slots and 1 work unit.
    active = list(view.admitted)
    while active and sum(r.kv + 1 for r in active) > view.kv_capacity:
        victim = min(active, key=victim_key)
        active.remove(victim)
        plan.preempt.append(victim.id)

    # 2. Decode the survivors, in view.admitted order.
    survivors = {r.id for r in active}
    work_budget = view.max_work
    kv_used = 0
    for request in view.admitted:
        if request.id in survivors:
            plan.decode.append(request.id)
            work_budget -= 1
            kv_used += request.kv + 1

    # 3. Admit, highest priority first. sorted() is stable, so requests of
    #    equal priority keep view.waiting's arrival order.
    for candidate in sorted(view.waiting, key=lambda r: -r.priority):
        cost = candidate.prefill_remaining

        # A budget shortfall stops admission outright - eviction cannot buy
        # work, only memory.
        if cost > work_budget:
            break

        shortfall = cost - (view.kv_capacity - kv_used)
        if shortfall > 0:
            pool = sorted(
                (r for r in active if r.priority < candidate.priority),
                key=victim_key,
            )
            # All-or-nothing: if even evicting every strictly-lower-priority
            # request wouldn't free enough, evict no one and stop admitting.
            if sum(r.kv + 1 for r in pool) < shortfall:
                break

            freed = 0
            for victim in pool:
                if freed >= shortfall:
                    break
                plan.decode.remove(victim.id)
                plan.preempt.append(victim.id)
                active.remove(victim)
                work_budget += 1  # refunded: the victim no longer decodes
                freed += victim.kv + 1
                kv_used -= victim.kv + 1

        plan.prefill.append(PrefillOp(id=candidate.id, tokens=cost))
        work_budget -= cost
        kv_used += cost

    return plan
