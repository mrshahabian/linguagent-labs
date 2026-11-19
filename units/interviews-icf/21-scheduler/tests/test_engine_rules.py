"""What the engine rejects, and why.

Read this file as documentation. Each test names one rule from the spec and
shows the plan that breaks it. When you hit a SchedulingError while solving a
level, find the matching test here to see what the engine was protecting.
"""

import pytest

from scheduler import Arrival, Engine, PrefillOp, SchedulingError, StepPlan


def engine(**kwargs):
    defaults = dict(
        arrivals=[Arrival("A", 2, 2), Arrival("B", 4, 2)],
        max_work=10,
    )
    defaults.update(kwargs)
    arrivals = defaults.pop("arrivals")
    return Engine(arrivals, **defaults)


def plan_of(**kwargs):
    """A plan_step that always returns the same fixed plan."""
    return lambda view: StepPlan(**kwargs)


def fails(eng, plan_step, match):
    with pytest.raises(SchedulingError, match=match):
        eng.step(plan_step)


# ----------------------------------------------------- the work budget

def test_work_budget_is_enforced():
    """A prefill of n tokens costs n units; a decode costs 1."""
    fails(
        engine(max_work=3),
        plan_of(prefill=[PrefillOp("A", 2), PrefillOp("B", 4)]),
        "uses 6 units of work but max_work is 3",
    )


def test_decodes_count_against_the_same_budget():
    eng = engine(max_work=2)
    eng.step(plan_of(prefill=[PrefillOp("A", 2)]))  # admits A
    fails(
        eng,
        plan_of(decode=["A"], prefill=[PrefillOp("B", 4)]),
        "uses 5 units of work but max_work is 2",
    )


# ----------------------------------------------------- prefill legality

def test_prefill_must_cover_the_whole_context_when_chunking_is_off():
    """Levels 1, 3a, 3b and 3c: a prefill either happens in full or not at all."""
    fails(
        engine(),
        plan_of(prefill=[PrefillOp("A", 1)]),
        "chunking is off",
    )


def test_chunked_prefill_may_take_part_of_it():
    eng = engine(chunked=True)
    record = eng.step(plan_of(prefill=[PrefillOp("B", 3)]))
    assert record.prefill == ("B:3",)


def test_prefill_cannot_exceed_what_remains():
    fails(
        engine(chunked=True),
        plan_of(prefill=[PrefillOp("A", 99)]),
        "is 99 tokens but only 2 remain",
    )


def test_cannot_prefill_a_fully_prefilled_request():
    eng = engine()
    eng.step(plan_of(prefill=[PrefillOp("A", 2)]))
    fails(
        eng,
        plan_of(prefill=[PrefillOp("A", 2)]),
        "already fully prefilled",
    )


# ----------------------------------------------------- decode legality

def test_cannot_decode_a_waiting_request():
    """A request must be admitted - i.e. prefilled - before it can decode."""
    fails(engine(), plan_of(decode=["A"]), "it is not admitted")


def test_cannot_decode_a_partially_prefilled_request():
    eng = engine(chunked=True)
    eng.step(plan_of(prefill=[PrefillOp("B", 1)]))
    fails(eng, plan_of(decode=["B"]), "prefill tokens outstanding")


def test_ids_may_not_repeat():
    fails(
        engine(),
        plan_of(prefill=[PrefillOp("A", 2), PrefillOp("A", 2)]),
        "appears twice in prefill",
    )


# ----------------------------------------------------- preemption legality

def test_cannot_preempt_a_waiting_request():
    """Only admitted requests hold memory, so only they can be evicted."""
    fails(
        engine(kv_capacity=20),
        plan_of(preempt=["A"], prefill=[PrefillOp("B", 4)]),
        "only admitted requests can be preempted",
    )


def test_a_victim_does_no_work_on_the_step_it_is_evicted():
    """This is the rule that makes eviction refund a unit of work budget."""
    eng = engine(kv_capacity=20)
    eng.step(plan_of(prefill=[PrefillOp("A", 2)]))
    fails(
        eng,
        plan_of(preempt=["A"], decode=["A"]),
        "in both preempt and decode",
    )


# ----------------------------------------------------- memory

def test_kv_capacity_is_enforced():
    fails(
        engine(kv_capacity=4),
        plan_of(prefill=[PrefillOp("A", 2), PrefillOp("B", 4)]),
        "needs 6 KV slots but kv_capacity is 4",
    )


def test_a_decode_grows_its_request_by_one_slot():
    """4 slots held + 1 decode = 5 needed, even though nothing was prefilled."""
    eng = engine(kv_capacity=5, arrivals=[Arrival("A", 4, 3)])
    eng.step(plan_of(prefill=[PrefillOp("A", 4)]))
    eng.step(plan_of(decode=["A"]))
    fails(eng, plan_of(decode=["A"]), "needs 6 KV slots but kv_capacity is 5")


def test_a_finishing_request_still_holds_its_slots_during_the_step():
    """Memory is reclaimed *after* the step, so B cannot move into A's slots on
    the very step A finishes. A grows 3 -> 4 as it generates its last token,
    which leaves nothing for B."""
    eng = engine(kv_capacity=4, arrivals=[Arrival("A", 3, 2), Arrival("B", 1, 1)])
    eng.step(plan_of(prefill=[PrefillOp("A", 3)]))
    fails(
        eng,
        plan_of(decode=["A"], prefill=[PrefillOp("B", 1)]),
        "needs 5 KV slots but kv_capacity is 4",
    )


def test_a_preempted_victim_frees_its_slots_immediately():
    """The mirror image: a victim does no work, so its memory is usable now."""
    eng = engine(kv_capacity=5, arrivals=[Arrival("A", 4, 3), Arrival("B", 4, 1)])
    eng.step(plan_of(prefill=[PrefillOp("A", 4)]))
    record = eng.step(plan_of(preempt=["A"], prefill=[PrefillOp("B", 4)]))
    assert record.preempt == ("A",)
    assert record.prefill == ("B:4",)


# ----------------------------------------------------- progress

def test_an_idle_plan_is_rejected():
    """The error you will see most often while developing: your plan came back
    empty while there was still work to do."""
    fails(
        engine(),
        plan_of(),
        "no decode or prefill scheduled, but requests are waiting or admitted",
    )


def test_an_empty_plan_is_fine_once_everything_is_done():
    eng = Engine([Arrival("A", 2, 1)], max_work=10)
    eng.run(plan_of(prefill=[PrefillOp("A", 2)]))
    assert eng.done()


# ----------------------------------------------------- token bookkeeping

def test_prefill_produces_the_first_token_and_decodes_produce_the_rest():
    """max_tokens = n costs one prefill plus n - 1 decodes."""
    eng = Engine([Arrival("A", 2, 3)], max_work=10)
    first = eng.step(plan_of(prefill=[PrefillOp("A", 2)]))
    second = eng.step(plan_of(decode=["A"]))
    third = eng.step(plan_of(decode=["A"]))
    assert first.finished == ()
    assert second.finished == ()
    assert third.finished == ("A",)


def test_a_preempted_request_must_re_prefill_its_generated_tokens():
    """Saved output tokens become context: prefill_remaining grows to
    prompt_len + tokens generated so far."""
    eng = Engine(
        [Arrival("A", 2, 5), Arrival("B", 1, 1)], max_work=10, kv_capacity=10
    )
    eng.step(plan_of(prefill=[PrefillOp("A", 2)]))  # generated 1
    eng.step(plan_of(decode=["A"]))  # generated 2
    eng.step(plan_of(preempt=["A"], prefill=[PrefillOp("B", 1)]))

    # The engine put A back in waiting with 2 + 2 = 4 tokens of context.
    victim = eng.view().waiting[0]
    assert victim.id == "A"
    assert victim.prefill_remaining == 4
    assert victim.kv == 0


def test_a_preempted_request_rejoins_at_its_arrival_position():
    """Not at the back of the queue - that is what stops eviction starving it."""
    eng = Engine(
        [Arrival("OLD", 2, 5), Arrival("NEW", 2, 5, at=1)],
        max_work=10,
        kv_capacity=10,
    )
    eng.step(plan_of(prefill=[PrefillOp("OLD", 2)]))
    eng.step(plan_of(preempt=["OLD"], prefill=[PrefillOp("NEW", 2)]))
    assert [r.id for r in eng.view().waiting] == ["OLD"]

    eng.step(plan_of(decode=["NEW"]))
    assert [r.id for r in eng.view().waiting] == ["OLD"]


# ----------------------------------------------------- the view itself

def test_the_view_is_a_snapshot_taken_before_your_plan_runs():
    """A request you admit this step is still in view.waiting; nothing you do
    updates the view. Counting your own consumption is your job."""
    seen = []

    def plan_step(view):
        seen.append(([r.id for r in view.waiting], [r.id for r in view.admitted]))
        if view.waiting:
            return StepPlan(
                prefill=[PrefillOp(r.id, r.prefill_remaining) for r in view.waiting]
            )
        return StepPlan(decode=[r.id for r in view.admitted])

    Engine([Arrival("A", 2, 2)], max_work=10).run(plan_step)

    # Step 0 admitted A - but the step-0 view still showed it as waiting.
    assert seen[0] == (["A"], [])
    assert seen[1] == ([], ["A"])


def test_the_view_is_immutable():
    eng = Engine([Arrival("A", 2, 2)], max_work=10)
    view = eng.view()
    with pytest.raises(Exception):
        view.waiting[0].kv = 99
