"""Every worked example from the level specs, asserted step by step.

These are the tests to trust. If your scheduler reproduces these tables you
have understood the level; if it doesn't, `print(render(records))` next to the
table in docs/ shows you the exact step where you diverged.

The `expected` tables below are transcribed directly from the specs, in the
same column order the specs use.
"""

import pytest

from scheduler import Arrival, Engine, render
from scheduler.trace import as_rows

from ._target import load

level1 = load("1")
level2 = load("2")
level3a = load("3a")
level3b = load("3b")
level3c = load("3c")


def check(records, expected, kv=False):
    """Compare a run against a spec table, reporting the first bad step."""
    actual = as_rows(records, kv=kv)
    assert len(actual) == len(expected), (
        f"expected {len(expected)} steps, got {len(actual)}\n\n{render(records, kv=kv)}"
    )
    for actual_row, expected_row in zip(actual, expected):
        assert actual_row == expected_row, (
            f"step {actual_row[0]} diverges\n"
            f"  expected {expected_row}\n"
            f"  actual   {actual_row}\n\n{render(records, kv=kv)}"
        )


# ------------------------------------------------------------------ Level 1

def test_level1_readme_example():
    """max_work=10; A(6 prompt, 3 max_tokens), B(3, 2), C(5, 2) at step 0.

    step  waiting  admitted  prefill    decode  finished
    0     A, B, C  -         A:6, B:3   -       -
    1     C        A, B      C:5        A, B    B
    2     -        A, C      -          A, C    A, C
    """
    engine = Engine(
        [Arrival("A", 6, 3), Arrival("B", 3, 2), Arrival("C", 5, 2)],
        max_work=10,
    )
    check(
        engine.run(level1.plan_step),
        [
            (0, ("A", "B", "C"), (), (), ("A:6", "B:3"), (), ()),
            (1, ("C",), ("A", "B"), (), ("C:5",), ("A", "B"), ("B",)),
            (2, (), ("A", "C"), (), (), ("A", "C"), ("A", "C")),
        ],
    )


def test_level1_stops_rather_than_skipping_ahead():
    """Fairness: B doesn't fit, so C is not pulled forward even though it would."""
    engine = Engine(
        [Arrival("A", 4, 1), Arrival("B", 8, 1), Arrival("C", 1, 1)],
        max_work=10,
    )
    records = engine.run(level1.plan_step)
    assert records[0].prefill == ("A:4",), "C must not jump the queue past B"


def test_level1_prefill_produces_the_first_token():
    """max_tokens=1 means prefill alone finishes the request - no decode."""
    engine = Engine([Arrival("A", 3, 1)], max_work=10)
    records = engine.run(level1.plan_step)
    assert len(records) == 1
    assert records[0].finished == ("A",)
    assert records[0].decode == ()


# ------------------------------------------------------------------ Level 2

def test_level2_readme_example():
    """max_work=8; W(2, 4), X(13, 2), Y(3, 2) at step 0.

    step  waiting  admitted  prefill    decode  finished
    0     W, X, Y  -         W:2, X:6   -       -
    1     Y        W, X      X:7        W       -
    2     Y        W, X      Y:3        W, X    X
    3     -        W, Y      -          W, Y    W, Y
    """
    engine = Engine(
        [Arrival("W", 2, 4), Arrival("X", 13, 2), Arrival("Y", 3, 2)],
        max_work=8,
        chunked=True,
    )
    check(
        engine.run(level2.plan_step),
        [
            (0, ("W", "X", "Y"), (), (), ("W:2", "X:6"), (), ()),
            (1, ("Y",), ("W", "X"), (), ("X:7",), ("W",), ()),
            (2, ("Y",), ("W", "X"), (), ("Y:3",), ("W", "X"), ("X",)),
            (3, (), ("W", "Y"), (), (), ("W", "Y"), ("W", "Y")),
        ],
    )


def test_level2_prompt_longer_than_max_work_still_completes():
    """The whole point of the level: Level 1 would deadlock here."""
    engine = Engine([Arrival("BIG", 25, 1)], max_work=4, chunked=True)
    records = engine.run(level2.plan_step)
    assert [r.prefill for r in records[:6]] == [
        ("BIG:4",), ("BIG:4",), ("BIG:4",), ("BIG:4",), ("BIG:4",), ("BIG:4",)
    ]
    assert records[-1].finished == ("BIG",)


def test_level2_finishes_a_started_prefill_before_starting_a_new_one():
    engine = Engine(
        [Arrival("BIG", 10, 1), Arrival("SMALL", 1, 1)],
        max_work=6,
        chunked=True,
    )
    records = engine.run(level2.plan_step)
    assert records[1].prefill == ("BIG:4", "SMALL:1"), (
        "BIG's outstanding prefill must be topped up before SMALL starts"
    )


# ----------------------------------------------------------------- Level 3a

def test_level3a_readme_example():
    """max_work=10, kv_capacity=6; A(2,2)->peak 3, B(4,2)->peak 5, C(2,2)->peak 3.

    step  waiting  admitted  prefill  decode  finished
    0     A, B, C  -         A:2      -       -
    1     B, C     A         -        A       A
    2     B, C     -         B:4      -       -
    3     C        B         -        B       B
    4     C        -         C:2      -       -
    5     -        C         -        C       C
    """
    engine = Engine(
        [Arrival("A", 2, 2), Arrival("B", 4, 2), Arrival("C", 2, 2)],
        max_work=10,
        kv_capacity=6,
    )
    check(
        engine.run(level3a.plan_step),
        [
            (0, ("A", "B", "C"), (), (), ("A:2",), (), ()),
            (1, ("B", "C"), ("A",), (), (), ("A",), ("A",)),
            (2, ("B", "C"), (), (), ("B:4",), (), ()),
            (3, ("C",), ("B",), (), (), ("B",), ("B",)),
            (4, ("C",), (), (), ("C:2",), (), ()),
            (5, (), ("C",), (), (), ("C",), ("C",)),
        ],
    )


def test_level3a_peak_formula():
    """peak = prompt_len + max_tokens - 1, reached on the final step.

    A(2, 3) peaks at 2 + 3 - 1 = 4, so it just fits a pool of 4 - and the
    engine would reject the plan if the formula were wrong.
    """
    engine = Engine([Arrival("A", 2, 3)], max_work=10, kv_capacity=4)
    records = engine.run(level3a.plan_step)
    assert [r.kv_held for r in records] == [0, 2, 3]


def test_level3a_never_skips_ahead_to_a_smaller_request():
    """B is too big at peak; tiny C must still wait behind it."""
    engine = Engine(
        [Arrival("A", 2, 2), Arrival("B", 5, 2), Arrival("C", 1, 1)],
        max_work=20,
        kv_capacity=6,
    )
    records = engine.run(level3a.plan_step)
    assert records[0].prefill == ("A:2",)


# ----------------------------------------------------------------- Level 3b

def test_level3b_readme_example():
    """max_work=10, kv_capacity=6; A(2,2), B(4,2), C(2,2). Same requests as 3a,
    but finishing at step 3 instead of step 5.

    step  waiting  admitted  kv held  preempt  prefill    decode  finished
    0     A, B, C  -         0        -        A:2, B:4   -       -
    1     C        A, B      6        B        C:2        A       A
    2     B        C         2        -        -          C       C
    3     B        -         0        -        B:5        -       B
    """
    engine = Engine(
        [Arrival("A", 2, 2), Arrival("B", 4, 2), Arrival("C", 2, 2)],
        max_work=10,
        kv_capacity=6,
    )
    check(
        engine.run(level3b.plan_step),
        [
            (0, ("A", "B", "C"), (), 0, (), ("A:2", "B:4"), (), ()),
            (1, ("C",), ("A", "B"), 6, ("B",), ("C:2",), ("A",), ("A",)),
            (2, ("B",), ("C",), 2, (), (), ("C",), ("C",)),
            (3, ("B",), (), 0, (), ("B:5",), (), ("B",)),
        ],
        kv=True,
    )


def test_level3b_beats_level3a_on_throughput():
    """Same workload, fewer steps - that is what optimistic admission buys."""
    def steps(plan_step):
        engine = Engine(
            [Arrival("A", 2, 2), Arrival("B", 4, 2), Arrival("C", 2, 2)],
            max_work=10,
            kv_capacity=6,
        )
        return len(engine.run(plan_step))

    assert steps(level3b.plan_step) < steps(level3a.plan_step)


def test_level3b_victim_keeps_its_tokens_and_its_queue_position():
    """A preempted request re-enters waiting ahead of later arrivals, and its
    already-generated tokens become context it must re-prefill."""
    engine = Engine(
        [Arrival("A", 2, 2), Arrival("B", 4, 2), Arrival("C", 2, 2)],
        max_work=10,
        kv_capacity=6,
    )
    records = engine.run(level3b.plan_step)
    assert records[1].preempt == ("B",)
    # B had generated 1 token from its prefill, so it must re-process 4 + 1.
    assert records[3].prefill == ("B:5",)


# ----------------------------------------------------------------- Level 3c

def test_level3c_readme_example():
    """max_work=10, kv_capacity=8.
    LOW(2, 3, priority 0), MID(2, 3, priority 1), HI(4, 1, priority 2, at step 2).
    LOW is listed first, so it is ahead of MID in arrival order.

    step  waiting   admitted  preempt  prefill      decode    finished
    0     LOW, MID  -         -        MID:2, LOW:2 -         -
    1     -         MID, LOW  -        -            MID, LOW  -
    2     HI        MID, LOW  LOW      HI:4         MID       MID, HI
    3     LOW       -         -        LOW:4        -         LOW
    """
    engine = Engine(
        [
            Arrival("LOW", 2, 3, priority=0),
            Arrival("MID", 2, 3, priority=1),
            Arrival("HI", 4, 1, priority=2, at=2),
        ],
        max_work=10,
        kv_capacity=8,
    )
    check(
        engine.run(level3c.plan_step),
        [
            (0, ("LOW", "MID"), (), (), ("MID:2", "LOW:2"), (), ()),
            (1, (), ("MID", "LOW"), (), (), ("MID", "LOW"), ()),
            (2, ("HI",), ("MID", "LOW"), ("LOW",), ("HI:4",), ("MID",), ("MID", "HI")),
            (3, ("LOW",), (), (), ("LOW:4",), (), ("LOW",)),
        ],
    )


def test_level3c_admits_by_priority_not_arrival():
    """MID is prefilled before LOW even though LOW is ahead in queue order."""
    engine = Engine(
        [Arrival("LOW", 2, 3, priority=0), Arrival("MID", 2, 3, priority=1)],
        max_work=10,
        kv_capacity=8,
    )
    records = engine.run(level3c.plan_step)
    assert records[0].prefill == ("MID:2", "LOW:2")


def test_level3c_evicts_the_minimum_needed():
    """Evicting LOW alone suffices for HI, so MID is spared even though it too
    is lower priority."""
    engine = Engine(
        [
            Arrival("LOW", 2, 3, priority=0),
            Arrival("MID", 2, 3, priority=1),
            Arrival("HI", 4, 1, priority=2, at=2),
        ],
        max_work=10,
        kv_capacity=8,
    )
    records = engine.run(level3c.plan_step)
    assert records[2].preempt == ("LOW",)
    assert "MID" in records[2].decode


def test_level3c_ties_within_a_tier_stay_oldest_first():
    engine = Engine(
        [
            Arrival("FIRST", 2, 1, priority=5),
            Arrival("SECOND", 2, 1, priority=5),
        ],
        max_work=10,
        kv_capacity=20,
    )
    records = engine.run(level3c.plan_step)
    assert records[0].prefill == ("FIRST:2", "SECOND:2")


def test_level3c_will_not_evict_equal_or_higher_priority():
    """A candidate can only displace requests of *strictly* lower priority. Here
    nothing is lower, so it simply waits."""
    engine = Engine(
        [
            Arrival("HOLDER", 6, 3, priority=1),
            Arrival("PEER", 6, 1, priority=1, at=1),
        ],
        max_work=20,
        kv_capacity=8,
    )
    records = engine.run(level3c.plan_step)
    assert records[1].preempt == ()
    assert records[1].prefill == ()


def test_level3c_all_or_nothing_eviction():
    """If evicting every *strictly lower priority* request still wouldn't free
    enough, evict no one - don't thrash for nothing.

    At step 2 HUGE needs 10 slots and only 2 are free. PEER holds 5 but is
    equal priority, so it is untouchable; LOW is evictable but frees only
    3 + 1 = 4 of the 8 needed. So nothing is evicted and HUGE simply waits.
    """
    engine = Engine(
        [
            Arrival("PEER", 4, 5, priority=9),
            Arrival("LOW", 2, 5, priority=0),
            Arrival("HUGE", 10, 1, priority=9, at=2),
        ],
        max_work=20,
        kv_capacity=12,
    )
    records = engine.run(level3c.plan_step)
    assert records[2].preempt == (), "LOW alone cannot cover the shortfall"
    assert records[2].decode == ("PEER", "LOW"), "so both keep running"
    assert records[2].prefill == (), "and HUGE is not admitted"
