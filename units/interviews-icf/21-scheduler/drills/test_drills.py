"""Checks for drills.py.

    ./drills/drill          run everything
    ./drills/drill B        run one block, verbose
    ./drills/drill --check  run against solutions.py instead
"""

import importlib
import os
import unittest
from dataclasses import dataclass

d = importlib.import_module(os.environ.get("DRILL_TARGET", "drills"))


@dataclass(frozen=True)
class R:
    """A stand-in for api.Request."""

    id: str
    priority: int = 0
    prefill_remaining: int = 0
    kv: int = 0


class TestBlockA(unittest.TestCase):
    def test_plan_fields_are_independent_lists(self):
        first, second = d.Plan(), d.Plan()
        first.decode.append("A")
        self.assertEqual(first.decode, ["A"])
        self.assertEqual(second.decode, [], "each Plan needs its own list")
        self.assertEqual(second.preempt, [])
        self.assertEqual(second.prefill, [])

    def test_collect_ids(self):
        self.assertEqual(d.collect_ids((R("a"), R("b"))), ["a", "b"])
        self.assertEqual(d.collect_ids(()), [])

    def test_newest_first(self):
        a, b, c = R("a"), R("b"), R("c")
        self.assertEqual(d.newest_first((a, b, c)), [c, b, a])
        self.assertEqual(d.newest_first(()), [])

    def test_split_by_prefill(self):
        a, b, c = R("a", prefill_remaining=0), R("b", prefill_remaining=3), R("c")
        ready, started = d.split_by_prefill((a, b, c))
        self.assertEqual(ready, [a, c])
        self.assertEqual(started, [b])


class TestBlockB(unittest.TestCase):
    def test_by_total_desc(self):
        self.assertEqual(
            d.by_total_desc([("b", 5), ("a", 5), ("c", 9)]),
            [("c", 9), ("a", 5), ("b", 5)],
        )
        self.assertEqual(d.by_total_desc([]), [])

    def test_top_k_formatted(self):
        self.assertEqual(
            d.top_k_formatted({"a": 1, "b": 2, "c": 0}, 5), ["b(2)", "a(1)", "c(0)"]
        )
        self.assertEqual(d.top_k_formatted({"a": 1, "b": 1}, 1), ["a(1)"])
        self.assertEqual(d.top_k_formatted({}, 3), [])

    def test_by_priority_desc_is_stable(self):
        low, mid, other = R("LOW", 0), R("MID", 1), R("OTHER", 1)
        self.assertEqual(
            d.by_priority_desc([low, mid, other]),
            [mid, other, low],
            "MID must stay ahead of OTHER - stability preserves arrival order",
        )

    def test_lowest_priority_newest(self):
        first, second, third = R("a", 1), R("b", 0), R("c", 0)
        # b and c tie on priority; c was admitted later, so c is the victim.
        self.assertEqual(d.lowest_priority_newest((first, second, third)), third)
        self.assertEqual(d.lowest_priority_newest((first,)), first)


class TestBlockC(unittest.TestCase):
    def test_take_while_fits_stops(self):
        self.assertEqual(d.take_while_fits([4, 8, 1], 10), [4])
        self.assertEqual(d.take_while_fits([4, 3, 1], 10), [4, 3, 1])
        self.assertEqual(d.take_while_fits([99], 10), [])

    def test_take_greedily_skips(self):
        self.assertEqual(d.take_greedily([4, 8, 1], 10), [4, 1])

    def test_chunk_to_budget(self):
        self.assertEqual(d.chunk_to_budget([2, 13, 3], 8), [2, 6])
        self.assertEqual(d.chunk_to_budget([2, 3], 8), [2, 3])
        self.assertEqual(d.chunk_to_budget([5], 0), [])

    def test_first_fit_index(self):
        self.assertEqual(d.first_fit_index([4, 3, 9], 10), 2)
        self.assertIsNone(d.first_fit_index([4, 3], 10))
        self.assertEqual(d.first_fit_index([99], 10), 0)


class TestBlockD(unittest.TestCase):
    def test_peak_slots(self):
        self.assertEqual(d.peak_slots(2, 2), 3)
        self.assertEqual(d.peak_slots(7, 4), 10)
        self.assertEqual(d.peak_slots(5, 1), 5, "max_tokens=1 peaks at prompt_len")

    def test_slots_after_decodes(self):
        self.assertEqual(d.slots_after_decodes(2, 0), 2)
        self.assertEqual(d.slots_after_decodes(2, 2), 4)

    def test_growth_need(self):
        self.assertEqual(d.growth_need([2, 4]), 8)
        self.assertEqual(d.growth_need([]), 0)

    def test_evict_until_fits(self):
        self.assertEqual(d.evict_until_fits([2, 4], 6), ([2], [1]))
        self.assertEqual(d.evict_until_fits([2, 4], 8), ([2, 4], []))
        self.assertEqual(d.evict_until_fits([2, 2, 2], 4), ([2], [2, 1]))

    def test_replay_cost(self):
        self.assertEqual(d.replay_cost(4, 1), 5)
        self.assertEqual(d.replay_cost(4, 0), 4)


class TestBlockE(unittest.TestCase):
    def test_evictable_is_strict(self):
        low, peer, high = R("low", 0), R("peer", 2), R("high", 3)
        self.assertEqual(d.evictable([low, peer, high], 2), [low])
        self.assertEqual(d.evictable([peer], 2), [], "equal priority is not evictable")

    def test_enough_to_free(self):
        self.assertTrue(d.enough_to_free([3, 3], 4))
        self.assertFalse(d.enough_to_free([3], 8))
        self.assertFalse(d.enough_to_free([], 1))

    def test_evict_for_shortfall_evicts_as_few_as_possible(self):
        self.assertEqual(d.evict_for_shortfall([3, 3], 4), ([0], 4))
        self.assertEqual(d.evict_for_shortfall([1, 3], 4), ([0, 1], 6))
        self.assertEqual(d.evict_for_shortfall([9], 4), ([0], 10))


class TestBlockF(unittest.TestCase):
    def test_add_and_total(self):
        ledger = d.Ledger()
        self.assertEqual(ledger.add("a", 5), 5)
        self.assertEqual(ledger.add("a", 3), 8)
        self.assertEqual(ledger.total("a"), 8)
        self.assertIsNone(ledger.total("missing"))

    def test_remove(self):
        ledger = d.Ledger()
        ledger.add("a", 1)
        self.assertTrue(ledger.remove("a"))
        self.assertFalse(ledger.remove("a"))
        self.assertIsNone(ledger.total("a"))

    def test_top(self):
        ledger = d.Ledger()
        ledger.add("a", 1)
        ledger.add("b", 5)
        ledger.add("c", 5)
        self.assertEqual(ledger.top(2), ["b(5)", "c(5)"])
        self.assertEqual(ledger.top(9), ["b(5)", "c(5)", "a(1)"])
        self.assertEqual(ledger.top(0), [])

    def test_add_does_not_touch_last_touched(self):
        ledger = d.Ledger()
        ledger.add("a", 1)
        self.assertIsNone(ledger.last_touched("a"))

    def test_add_at_records_the_timestamp(self):
        ledger = d.Ledger()
        self.assertEqual(ledger.add_at(100, "b", 3), 3)
        self.assertEqual(ledger.last_touched("b"), 100)
        ledger.add("b", 2)
        self.assertEqual(ledger.total("b"), 5)
        self.assertEqual(ledger.last_touched("b"), 100, "add must not update it")
        ledger.add_at(200, "b", 1)
        self.assertEqual(ledger.last_touched("b"), 200)


class TestBlockG(unittest.TestCase):
    def test_rebase(self):
        self.assertEqual(
            d.rebase({"a": (1, 10), "b": (2, None)}, 100),
            {"a": (1, 110), "b": (2, None)},
        )
        self.assertEqual(d.rebase({}, 5), {})

    def test_rebase_does_not_mutate(self):
        saved = {"a": (1, 10)}
        d.rebase(saved, 100)
        self.assertEqual(saved, {"a": (1, 10)})

    def test_pyramid(self):
        self.assertEqual(d.pyramid(1), ["*"])
        self.assertEqual(d.pyramid(3), ["  *", " ***", "*****"])
        self.assertEqual(len(d.pyramid(10)), 10)
        self.assertEqual(d.pyramid(10)[-1], "*" * 19)

    def test_format_ops(self):
        self.assertEqual(d.format_ops([("A", 6), ("B", 3)]), "A:6, B:3")
        self.assertEqual(d.format_ops([]), "-")


if __name__ == "__main__":
    unittest.main()
