import unittest

import importlib, os
d = importlib.import_module(os.environ.get("DRILL_TARGET", "drills"))


class TestBlockA(unittest.TestCase):
    def test_word_count(self):
        self.assertEqual(d.word_count(["a", "b", "a"]), {"a": 2, "b": 1})
        self.assertEqual(d.word_count([]), {})

    def test_merge_counts(self):
        a = {"a": 1}
        b = {"a": 2, "b": 5}
        self.assertEqual(d.merge_counts(a, b), {"a": 3, "b": 5})
        self.assertEqual(a, {"a": 1})
        self.assertEqual(b, {"a": 2, "b": 5})

    def test_invert(self):
        self.assertEqual(d.invert({"a": 1, "b": 2, "c": 1}), {1: ["a", "c"], 2: ["b"]})
        self.assertEqual(d.invert({}), {})

    def test_bump(self):
        store = {}
        self.assertEqual(d.bump(store, "x"), 1)
        self.assertEqual(d.bump(store, "x"), 2)
        self.assertEqual(d.bump(store, "x", 5), 7)
        self.assertEqual(store, {"x": 7})


class TestBlockB(unittest.TestCase):
    def test_by_score_desc(self):
        self.assertEqual(
            d.by_score_desc([("b", 5), ("a", 5), ("c", 9)]),
            [("c", 9), ("a", 5), ("b", 5)],
        )
        self.assertEqual(d.by_score_desc([]), [])

    def test_top_k_formatted(self):
        self.assertEqual(d.top_k_formatted({"a": 1, "b": 2, "c": 0}, 5), ["b(2)", "a(1)"])
        self.assertEqual(d.top_k_formatted({"a": 1, "b": 1}, 1), ["a(1)"])
        self.assertEqual(d.top_k_formatted({"a": 1}, 0), [])
        self.assertEqual(d.top_k_formatted({}, 3), [])

    def test_by_length_then_alpha(self):
        self.assertEqual(
            d.by_length_then_alpha(["bbb", "a", "cc", "bb"]),
            ["a", "bb", "cc", "bbb"],
        )

    def test_ranked(self):
        self.assertEqual(d.ranked([("b", 5), ("a", 5)]), ["1. a(5)", "2. b(5)"])
        self.assertEqual(d.ranked([]), [])

    def test_second_largest(self):
        self.assertEqual(d.second_largest([1, 5, 3]), 3)
        self.assertEqual(d.second_largest([4, 4, 2]), 2)
        self.assertIsNone(d.second_largest([7, 7]))
        self.assertIsNone(d.second_largest([]))


class TestBlockC(unittest.TestCase):
    def test_with_prefix(self):
        names = ["web-b", "web-a", "db-a"]
        self.assertEqual(d.with_prefix(names, "web"), ["web-a", "web-b"])
        self.assertEqual(d.with_prefix(names, ""), ["db-a", "web-a", "web-b"])
        self.assertEqual(d.with_prefix(names, "zz"), [])

    def test_group_by_first(self):
        self.assertEqual(
            d.group_by_first(["bat", "ant", "ape", ""]),
            {"b": ["bat"], "a": ["ant", "ape"]},
        )
        self.assertEqual(d.group_by_first([]), {})

    def test_longest_common_prefix(self):
        self.assertEqual(d.longest_common_prefix(["web-a", "web-b"]), "web-")
        self.assertEqual(d.longest_common_prefix(["abc", "xyz"]), "")
        self.assertEqual(d.longest_common_prefix(["solo"]), "solo")
        self.assertEqual(d.longest_common_prefix([]), "")


class TestBlockD(unittest.TestCase):
    def test_is_alive(self):
        self.assertTrue(d.is_alive(10, 5, 10))
        self.assertTrue(d.is_alive(10, 5, 14))
        self.assertFalse(d.is_alive(10, 5, 15))
        self.assertFalse(d.is_alive(10, 5, 9))
        self.assertTrue(d.is_alive(10, None, 10 ** 9))
        self.assertFalse(d.is_alive(10, None, 9))

    def test_remaining(self):
        self.assertIsNone(d.remaining(0, None, 50))
        self.assertEqual(d.remaining(0, 10, 4), 6)
        self.assertEqual(d.remaining(0, 10, 10), 0)
        self.assertEqual(d.remaining(0, 10, 99), 0)

    def test_filter_alive(self):
        recs = {"a": (0, 5), "b": (0, None), "c": (10, 5)}
        self.assertEqual(d.filter_alive(recs, 3), ["a", "b"])
        self.assertEqual(d.filter_alive(recs, 12), ["b", "c"])
        self.assertEqual(d.filter_alive({}, 0), [])

    def test_latest_before(self):
        events = [(5, "five"), (1, "one"), (9, "nine")]
        self.assertEqual(d.latest_before(events, 7), "five")
        self.assertEqual(d.latest_before(events, 9), "nine")
        self.assertIsNone(d.latest_before(events, 0))
        self.assertIsNone(d.latest_before([], 5))


class TestBlockE(unittest.TestCase):
    def test_snapshot_is_independent(self):
        state = {"a": [1, 2], "b": []}
        copy = d.snapshot(state)
        self.assertEqual(copy, state)
        copy["a"].append(3)
        copy["c"] = [9]
        self.assertEqual(state, {"a": [1, 2], "b": []})

    def test_rebase(self):
        saved = {"a": (10, 6), "b": (20, None)}
        self.assertEqual(d.rebase(saved, 100), {"a": (10, 106), "b": (20, None)})
        self.assertEqual(d.rebase({}, 0), {})


class TestBlockF(unittest.TestCase):
    def test_add_and_total(self):
        led = d.Ledger()
        self.assertEqual(led.add("a", 3), 3)
        self.assertEqual(led.add("a", 4), 7)
        self.assertEqual(led.total("a"), 7)
        self.assertIsNone(led.total("zz"))

    def test_remove(self):
        led = d.Ledger()
        led.add("a", 1)
        self.assertTrue(led.remove("a"))
        self.assertFalse(led.remove("a"))
        self.assertIsNone(led.total("a"))

    def test_top(self):
        led = d.Ledger()
        led.add("a", 1)
        led.add("b", 2)
        led.add("c", 1)
        self.assertEqual(led.top(2), ["b(2)", "a(1)"])
        self.assertEqual(led.top(9), ["b(2)", "a(1)", "c(1)"])

    def test_add_at_records_timestamp(self):
        led = d.Ledger()
        led.add_at(5, "a", 2)
        self.assertEqual(led.total("a"), 2)
        self.assertEqual(led.last_touched("a"), 5)
        led.add_at(9, "a", 1)
        self.assertEqual(led.total("a"), 3)
        self.assertEqual(led.last_touched("a"), 9)

    def test_plain_add_leaves_last_touched_alone(self):
        led = d.Ledger()
        led.add("a", 1)
        self.assertIsNone(led.last_touched("a"))
        led.add_at(4, "a", 1)
        led.add("a", 1)
        self.assertEqual(led.last_touched("a"), 4)
        self.assertEqual(led.total("a"), 3)


class TestBlockG(unittest.TestCase):
    def test_format_entry(self):
        self.assertEqual(d.format_entry("a", 5), "a(5)")

    def test_parse_entry(self):
        self.assertEqual(d.parse_entry("web-utils(500)"), ("web-utils", 500))
        self.assertEqual(d.parse_entry("a.b(0)"), ("a.b", 0))

    def test_round_trip(self):
        for name, value in (("x", 1), ("long-name.v2", 4321)):
            self.assertEqual(d.parse_entry(d.format_entry(name, value)), (name, value))

    def test_summarise(self):
        self.assertEqual(d.summarise({"a": 1, "b": 2, "c": 0}), "b(2), a(1)")
        self.assertEqual(d.summarise({}), "")


if __name__ == "__main__":
    unittest.main()
