import unittest

from registry import Registry


class TestLevel2(unittest.TestCase):
    def test_search_orders_by_size_descending(self):
        r = Registry()
        r.publish("web-core", 300)
        r.publish("web-utils", 500)
        r.publish("db-core", 900)
        self.assertEqual(r.search("web"), ["web-utils(500)", "web-core(300)"])

    def test_search_ties_broken_by_name_ascending(self):
        r = Registry()
        r.publish("web-b", 100)
        r.publish("web-a", 100)
        r.publish("web-c", 100)
        self.assertEqual(r.search("web"), ["web-a(100)", "web-b(100)", "web-c(100)"])

    def test_search_returns_at_most_five(self):
        r = Registry()
        for i in range(8):
            r.publish("web-{}".format(i), (i + 1) * 10)
        self.assertEqual(
            r.search("web"),
            ["web-7(80)", "web-6(70)", "web-5(60)", "web-4(50)", "web-3(40)"],
        )

    def test_search_no_match_returns_empty_list(self):
        r = Registry()
        r.publish("alpha", 10)
        self.assertEqual(r.search("zz"), [])

    def test_search_empty_prefix_matches_everything(self):
        r = Registry()
        r.publish("alpha", 10)
        r.publish("beta", 20)
        self.assertEqual(r.search(""), ["beta(20)", "alpha(10)"])

    def test_search_ignores_deleted(self):
        r = Registry()
        r.publish("web-a", 100)
        r.publish("web-b", 50)
        r.delete("web-a")
        self.assertEqual(r.search("web"), ["web-b(50)"])

    def test_total_size_empty(self):
        r = Registry()
        self.assertEqual(r.total_size(), 0)

    def test_total_size_reflects_updates_and_deletes(self):
        r = Registry()
        r.publish("a", 10)
        r.publish("b", 25)
        self.assertEqual(r.total_size(), 35)
        r.publish("a", 40)
        self.assertEqual(r.total_size(), 65)
        r.delete("b")
        self.assertEqual(r.total_size(), 40)


if __name__ == "__main__":
    unittest.main()
