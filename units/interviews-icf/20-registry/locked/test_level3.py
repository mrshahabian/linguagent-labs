import unittest

from registry import Registry


class TestLevel3(unittest.TestCase):
    def test_ttl_boundary_is_exclusive(self):
        r = Registry()
        self.assertTrue(r.publish_at(10, "alpha", 100, 5))
        self.assertEqual(r.get_at(10, "alpha"), 100)
        self.assertEqual(r.get_at(14, "alpha"), 100)
        self.assertIsNone(r.get_at(15, "alpha"))
        self.assertIsNone(r.get_at(99, "alpha"))

    def test_no_ttl_never_expires(self):
        r = Registry()
        r.publish_at(1, "alpha", 100)
        self.assertEqual(r.get_at(10 ** 6, "alpha"), 100)

    def test_publish_over_expired_returns_true(self):
        r = Registry()
        r.publish_at(0, "alpha", 100, 5)
        self.assertTrue(r.publish_at(10, "alpha", 200, 5))
        self.assertEqual(r.get_at(11, "alpha"), 200)

    def test_publish_over_alive_returns_false_and_resets_lifetime(self):
        r = Registry()
        r.publish_at(0, "alpha", 100, 5)
        self.assertFalse(r.publish_at(3, "alpha", 200, 5))
        self.assertEqual(r.get_at(7, "alpha"), 200)
        self.assertIsNone(r.get_at(8, "alpha"))

    def test_publish_can_remove_ttl(self):
        r = Registry()
        r.publish_at(0, "alpha", 100, 5)
        r.publish_at(2, "alpha", 100)
        self.assertEqual(r.get_at(500, "alpha"), 100)

    def test_delete_at_alive_returns_true(self):
        r = Registry()
        r.publish_at(0, "alpha", 100, 5)
        self.assertTrue(r.delete_at(4, "alpha"))
        self.assertIsNone(r.get_at(4, "alpha"))

    def test_delete_at_expired_returns_false(self):
        r = Registry()
        r.publish_at(0, "alpha", 100, 5)
        self.assertFalse(r.delete_at(5, "alpha"))

    def test_search_at_excludes_expired(self):
        r = Registry()
        r.publish_at(0, "web-a", 100, 5)
        r.publish_at(0, "web-b", 50)
        self.assertEqual(r.search_at(1, "web"), ["web-a(100)", "web-b(50)"])
        self.assertEqual(r.search_at(6, "web"), ["web-b(50)"])

    def test_search_at_still_caps_at_five(self):
        r = Registry()
        for i in range(7):
            r.publish_at(0, "pkg-{}".format(i), (i + 1) * 10)
        self.assertEqual(len(r.search_at(1, "pkg")), 5)

    def test_level_one_and_two_methods_still_work(self):
        r = Registry()
        self.assertTrue(r.publish("alpha", 100))
        self.assertEqual(r.get("alpha"), 100)
        self.assertEqual(r.get_at(999, "alpha"), 100)
        self.assertEqual(r.search("al"), ["alpha(100)"])
        self.assertEqual(r.total_size(), 100)
        self.assertTrue(r.delete("alpha"))

    def test_mixed_timestamped_and_plain_calls(self):
        r = Registry()
        r.publish("alpha", 100)
        self.assertFalse(r.publish_at(0, "alpha", 300))
        self.assertEqual(r.get("alpha"), 300)


if __name__ == "__main__":
    unittest.main()
