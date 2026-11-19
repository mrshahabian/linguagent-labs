import unittest

from registry import Registry


class TestLevel1(unittest.TestCase):
    def test_publish_new_returns_true(self):
        r = Registry()
        self.assertTrue(r.publish("alpha", 100))

    def test_publish_existing_returns_false_and_overwrites(self):
        r = Registry()
        r.publish("alpha", 100)
        self.assertFalse(r.publish("alpha", 250))
        self.assertEqual(r.get("alpha"), 250)

    def test_get_returns_size(self):
        r = Registry()
        r.publish("alpha", 100)
        r.publish("beta", 7)
        self.assertEqual(r.get("alpha"), 100)
        self.assertEqual(r.get("beta"), 7)

    def test_get_missing_returns_none(self):
        r = Registry()
        self.assertIsNone(r.get("nope"))

    def test_delete_existing_returns_true(self):
        r = Registry()
        r.publish("alpha", 100)
        self.assertTrue(r.delete("alpha"))
        self.assertIsNone(r.get("alpha"))

    def test_delete_missing_returns_false(self):
        r = Registry()
        self.assertFalse(r.delete("alpha"))
        r.publish("alpha", 100)
        r.delete("alpha")
        self.assertFalse(r.delete("alpha"))

    def test_delete_then_republish_returns_true(self):
        r = Registry()
        r.publish("alpha", 100)
        r.delete("alpha")
        self.assertTrue(r.publish("alpha", 5))


if __name__ == "__main__":
    unittest.main()
