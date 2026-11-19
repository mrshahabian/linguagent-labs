import unittest

from registry import Registry


class TestLevel5(unittest.TestCase):
    def test_download_increments_and_returns_total(self):
        r = Registry()
        r.publish_at(0, "a", 10)
        self.assertEqual(r.download_at(1, "a"), 1)
        self.assertEqual(r.download_at(2, "a"), 2)
        self.assertEqual(r.download_at(3, "a"), 3)

    def test_download_missing_returns_none(self):
        r = Registry()
        self.assertIsNone(r.download_at(1, "a"))

    def test_download_expired_returns_none_and_records_nothing(self):
        r = Registry()
        r.publish_at(0, "a", 10, 3)
        self.assertIsNone(r.download_at(5, "a"))
        self.assertEqual(r.top_downloaded(5), [])

    def test_downloads_survive_delete_and_republish(self):
        r = Registry()
        r.publish_at(0, "a", 10)
        r.download_at(1, "a")
        r.download_at(2, "a")
        r.delete_at(3, "a")
        r.publish_at(4, "a", 10)
        self.assertEqual(r.download_at(5, "a"), 3)

    def test_top_downloaded_order_and_format(self):
        r = Registry()
        for name in ("a", "b", "c"):
            r.publish_at(0, name, 10)
        r.download_at(1, "a")
        r.download_at(1, "b")
        r.download_at(1, "b")
        r.download_at(1, "c")
        self.assertEqual(r.top_downloaded(3), ["b(2)", "a(1)", "c(1)"])

    def test_top_downloaded_respects_n(self):
        r = Registry()
        for name in ("a", "b", "c"):
            r.publish_at(0, name, 10)
            r.download_at(1, name)
        self.assertEqual(r.top_downloaded(2), ["a(1)", "b(1)"])
        self.assertEqual(r.top_downloaded(0), [])

    def test_top_downloaded_excludes_zero(self):
        r = Registry()
        r.publish_at(0, "a", 10)
        r.publish_at(0, "b", 10)
        r.download_at(1, "a")
        self.assertEqual(r.top_downloaded(5), ["a(1)"])

    def test_top_downloaded_includes_expired_packages(self):
        r = Registry()
        r.publish_at(0, "a", 10, 5)
        r.download_at(1, "a")
        self.assertEqual(r.top_downloaded(5), ["a(1)"])
        self.assertIsNone(r.get_at(6, "a"))
        self.assertEqual(r.top_downloaded(5), ["a(1)"])

    def test_restore_does_not_reset_downloads(self):
        r = Registry()
        r.publish_at(0, "a", 10)
        r.backup(1)
        r.download_at(2, "a")
        r.download_at(2, "a")
        r.restore(3, 1)
        self.assertEqual(r.top_downloaded(1), ["a(2)"])
        self.assertEqual(r.download_at(4, "a"), 3)


if __name__ == "__main__":
    unittest.main()
