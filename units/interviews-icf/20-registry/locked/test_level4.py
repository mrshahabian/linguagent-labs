import unittest

from registry import Registry


class TestLevel4(unittest.TestCase):
    def test_backup_returns_live_count(self):
        r = Registry()
        r.publish_at(0, "a", 10)
        r.publish_at(0, "b", 20, 5)
        self.assertEqual(r.backup(3), 2)
        self.assertEqual(r.backup(6), 1)

    def test_backup_of_empty_registry(self):
        r = Registry()
        self.assertEqual(r.backup(0), 0)
        r.publish_at(1, "a", 10)
        r.restore(2, 0)
        self.assertIsNone(r.get_at(3, "a"))

    def test_restore_brings_back_deleted_package(self):
        r = Registry()
        r.publish_at(0, "a", 10)
        r.backup(1)
        r.delete_at(2, "a")
        self.assertIsNone(r.get_at(3, "a"))
        r.restore(4, 1)
        self.assertEqual(r.get_at(5, "a"), 10)

    def test_restore_recomputes_remaining_lifetime(self):
        r = Registry()
        r.publish_at(0, "a", 10, 10)
        r.backup(4)
        r.delete_at(5, "a")
        r.restore(100, 4)
        self.assertEqual(r.get_at(100, "a"), 10)
        self.assertEqual(r.get_at(105, "a"), 10)
        self.assertIsNone(r.get_at(106, "a"))

    def test_restore_preserves_never_expiring(self):
        r = Registry()
        r.publish_at(0, "a", 10)
        r.backup(1)
        r.restore(50, 1)
        self.assertEqual(r.get_at(10 ** 6, "a"), 10)

    def test_restore_discards_packages_published_after_backup(self):
        r = Registry()
        r.publish_at(0, "a", 10)
        r.backup(1)
        r.publish_at(2, "b", 20)
        r.restore(3, 1)
        self.assertEqual(r.get_at(4, "a"), 10)
        self.assertIsNone(r.get_at(4, "b"))

    def test_restore_reverts_size(self):
        r = Registry()
        r.publish_at(0, "a", 10)
        r.backup(1)
        r.publish_at(2, "a", 999)
        r.restore(3, 1)
        self.assertEqual(r.get_at(4, "a"), 10)

    def test_backup_excludes_expired(self):
        r = Registry()
        r.publish_at(0, "a", 10, 3)
        self.assertEqual(r.backup(5), 0)
        r.restore(6, 5)
        self.assertIsNone(r.get_at(6, "a"))

    def test_multiple_backups_are_independent(self):
        r = Registry()
        r.publish_at(0, "a", 10)
        r.backup(1)
        r.publish_at(2, "b", 20)
        r.backup(3)
        r.restore(4, 1)
        self.assertIsNone(r.get_at(4, "b"))
        r.restore(5, 3)
        self.assertEqual(r.get_at(5, "b"), 20)
        self.assertEqual(r.get_at(5, "a"), 10)

    def test_restore_does_not_consume_backup(self):
        r = Registry()
        r.publish_at(0, "a", 10)
        r.backup(1)
        r.restore(2, 1)
        r.delete_at(3, "a")
        r.restore(4, 1)
        self.assertEqual(r.get_at(5, "a"), 10)


if __name__ == "__main__":
    unittest.main()
