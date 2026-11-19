"""Tests for 61 - An LRU cache with expiry."""

import importlib
import os
import unittest

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


def make_clock(initial_time: float = 0.0):
    """Helper function to create a stateful mock clock without using a class."""
    current = [initial_time]
    def clock() -> float:
        return current[0]
    def advance(seconds: float):
        current[0] += seconds
    clock.advance = advance
    return clock


class TestLRUCache(unittest.TestCase):
    def test_basic_put_get(self):
        cache = m.LRUCache(capacity=3)
        cache.put("a", 1)
        cache.put("b", 2)
        self.assertEqual(cache.get("a"), 1)
        self.assertEqual(cache.get("b"), 2)
        self.assertIsNone(cache.get("c"))

    def test_eviction_at_capacity(self):
        cache = m.LRUCache(capacity=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)  # Should evict "a"

        self.assertIsNone(cache.get("a"))
        self.assertEqual(cache.get("b"), 2)
        self.assertEqual(cache.get("c"), 3)

    def test_get_updates_recency(self):
        cache = m.LRUCache(capacity=2)
        cache.put("a", 1)
        cache.put("b", 2)
        
        # Access "a", making "b" the least recently used
        self.assertEqual(cache.get("a"), 1)
        
        cache.put("c", 3)  # Should evict "b"
        
        self.assertEqual(cache.get("a"), 1)
        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.get("c"), 3)

    def test_put_updates_existing_key_and_recency(self):
        cache = m.LRUCache(capacity=2)
        cache.put("a", 1)
        cache.put("b", 2)
        
        # Update "a", making "b" the least recently used
        cache.put("a", 10)
        
        cache.put("c", 3)  # Should evict "b"
        
        self.assertEqual(cache.get("a"), 10)
        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.get("c"), 3)

    def test_invalid_capacity(self):
        with self.assertRaises(ValueError):
            m.LRUCache(capacity=0)
        with self.assertRaises(ValueError):
            m.LRUCache(capacity=-5)

    def test_len_returns_correct_count(self):
        cache = m.LRUCache(capacity=5)
        self.assertEqual(len(cache), 0)
        cache.put("a", 1)
        cache.put("b", 2)
        self.assertEqual(len(cache), 2)


class TestTTLCache(unittest.TestCase):
    def test_lazy_expiry(self):
        clock = make_clock(100.0)
        cache = m.TTLCache(capacity=2, ttl=10.0, clock=clock)
        
        cache.put("a", 1)
        clock.advance(5.0)
        self.assertEqual(cache.get("a"), 1)  # 5s elapsed, not expired
        
        clock.advance(6.0)
        self.assertIsNone(cache.get("a"))  # 11s elapsed, expired!

    def test_purge_expired(self):
        clock = make_clock(100.0)
        cache = m.TTLCache(capacity=3, ttl=10.0, clock=clock)
        
        cache.put("a", 1)
        clock.advance(5.0)
        cache.put("b", 2)
        
        clock.advance(6.0)  # Total 11s since "a", 6s since "b"
        # "a" is expired, "b" is not
        
        cache.purge_expired()
        
        # "a" should be completely gone from memory
        self.assertEqual(len(cache), 1)
        self.assertIsNone(cache.get("a"))
        self.assertEqual(cache.get("b"), 2)

    def test_len_excludes_expired_items(self):
        clock = make_clock(100.0)
        cache = m.TTLCache(capacity=3, ttl=5.0, clock=clock)
        
        cache.put("a", 1)
        cache.put("b", 2)
        self.assertEqual(len(cache), 2)
        
        clock.advance(6.0)
        self.assertEqual(len(cache), 0)  # Both expired

    def test_put_resets_ttl(self):
        clock = make_clock(100.0)
        cache = m.TTLCache(capacity=2, ttl=5.0, clock=clock)
        
        cache.put("a", 1)
        clock.advance(4.0)
        cache.put("a", 10)  # Reset TTL
        
        clock.advance(2.0)
        self.assertEqual(cache.get("a"), 10)  # Not expired yet because of reset


class TestStatsCache(unittest.TestCase):
    def test_hits_and_misses_basic(self):
        cache = m.StatsCache(capacity=2)
        self.assertEqual(cache.hits, 0)
        self.assertEqual(cache.misses, 0)
        
        cache.put("a", 1)
        cache.get("a")  # Hit
        cache.get("b")  # Miss
        
        self.assertEqual(cache.hits, 1)
        self.assertEqual(cache.misses, 1)

    def test_expired_item_counts_as_miss(self):
        clock = make_clock(100.0)
        cache = m.StatsCache(capacity=2, ttl=5.0, clock=clock)
        
        cache.put("a", 1)
        clock.advance(6.0)
        
        val = cache.get("a")  # Miss because it expired
        self.assertIsNone(val)
        self.assertEqual(cache.hits, 0)
        self.assertEqual(cache.misses, 1)

    def test_stats_accumulation(self):
        cache = m.StatsCache(capacity=2)
        cache.put("a", 1)
        cache.get("a")  # Hit 1
        cache.get("a")  # Hit 2
        cache.get("nonexistent")  # Miss 1
        cache.put("b", 2)
        cache.put("c", 3)  # Evicts "a"
        cache.get("a")  # Miss 2
        
        self.assertEqual(cache.hits, 2)
        self.assertEqual(cache.misses, 2)

    def test_no_ttl_by_default(self):
        clock = make_clock(100.0)
        cache = m.StatsCache(capacity=2, clock=clock)
        
        cache.put("a", 1)
        clock.advance(10000.0)
        self.assertEqual(cache.get("a"), 1)  # Never expires


class TestMemoize(unittest.TestCase):
    def test_memoize_caches_results(self):
        call_count = 0
        
        @m.memoize(capacity=2)
        def add(x, y):
            nonlocal call_count
            call_count += 1
            return x + y

        self.assertEqual(add(1, 2), 3)
        self.assertEqual(add(1, 2), 3)  # Cached
        self.assertEqual(call_count, 1)
        
        self.assertEqual(add(2, 3), 5)
        self.assertEqual(call_count, 2)

    def test_memoize_respects_ttl(self):
        clock = make_clock(100.0)
        call_count = 0
        
        @m.memoize(capacity=2, ttl=10.0, clock=clock)
        def square(n):
            nonlocal call_count
            call_count += 1
            return n * n

        self.assertEqual(square(4), 16)
        clock.advance(5.0)
        self.assertEqual(square(4), 16)  # Cached
        self.assertEqual(call_count, 1)
        
        clock.advance(6.0)  # Expired
        self.assertEqual(square(4), 16)  # Re-computed
        self.assertEqual(call_count, 2)

    def test_memoize_exposes_cache_instance(self):
        @m.memoize(capacity=5)
        def greet(name):
            return f"Hello, {name}"

        self.assertTrue(hasattr(greet, "cache"))
        self.assertIsInstance(greet.cache, m.StatsCache)
        
        greet("Alice")
        greet("Bob")
        greet("Alice")
        
        self.assertEqual(greet.cache.hits, 1)
        self.assertEqual(greet.cache.misses, 2)

    def test_memoize_handles_kwargs(self):
        call_count = 0

        @m.memoize(capacity=2)
        def power(base, exponent=2):
            nonlocal call_count
            call_count += 1
            return base ** exponent

        self.assertEqual(power(3, exponent=3), 27)
        # Order of kwargs should not affect cache hit
        self.assertEqual(power(3, exponent=3), 27)
        self.assertEqual(call_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
