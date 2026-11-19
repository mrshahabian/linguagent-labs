# 61 — An LRU cache with expiry

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/interview-drills/61-lru-ttl-cache/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/interview-drills/lru-ttl-cache?utm_source=github&utm_medium=labs)

*~45 minutes · 18 tests · Interview*

In high-throughput systems, caching is the primary tool to reduce latency and save database or API costs. However, memory is finite. A production-grade cache must solve two problems: how to discard items when the cache is full (**eviction**), and how to discard items when they become too old to be trusted (**expiry**).

The standard solution for eviction is the **Least Recently Used (LRU)** policy: when capacity is reached, discard the item that hasn't been accessed for the longest time. For expiry, we use a **Time-To-Live (TTL)**. Combining these two patterns is a classic, high-frequency systems design and coding interview problem that tests your understanding of data structures, time complexity, and clean API design.

In this unit, you will build a robust, production-ready LRU cache with TTL expiry, hit/miss tracking, and a reusable memoization decorator. You will learn how to achieve $O(1)$ time complexity for all core operations using Python's standard library, and how to write deterministic tests for time-dependent code using dependency injection.

## What you'll build

| Function / class | You'll implement | Shows up in real code as |
|---|---|---|
| `LRUCache` | Bounded cache with $O(1)$ `get` and `put` | In-memory caching layers, session stores |
| `TTLCache` | LRU cache with Time-To-Live (TTL) and lazy expiry | Expiring API tokens, temporary database query caches |
| `StatsCache` | LRU + TTL cache with hit/miss statistics | Cache performance monitoring and tuning dashboards |
| `memoize` | A decorator that wraps any function with a `StatsCache` | Optimizing expensive computations or external API calls |

---

## 1. The Core LRU Cache

An LRU cache requires two operations to run in $O(1)$ time:
1. `get(key)`: Look up a value by key and mark it as the most recently used.
2. `put(key, value)`: Insert or update a key-value pair. If the cache exceeds its capacity, evict the least recently used item.

To achieve $O(1)$ lookup, we need a hash map (a Python `dict`). To achieve $O(1)$ ordering and eviction, we need a doubly-linked list. In Python, `collections.OrderedDict` combines these two structures perfectly. It maintains keys in insertion order, and provides a `move_to_end(key)` method to move an accessed key to the end (marking it as most recently used) in $O(1)$ time.

```python
from collections import OrderedDict

cache = OrderedDict()
cache["a"] = 1
cache["b"] = 2
# Move "a" to the end to make it the most recently used
cache.move_to_end("a")
# Pop the first item (the least recently used)
least_used = cache.popitem(last=False)
```

**Pitfall:** A common mistake is forgetting to update the recency of an item during a `put` operation when the key already exists. If a key is updated with a new value, it must also be moved to the end of the access order.

**In `exercises.py`:** Implement the core `LRUCache` initialization, `get`, `put`, and `__len__` methods.

**Test it:** `python3 -m unittest test_exercises.TestLRUCache -v`

---

## 2. Adding TTL and Lazy Expiry

In real-world systems, data changes. A cache that keeps stale data forever is dangerous. We introduce a **Time-To-Live (TTL)**: a duration in seconds after which a cached item is considered expired and must not be returned.

Checking and deleting expired items on a background thread is expensive and introduces concurrency bugs. Instead, we use **lazy expiry**: we store a timestamp alongside each value. When a key is requested via `get`, we check if it has expired. If it has, we delete it on the fly and act as if the key was never there.

To make our cache completely testable and deterministic, we avoid calling `time.time()` directly inside our class. Instead, we inject a `clock` callable (which defaults to `time.time`). This allows our tests to simulate the passage of time precisely without sleeping.

```python
class TTLCache:
    def __init__(self, capacity, ttl, clock=time.time):
        self.capacity = capacity
        self.ttl = ttl
        self.clock = clock
        self._cache = OrderedDict()  # Stores (value, timestamp)
```

We also need an explicit `purge_expired()` method to proactively clean up memory when the cache isn't being read from frequently.

**Pitfall:** When checking if an item is expired, ensure you use `>=` rather than `>` to prevent edge-case race conditions where an item with a TTL of 0 or a sub-millisecond lifespan is kept alive. Also, remember that `__len__` should only count non-expired items!

**In `exercises.py`:** Implement `TTLCache` with TTL support, lazy expiry in `get`, updated `__len__` to filter out expired items, and the `purge_expired` method.

**Test it:** `python3 -m unittest test_exercises.TestTTLCache -v`

---

## 3. Cache Statistics

To monitor cache efficiency, we track two metrics:
- **Hits**: The number of times a requested key was found in the cache and was *not* expired.
- **Misses**: The number of times a requested key was either not in the cache, or was in the cache but had already expired.

These statistics are critical for tuning cache capacities and TTLs in production. 

```python
# Example of tracking stats
cache = StatsCache(capacity=2)
cache.put("x", 100)
cache.get("x")  # Hit! hits becomes 1
cache.get("y")  # Miss! misses becomes 1
```

**Pitfall:** If a key is present in the cache but has expired, a `get` call must increment `misses`, *not* `hits`, and then delete the expired key. A common bug is incrementing `hits` before checking the timestamp.

**In `exercises.py`:** Implement the `StatsCache` class, which combines LRU eviction, optional TTL expiry, and hit/miss tracking.

**Test it:** `python3 -m unittest test_exercises.TestStatsCache -v`

---

## 4. The Memoization Decorator

Writing a cache is useful, but applying it manually to every function call is tedious. We can wrap our `StatsCache` in a reusable Python decorator: `@memoize`.

This decorator should accept `capacity`, `ttl`, and an optional `clock` argument. It wraps any function, converting its arguments into a cache key. Since function arguments can include keyword arguments, we must serialize them into a hashable format (like a sorted tuple of key-value pairs) to use them as dictionary keys.

```python
def make_key(args, kwargs):
    # Convert kwargs dict to a sorted tuple of pairs to make it hashable
    return (args, tuple(sorted(kwargs.items())))
```

Additionally, the decorator should expose the underlying `StatsCache` instance as an attribute on the wrapped function (e.g., `func.cache`). This allows callers to inspect hits/misses or manually purge expired items.

**Pitfall:** Python decorators must use `functools.wraps` to preserve the original function's docstring, name, and signature. Without it, debugging and auto-documentation tools will break.

**In `exercises.py`:** Implement the `memoize` decorator.

**Test it:** `python3 -m unittest test_exercises.TestMemoize -v`

---

## The exercises

| Class / Function | What it teaches |
|---|---|
| `LRUCache` | Bounded eviction and OrderedDict manipulation |
| `TTLCache` | Lazy expiry and dependency-injected clocks |
| `StatsCache` | Hit/miss tracking on top of eviction and expiry |
| `memoize` | Decorators with arguments and argument serialization |

## Check your understanding

1. **Why do we use `OrderedDict` instead of a plain Python `dict`?**
   <details><summary>Answer</summary>While modern Python dicts preserve insertion order, they do not provide an efficient $O(1)$ way to move an existing key to the end of the order (like <code>move_to_end</code>) or pop from the beginning (like <code>popitem(last=False)</code>). <code>OrderedDict</code> explicitly supports these operations in $O(1)$ time.</details>

2. **Why is lazy expiry preferred over a background thread that periodically purges expired items?**
   <details><summary>Answer</summary>Lazy expiry avoids the CPU overhead, lock contention, and complexity of running background threads. It guarantees that expired data is never returned, while only cleaning up memory when the key is accessed or when <code>purge_expired</code> is explicitly called.</details>

3. **Why do we inject a `clock` function instead of calling `time.time()` directly?**
   <details><summary>Answer</summary>Dependency injection makes the class deterministic and testable. In tests, we can pass a mock clock that we control manually, allowing us to simulate hours of elapsed time instantly without using <code>time.sleep()</code>.</details>

4. **What happens to the hit/miss statistics when an expired item is requested?**
   <details><summary>Answer</summary>It counts as a <strong>miss</strong>. The cache lazily deletes the expired item and returns the default value (usually <code>None</code>).</details>

5. **Why do we convert `kwargs` to a sorted tuple of pairs in the memoization key?**
   <details><summary>Answer</summary>Dictionary keys must be hashable. A standard Python <code>dict</code> (like <code>kwargs</code>) is mutable and therefore unhashable. Converting it to a tuple of pairs makes it hashable, and sorting it ensures that <code>func(a=1, b=2)</code> and <code>func(b=2, a=1)</code> resolve to the exact same cache key.</details>

## Interview questions this module answers

- *How do you implement an LRU Cache with $O(1)$ reads and writes?*
- *How do you handle TTL and cache expiration efficiently without background threads?*
- *How do you design a testable class that depends on the system clock?*
- *How do you write a decorator that accepts arguments and maintains state?*
