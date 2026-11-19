"""61 - An LRU cache with expiry

Implement the LRUCache, TTLCache, and StatsCache classes, along with the
memoize decorator, so that the tests in test_exercises.py pass.

Read README.md first; every exercise is covered there.
"""

from collections import OrderedDict
import functools
import time
from typing import Any, Callable, Optional


class LRUCache:
    """A bounded Least Recently Used (LRU) cache.

    All core operations (get, put) must run in O(1) average time complexity.
    """

    def __init__(self, capacity: int):
        """Initialize the cache with a maximum capacity."""
        raise NotImplementedError("your code here")

    def get(self, key: Any, default: Any = None) -> Any:
        """Retrieve an item from the cache.

        If the key exists, return its value and mark it as the most recently used.
        If the key does not exist, return the default value.
        """
        raise NotImplementedError("your code here")

    def put(self, key: Any, value: Any) -> None:
        """Insert or update a key-value pair in the cache.

        Mark the key as the most recently used. If inserting causes the cache
        to exceed its capacity, evict the least recently used item.
        """
        raise NotImplementedError("your code here")

    def __len__(self) -> int:
        """Return the number of items currently in the cache."""
        raise NotImplementedError("your code here")


class TTLCache:
    """An LRU cache that supports Time-To-Live (TTL) expiry.

    Items expire after `ttl` seconds. Expired items are lazily removed on access,
    or can be proactively removed using `purge_expired`.
    """

    def __init__(
        self, 
        capacity: int, 
        ttl: float, 
        clock: Callable[[], float] = time.time
    ):
        """Initialize the cache with capacity, TTL (in seconds), and a clock source."""
        raise NotImplementedError("your code here")

    def get(self, key: Any, default: Any = None) -> Any:
        """Retrieve an item. If expired, remove it and return default."""
        raise NotImplementedError("your code here")

    def put(self, key: Any, value: Any) -> None:
        """Insert or update an item with the current timestamp."""
        raise NotImplementedError("your code here")

    def purge_expired(self) -> None:
        """Remove all expired items from the cache."""
        raise NotImplementedError("your code here")

    def __len__(self) -> int:
        """Return the number of non-expired items currently in the cache."""
        raise NotImplementedError("your code here")


class StatsCache:
    """An LRU cache with TTL that also tracks hit and miss statistics."""

    def __init__(
        self, 
        capacity: int, 
        ttl: Optional[float] = None, 
        clock: Callable[[], float] = time.time
    ):
        """Initialize the cache with capacity, optional TTL, and clock."""
        raise NotImplementedError("your code here")

    @property
    def hits(self) -> int:
        """Return the number of successful cache lookups."""
        raise NotImplementedError("your code here")

    @property
    def misses(self) -> int:
        """Return the number of failed cache lookups (not found or expired)."""
        raise NotImplementedError("your code here")

    def get(self, key: Any, default: Any = None) -> Any:
        """Retrieve an item, updating hits/misses and recency/expiry."""
        raise NotImplementedError("your code here")

    def put(self, key: Any, value: Any) -> None:
        """Insert or update a key-value pair."""
        raise NotImplementedError("your code here")

    def purge_expired(self) -> None:
        """Remove all expired items."""
        raise NotImplementedError("your code here")

    def __len__(self) -> int:
        """Return the number of non-expired items."""
        raise NotImplementedError("your code here")


def memoize(
    capacity: int, 
    ttl: Optional[float] = None, 
    clock: Callable[[], float] = time.time
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """A decorator that wraps a function with a StatsCache.

    The wrapped function must expose its underlying StatsCache instance via 
    the `.cache` attribute.
    """
    raise NotImplementedError("your code here")
