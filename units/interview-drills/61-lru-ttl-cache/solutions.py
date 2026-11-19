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
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer")
        self.capacity = capacity
        self._cache: OrderedDict[Any, Any] = OrderedDict()

    def get(self, key: Any, default: Any = None) -> Any:
        if key not in self._cache:
            return default
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: Any, value: Any) -> None:
        if key in self._cache:
            self._cache.pop(key)
        elif len(self._cache) >= self.capacity:
            self._cache.popitem(last=False)
        self._cache[key] = value

    def __len__(self) -> int:
        return len(self._cache)


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
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer")
        self.capacity = capacity
        self.ttl = ttl
        self.clock = clock
        self._cache: OrderedDict[Any, tuple[Any, float]] = OrderedDict()

    def _is_expired(self, timestamp: float) -> bool:
        return (self.clock() - timestamp) >= self.ttl

    def get(self, key: Any, default: Any = None) -> Any:
        if key not in self._cache:
            return default
        value, timestamp = self._cache[key]
        if self._is_expired(timestamp):
            self._cache.pop(key)
            return default
        self._cache.move_to_end(key)
        return value

    def put(self, key: Any, value: Any) -> None:
        now = self.clock()
        if key in self._cache:
            self._cache.pop(key)
        elif len(self._cache) >= self.capacity:
            self._cache.popitem(last=False)
        self._cache[key] = (value, now)

    def purge_expired(self) -> None:
        now = self.clock()
        expired_keys = [
            k for k, (_, ts) in self._cache.items() 
            if (now - ts) >= self.ttl
        ]
        for k in expired_keys:
            self._cache.pop(k, None)

    def __len__(self) -> int:
        now = self.clock()
        return sum(
            1 for _, ts in self._cache.values() 
            if (now - ts) < self.ttl
        )


class StatsCache:
    """An LRU cache with TTL that also tracks hit and miss statistics."""

    def __init__(
        self, 
        capacity: int, 
        ttl: Optional[float] = None, 
        clock: Callable[[], float] = time.time
    ):
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer")
        self.capacity = capacity
        self.ttl = ttl
        self.clock = clock
        self._cache: OrderedDict[Any, tuple[Any, float]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    def _is_expired(self, timestamp: float) -> bool:
        if self.ttl is None:
            return False
        return (self.clock() - timestamp) >= self.ttl

    def get(self, key: Any, default: Any = None) -> Any:
        if key not in self._cache:
            self._misses += 1
            return default
        
        value, timestamp = self._cache[key]
        if self._is_expired(timestamp):
            self._cache.pop(key)
            self._misses += 1
            return default

        self._cache.move_to_end(key)
        self._hits += 1
        return value

    def put(self, key: Any, value: Any) -> None:
        now = self.clock()
        if key in self._cache:
            self._cache.pop(key)
        elif len(self._cache) >= self.capacity:
            self._cache.popitem(last=False)
        self._cache[key] = (value, now)

    def purge_expired(self) -> None:
        if self.ttl is None:
            return
        now = self.clock()
        expired_keys = [
            k for k, (_, ts) in self._cache.items() 
            if (now - ts) >= self.ttl
        ]
        for k in expired_keys:
            self._cache.pop(k, None)

    def __len__(self) -> int:
        if self.ttl is None:
            return len(self._cache)
        now = self.clock()
        return sum(
            1 for _, ts in self._cache.values() 
            if (now - ts) < self.ttl
        )


def memoize(
    capacity: int, 
    ttl: Optional[float] = None, 
    clock: Callable[[], float] = time.time
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """A decorator that wraps a function with a StatsCache.

    The wrapped function must expose its underlying StatsCache instance via 
    the `.cache` attribute.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        cache = StatsCache(capacity=capacity, ttl=ttl, clock=clock)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = (args, tuple(sorted(kwargs.items())))
            
            sentinel = object()
            val = cache.get(key, default=sentinel)
            if val is sentinel:
                val = func(*args, **kwargs)
                cache.put(key, val)
            return val

        wrapper.cache = cache  # type: ignore
        return wrapper
    return decorator
