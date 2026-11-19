"""60 — Random unique numbers

Implement each function and class so the tests in test_exercises.py pass.
Read README.md first; every exercise below is covered there.

Standard library only. Inject the provided `rng` (random.Random instance) 
for all random operations to ensure deterministic results.
"""

import random
import bisect
from typing import Any, Iterator, List


def sample_without_replacement(lst: List[Any], k: int, rng: random.Random) -> List[Any]:
    """Return a random sample of k unique elements from lst.

    Must run in O(k) time and O(n) space (to copy the list).
    Must not mutate the original list.
    
    To ensure deterministic test behavior, use `rng.randint(i, len(lst) - 1)`
    to select the swap target for index i.
    
    Raises ValueError if k is negative or greater than the length of lst.
    """
    raise NotImplementedError("your code here")


def reservoir_sample(iterator: Iterator[Any], k: int, rng: random.Random) -> List[Any]:
    """Sample k items from an iterator of unknown length using reservoir sampling.

    Must run in O(n) time and O(k) auxiliary space.
    If the iterator has fewer than k items, return all of them in the order
    they were received.
    
    To ensure deterministic test behavior:
    - For the first k items, append them directly to the reservoir.
    - For the i-th item (0-indexed, where i >= k), generate a random index
      j using `rng.randint(0, i)`. If j < k, replace reservoir[j] with the item.
      
    Raises ValueError if k is negative.
    """
    raise NotImplementedError("your code here")


class UniqueRandomGenerator:
    """Generates unique random integers from 0 to n-1 using a lazy swap map.

    Must run in O(1) time per draw and O(d) space, where d is the number of draws.
    Does NOT allocate an array of size n.
    """

    def __init__(self, n: int, rng: random.Random):
        """Initialize the generator with range n and a seeded rng.

        Raises ValueError if n is negative.
        """
        raise NotImplementedError("your code here")

    def has_next(self) -> bool:
        """Return True if there are still unique numbers left to draw, else False."""
        raise NotImplementedError("your code here")

    def draw(self) -> int:
        """Draw the next unique random integer.

        To ensure deterministic test behavior, let i be the current draw count
        (starting at 0). Pick a random index j using `rng.randint(i, n - 1)`.
        Swap the values at i and j in your lazy map, and return the value
        swapped into index i.

        Raises IndexError if all n elements have already been drawn.
        """
        raise NotImplementedError("your code here")


def weighted_choice(weights: List[float], rng: random.Random) -> int:
    """Select a random index from weights proportional to its weight.

    Must run in O(log n) time per selection after O(n) preprocessing.
    Use cumulative weights (prefix sums) and binary search (`bisect.bisect_right`).
    
    To ensure deterministic test behavior, generate the random float using
    `rng.uniform(0, total_weight)`.
    
    Raises ValueError if weights is empty, if any weight is negative, 
    or if the sum of all weights is zero.
    """
    raise NotImplementedError("your code here")
