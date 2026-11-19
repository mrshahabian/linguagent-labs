"""60 — Random unique numbers

Reference solutions. Standard library only.
"""

import random
import bisect
from typing import Any, Iterator, List


def sample_without_replacement(lst: List[Any], k: int, rng: random.Random) -> List[Any]:
    if k < 0 or k > len(lst):
        raise ValueError("k must be non-negative and less than or equal to the list length")
    
    arr = list(lst)
    n = len(arr)
    for i in range(k):
        j = rng.randint(i, n - 1)
        arr[i], arr[j] = arr[j], arr[i]
    return arr[:k]


def reservoir_sample(iterator: Iterator[Any], k: int, rng: random.Random) -> List[Any]:
    if k < 0:
        raise ValueError("k must be non-negative")
    
    reservoir = []
    for i, item in enumerate(iterator):
        if i < k:
            reservoir.append(item)
        else: 
            j = rng.randint(0, i)
            if j < k:
                reservoir[j] = item
    return reservoir


class UniqueRandomGenerator:
    def __init__(self, n: int, rng: random.Random):
        if n < 0:
            raise ValueError("n must be non-negative")
        self.n = n
        self.rng = rng
        self.lookup = {}
        self.drawn = 0

    def has_next(self) -> bool:
        return self.drawn < self.n

    def draw(self) -> int:
        if not self.has_next():
            raise IndexError("All elements drawn")
        
        i = self.drawn
        j = self.rng.randint(i, self.n - 1)
        
        val_i = self.lookup.get(i, i)
        val_j = self.lookup.get(j, j)
        
        self.lookup[j] = val_i
        self.lookup[i] = val_j
        
        self.drawn += 1
        return val_j


def weighted_choice(weights: List[float], rng: random.Random) -> int:
    if not weights:
        raise ValueError("weights list cannot be empty")
    
    prefix_sums = []
    current_sum = 0.0
    for w in weights:
        if w < 0:
            raise ValueError("weights cannot be negative")
        current_sum += w
        prefix_sums.append(current_sum)
        
    if current_sum <= 0:
        raise ValueError("sum of weights must be greater than zero")
        
    r = rng.uniform(0, current_sum)
    idx = bisect.bisect_right(prefix_sums, r)
    
    # Guard against floating-point precision edge cases
    if idx >= len(weights):
        idx = len(weights) - 1
    return idx
