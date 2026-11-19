"""59 — Two-sum to four-sum

Implement each function so the tests in test_exercises.py pass. Read
README.md first; every exercise below is covered there.

Standard library only.
"""

from typing import List, Tuple, Optional
from collections import Counter


def two_sum(nums: List[int], target: int) -> Optional[Tuple[int, int]]:
    """Find the indices of two numbers in `nums` that sum to `target`.

    Returns a tuple of the two indices (i, j) where i < j.
    If no such pair exists, return None.
    Each input has exactly one solution or zero solutions. You cannot use the
    same element twice.

    Time Complexity: O(N)
    Space Complexity: O(N)
    """
    raise NotImplementedError("your code here")


def two_sum_sorted(nums: List[int], target: int) -> List[Tuple[int, int]]:
    """Find all unique pairs of values in a sorted list `nums` that sum to `target`.

    Returns a list of unique tuples (val1, val2) where val1 <= val2.
    The returned list of pairs must be sorted in ascending order of val1.
    Must use the two-pointer approach with O(1) auxiliary space.

    Time Complexity: O(N)
    Space Complexity: O(1) auxiliary
    """
    raise NotImplementedError("your code here")


def three_sum(nums: List[int], target: int) -> List[Tuple[int, int, int]]:
    """Find all unique triplets in `nums` that sum to `target`.

    Returns a list of unique tuples (val1, val2, val3) where val1 <= val2 <= val3.
    The returned list of triplets must be sorted lexicographically.
    Must avoid duplicate triplets.

    Time Complexity: O(N^2)
    Space Complexity: O(N) for sorting
    """
    raise NotImplementedError("your code here")


def four_sum(nums: List[int], target: int) -> List[Tuple[int, int, int, int]]:
    """Find all unique quadruplets in `nums` that sum to `target`.

    Returns a list of unique tuples (val1, val2, val3, val4) where val1 <= val2 <= val3 <= val4.
    The returned list of quadruplets must be sorted lexicographically.
    Must be implemented using a recursive k-sum pattern to avoid deeply nested loops.

    Time Complexity: O(N^3)
    Space Complexity: O(N) for sorting and recursion stack
    """
    raise NotImplementedError("your code here")


def count_pairs_with_difference(nums: List[int], k: int) -> int:
    """Count the number of unique pairs of values (a, b) such that a - b = k.

    Since we want unique pairs of values, (a, b) and (c, d) are the same pair
    if a == c and b == d.
    
    If k < 0, raise a ValueError.
    If k == 0, count the number of unique values that appear at least twice.

    Time Complexity: O(N)
    Space Complexity: O(N)
    """
    raise NotImplementedError("your code here")
