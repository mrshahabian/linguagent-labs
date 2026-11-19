"""58 - Move zeros, in place

Implement each function so the tests in test_exercises.py pass. Read
README.md first; every exercise below is covered there.
"""

from typing import List, Callable, Any


def move_zeros_to_end(nums: List[int]) -> List[int]:
    """Move all 0s to the end of the list in-place.

    The relative order of the non-zero elements must be maintained.
    Must operate in-place with O(1) auxiliary space.
    Returns the modified list.
    """
    raise NotImplementedError("your code here")


def move_value_to_end(nums: List[Any], target: Any) -> List[Any]:
    """Move all occurrences of target to the end of the list in-place.

    The relative order of non-target elements must be maintained.
    Must operate in-place with O(1) auxiliary space.
    Use proper equality checking (==) to identify target elements.
    Returns the modified list.
    """
    raise NotImplementedError("your code here")


def stable_partition(nums: List[Any], predicate: Callable[[Any], bool]) -> List[Any]:
    """Partition the list in-place based on a predicate function.

    Elements for which predicate(x) is False must be moved to the front,
    preserving their relative order.
    Elements for which predicate(x) is True must be moved to the end.
    Their relative order does not need to be preserved.
    Must operate in-place with O(1) auxiliary space.
    Returns the modified list.
    """
    raise NotImplementedError("your code here")


def min_swaps_to_group(nums: List[int]) -> int:
    """Calculate the minimum number of arbitrary swaps needed to group
    all zeros together anywhere in the list.

    The list should not be modified.
    Must operate in O(N) time and O(1) auxiliary space.
    """
    raise NotImplementedError("your code here")
