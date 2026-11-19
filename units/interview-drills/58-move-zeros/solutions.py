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
    w = 0
    for r in range(len(nums)):
        if nums[r] != 0:
            nums[w], nums[r] = nums[r], nums[w]
            w += 1
    return nums


def move_value_to_end(nums: List[Any], target: Any) -> List[Any]:
    """Move all occurrences of target to the end of the list in-place.

    The relative order of non-target elements must be maintained.
    Must operate in-place with O(1) auxiliary space.
    Use proper equality checking (==) to identify target elements.
    Returns the modified list.
    """
    w = 0
    for r in range(len(nums)):
        if nums[r] != target:
            nums[w], nums[r] = nums[r], nums[w]
            w += 1
    return nums


def stable_partition(nums: List[Any], predicate: Callable[[Any], bool]) -> List[Any]:
    """Partition the list in-place based on a predicate function.

    Elements for which predicate(x) is False must be moved to the front,
    preserving their relative order.
    Elements for which predicate(x) is True must be moved to the end.
    Their relative order does not need to be preserved.
    Must operate in-place with O(1) auxiliary space.
    Returns the modified list.
    """
    w = 0
    for r in range(len(nums)):
        if not predicate(nums[r]):
            nums[w], nums[r] = nums[r], nums[w]
            w += 1
    return nums


def min_swaps_to_group(nums: List[int]) -> int:
    """Calculate the minimum number of arbitrary swaps needed to group
    all zeros together anywhere in the list.

    The list should not be modified.
    Must operate in O(N) time and O(1) auxiliary space.
    """
    total_zeros = nums.count(0)
    if total_zeros == 0 or total_zeros == len(nums):
        return 0

    current_zeros = 0
    for i in range(total_zeros):
        if nums[i] == 0:
            current_zeros += 1

    max_zeros = current_zeros

    for i in range(total_zeros, len(nums)):
        if nums[i] == 0:
            current_zeros += 1
        if nums[i - total_zeros] == 0:
            current_zeros -= 1
        if current_zeros > max_zeros:
            max_zeros = current_zeros

    return total_zeros - max_zeros
