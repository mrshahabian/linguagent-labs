"""Reference answers for 07 - tooling and complexity basics."""


def linear_search(items, target):
    """Return the index of the first occurrence of target in items, or -1
    if it isn't present. Checks every item, in order - O(n)."""
    for i, item in enumerate(items):
        if item == target:
            return i
    return -1


def binary_search(sorted_items, target):
    """Return the index of target in sorted_items (which must already be
    sorted ascending), or -1 if it isn't present. Halves the search space
    each step - O(log n)."""
    low, high = 0, len(sorted_items) - 1
    while low <= high:
        mid = (low + high) // 2
        if sorted_items[mid] == target:
            return mid
        elif sorted_items[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1


def has_duplicate_pair(numbers):
    """True if any value appears more than once, checking every pair -
    deliberately the O(n^2) version. Compare this to what you already
    know from Module 03: len(numbers) != len(set(numbers)) answers the
    same question in O(n)."""
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if numbers[i] == numbers[j]:
                return True
    return False
