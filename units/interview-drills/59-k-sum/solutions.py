"""Reference solutions for 59 — Two-sum to four-sum
"""

from typing import List, Tuple, Optional
from collections import Counter


def two_sum(nums: List[int], target: int) -> Optional[Tuple[int, int]]:
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return None


def two_sum_sorted(nums: List[int], target: int) -> List[Tuple[int, int]]:
    results = []
    left = 0
    right = len(nums) - 1
    
    while left < right:
        current_sum = nums[left] + nums[right]
        if current_sum == target:
            results.append((nums[left], nums[right]))
            # Skip duplicates for left pointer
            while left < right and nums[left] == nums[left + 1]:
                left += 1
            # Skip duplicates for right pointer
            while left < right and nums[right] == nums[right - 1]:
                right -= 1
            left += 1
            right -= 1
        elif current_sum < target:
            left += 1
        else:
            right -= 1
            
    return results


def three_sum(nums: List[int], target: int) -> List[Tuple[int, int, int]]:
    sorted_nums = sorted(nums)
    results = []
    n = len(sorted_nums)
    
    for i in range(n - 2):
        # Skip duplicate values for the first element
        if i > 0 and sorted_nums[i] == sorted_nums[i - 1]:
            continue
            
        left = i + 1
        right = n - 1
        sub_target = target - sorted_nums[i]
        
        while left < right:
            current_sum = sorted_nums[left] + sorted_nums[right]
            if current_sum == sub_target:
                results.append((sorted_nums[i], sorted_nums[left], sorted_nums[right]))
                while left < right and sorted_nums[left] == sorted_nums[left + 1]:
                    left += 1
                while left < right and sorted_nums[right] == sorted_nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif current_sum < sub_target:
                left += 1
            else:
                right -= 1
                
    return results


def four_sum(nums: List[int], target: int) -> List[Tuple[int, int, int, int]]:
    sorted_nums = sorted(nums)
    n = len(sorted_nums)
    
    def k_sum(start: int, target_val: int, k: int) -> List[Tuple[int, ...]]:
        res = []
        if start == n:
            return res
            
        # Optimization check: if the smallest possible sum is greater than target_val,
        # or the largest possible sum is smaller than target_val, no solution exists.
        average_val = target_val // k
        if sorted_nums[start] > average_val or sorted_nums[-1] < average_val:
            return res
            
        if k == 2:
            # Base case: two-pointer search
            left = start
            right = n - 1
            while left < right:
                curr_sum = sorted_nums[left] + sorted_nums[right]
                if curr_sum == target_val:
                    res.append((sorted_nums[left], sorted_nums[right]))
                    while left < right and sorted_nums[left] == sorted_nums[left + 1]:
                        left += 1
                    while left < right and sorted_nums[right] == sorted_nums[right - 1]:
                        right -= 1
                    left += 1
                    right -= 1
                elif curr_sum < target_val:
                    left += 1
                else:
                    right -= 1
            return res
            
        # Recursive case: reduce k-sum to (k-1)-sum
        for i in range(start, n - k + 1):
            if i > start and sorted_nums[i] == sorted_nums[i - 1]:
                continue
            sub_results = k_sum(i + 1, target_val - sorted_nums[i], k - 1)
            for sub in sub_results:
                res.append((sorted_nums[i],) + sub)
        return res

    # Cast the tuples of length 4 to the exact return type
    return [ (r[0], r[1], r[2], r[3]) for r in k_sum(0, target, 4) ]


def count_pairs_with_difference(nums: List[int], k: int) -> int:
    if k < 0:
        raise ValueError("k must be non-negative")
        
    if k == 0:
        counts = Counter(nums)
        return sum(1 for val, count in counts.items() if count > 1)
        
    unique_vals = set(nums)
    count = 0
    for val in unique_vals:
        if (val + k) in unique_vals:
            count += 1
    return count
