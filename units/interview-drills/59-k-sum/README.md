# 59 — Two-sum to four-sum

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/interview-drills/59-k-sum/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/interview-drills/k-sum?utm_source=github&utm_medium=labs)

*~45 minutes · 21 tests · Interview*

Finding combinations of numbers that sum to a target is the quintessential coding interview pattern. It begins with the classic "Two Sum" (often the very first question on LeetCode) and quickly escalates as interviewers add constraints: sorted inputs, unique combinations, three elements, four elements, or a generalized $k$ elements. 

In a real-world setting, these patterns show up whenever you need to match complementary data points—such as balancing financial ledgers, optimizing bin-packing algorithms, or identifying pairs of transactions that cancel each other out. In interviews, this progression is used to test if you can move beyond brute-force nested loops ($O(N^k)$ complexity) to optimal linear or sub-quadratic solutions using hash maps, two-pointer techniques, and elegant recursion.

## What you'll build

You will implement five functions that progress from finding simple pairs to generalizing the $k$-sum problem and counting difference pairs:

| Function / class | You'll implement | Shows up in real code as |
|---|---|---|
| `two_sum` | Single-pass hash map to find indices in $O(N)$ time | Fast lookup of complementary pairs in unsorted datasets |
| `two_sum_sorted` | Two-pointer search on sorted values, skipping duplicates | Memory-efficient pair matching with zero extra space |
| `three_sum` | Sorting combined with a nested two-pointer search | Finding triplets (e.g., three-way portfolio hedges) without duplicates |
| `four_sum` | Recursive $k$-sum reduction to a base 2-sum case | Multi-dimensional matching and combinatorial optimization |
| `count_pairs_with_difference` | Hash set matching for absolute differences | Analyzing variance, clustering, or matching buy/sell orders |

---

## 1. Two-sum with a Hash Map

The simplest version of the problem asks you to find the *indices* of two numbers in an unsorted array that sum to a target. A brute-force solution uses nested loops, taking $O(N^2)$ time. We can do better by trading space for speed.

By storing each number's index in a hash map as we iterate, we can check in $O(1)$ time if the complement (`target - current_value`) has already been seen:

```python
# Conceptual idea (not the exact exercise solution)
seen = {}
for index, value in enumerate(nums):
    complement = target - value
    if complement in seen:
        return [seen[complement], index]
    seen[value] = index
```

**The Pitfall:** A common bug is returning the same index twice. For example, if `nums = [3, 2, 4]` and `target = 6`, a naive implementation might see `3` at index 0, calculate the complement as `3`, and immediately match it with index 0. By checking the hash map *before* adding the current element to it, you guarantee that you never match an element with itself.

**Test it:** `python3 -m unittest test_exercises.TestTwoSum -v`

---

## 2. Two-sum on Sorted Input (Two Pointers)

If the input array is already sorted, we can find matching pairs in $O(N)$ time *without* using $O(N)$ extra space. Instead, we use two pointers: one starting at the beginning (`left`) and one at the end (`right`).

If their sum is too small, we move the `left` pointer inward to increase the sum. If their sum is too large, we move the `right` pointer inward to decrease it:

```python
# Conceptual two-pointer movement
while left < right:
    current_sum = nums[left] + nums[right]
    if current_sum == target:
        # Found a match!
        left += 1
        right -= 1
    elif current_sum < target:
        left += 1
    else:
        right -= 1
```

**The Pitfall:** When asked to return all *unique* pairs of values (not indices), duplicates in the input array will cause duplicate pairs in your output. For example, if `nums = [1, 1, 2, 2, 3]` and `target = 3`, simply incrementing pointers can yield `(1, 2)` multiple times. To avoid this, you must advance the pointers past identical values after finding a match.

**Test it:** `python3 -m unittest test_exercises.TestTwoSumSorted -v`

---

## 3. Three-sum (Reducing to Two-sum)

To find all unique triplets that sum to a target, we can sort the array and iterate through it. For each element `nums[i]`, we treat the remainder of the problem as a 2-sum problem on the suffix of the array (`nums[i+1:]`) with a new target of `target - nums[i]`.

```python
# Conceptual structure
nums.sort()
for i in range(len(nums) - 2):
    # Run two-pointer search on nums[i+1:]
```

**The Pitfall:** Duplicate triplets are highly common if you don't skip duplicates at both levels. You must skip duplicates in the outer loop (`if i > 0 and nums[i] == nums[i-1]: continue`) *and* skip duplicates inside the inner two-pointer loop. Failing to do both will result in redundant triplets and a failing grade in interviews.

**Test it:** `python3 -m unittest test_exercises.TestThreeSum -v`

---

## 4. Four-sum and the General $k$-sum Pattern

Writing a dedicated nested loop for 4-sum is tedious, and what if the interviewer asks for 5-sum or $k$-sum? The clean way to solve this is to write a recursive $k$-sum function.

To find $k$ numbers summing to a target:
1. **Base Case:** If $k == 2$, use the optimal two-pointer `two_sum` approach.
2. **Recursive Case:** For $k > 2$, loop through the array. For each element, recursively call $(k-1)$-sum on the remaining elements to the right, with the target reduced by the current element's value.

```python
# Conceptual recursive structure
def k_sum(start_index, target, k):
    if k == 2:
        return two_pointer_search(start_index, target)
    
    results = []
    for i in range(start_index, len(nums)):
        # Skip duplicates...
        sub_results = k_sum(i + 1, target - nums[i], k - 1)
        for sub in sub_results:
            results.append([nums[i]] + sub)
    return results
```

**The Pitfall:** Slicing arrays (e.g., `nums[i+1:]`) inside recursive calls copies data, which degrades performance to $O(N^k)$ space and adds massive overhead. Instead, pass a `start` index through the recursion to keep space complexity at $O(k)$ for the call stack.

**Test it:** `python3 -m unittest test_exercises.TestFourSum -v`

---

## 5. Count Pairs with a Given Difference

A common twist is finding pairs with a specific absolute difference: $A[i] - A[j] = k$. Unlike the sum-based problems, difference-based problems are not symmetric. 

To solve this in $O(N)$ time, we can use a hash set of unique values. For each unique number $x$, we check if $x + k$ exists in our set. 

**The Pitfall:** The edge case $k = 0$ behaves completely differently. If $k = 0$, we are looking for elements that appear *at least twice* in the original array. If you use a standard set, you lose duplicate counts and will return 0. You must handle $k = 0$ separately by counting frequencies.

**Test it:** `python3 -m unittest test_exercises.TestCountPairsWithDifference -v`

---

## The exercises

| Function | What it teaches |
|---|---|
| `two_sum` | Hash map lookup for index-based matching |
| `two_sum_sorted` | Two-pointer technique to avoid extra space |
| `three_sum` | Reducing higher-order sum problems to lower-order ones |
| `four_sum` | The recursive $k$-sum pattern for clean, scalable code |
| `count_pairs_with_difference` | Handling non-symmetric relations and $k=0$ edge cases |

## Check your understanding

1. **Why does `two_sum` use a hash map instead of sorting first?**
   <details><summary>Answer</summary>Sorting the array takes $O(N \log N)$ time and scrambles the original indices. If the problem requires returning the original indices, a hash map is ideal because it preserves indices and runs in $O(N)$ time.</details>

2. **In `two_sum_sorted`, why do we only increment `left` when the sum is too small?**
   <details><summary>Answer</summary>Because the array is sorted in ascending order. Incrementing `left` moves us to a larger value, which increases the overall sum. Decrementing `right` moves us to a smaller value, which decreases the sum.</details>

3. **What is the time complexity of the recursive `four_sum` implementation?**
   <details><summary>Answer</summary>Sorting takes $O(N \log N)$ time. The recursive $k$-sum algorithm for $k=4$ runs in $O(N^{k-1})$ time, which is $O(N^3)$. Space complexity is $O(N)$ to store the sorted array and $O(k)$ for the recursion stack.</details>

4. **Why is it important to check if $k < 0$ in `count_pairs_with_difference`?**
   <details><summary>Answer</summary>Absolute difference is always non-negative ($|a - b| \ge 0$). If a negative $k$ is passed, it is mathematically impossible to have a valid pair, so the function should raise a <code>ValueError</code>.</details>

## Interview questions this module answers

- *How do you find two numbers in an array that sum to a target in linear time?*
- *How can you solve the Two-Sum problem with $O(1)$ extra space?*
- *How do you prevent duplicate combinations in Three-Sum and Four-Sum?*
- *How do you generalize the sum problem to $k$ elements recursively?*
- *How do you find pairs with a given difference in $O(N)$ time, and how do you handle a difference of zero?*
