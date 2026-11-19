# 58 — Move zeros, in place

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/interview-drills/58-move-zeros/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/interview-drills/move-zeros?utm_source=github&utm_medium=labs)

*~40 minutes · 30 tests · Interview*

In-place array manipulation is one of the most common themes in coding interviews. At first glance, moving all zeros to the end of an array seems trivial: you can just filter out the zeros and append them to a new list. However, in production systems—such as memory allocators, embedded firmware, or high-frequency trading engines—allocating extra memory is a luxury you cannot afford. You must modify the array *in-place*, using $O(1)$ auxiliary memory.

This module covers the "Two-Pointer" pattern, a fundamental technique for in-place array partitioning. You will start with the classic "Move Zeros" problem, generalize it to move any arbitrary value, extend it to partition by a dynamic predicate, and finally solve a challenging sliding-window follow-up: finding the minimum number of swaps to group all zeros together. Mastering these techniques will give you a systematic way to approach array partitioning without falling into index-out-of-bounds errors or infinite loops.

## What you'll build

| Function / class | You'll implement | Shows up in real code as |
|---|---|---|
| `move_zeros_to_end` | Move all zeros to the end of a list in-place while preserving order | Compacting sparse arrays or defragmenting memory buffers |
| `move_value_to_end` | Generalize the two-pointer swap to any target value | Filtering out sentinel values or invalid records from a stream |
| `stable_partition` | Partition elements based on a custom boolean predicate | Separating high-priority tasks from low-priority ones in a queue |
| `min_swaps_to_group` | Calculate the minimum swaps to group all zeros together | Optimizing disk block writes or contiguous memory allocation |

---

## 1. Move Zeros to End (Stable, In-Place)

```python
>>> nums = [0, 1, 0, 3, 12]
>>> move_zeros_to_end(nums)
[1, 3, 12, 0, 0]
```

The naive way to solve this is to iterate through the list, remove each zero as you find it, and append it to the end. 

### The Pitfall
Modifying a list while iterating over it is a recipe for silent bugs. In Python, deleting an element shifts all subsequent elements to the left, causing the iterator to skip the very next element. Furthermore, `list.remove()` is an $O(N)$ operation. Calling it inside a loop results in an $O(N^2)$ time complexity, which will immediately fail any scale test in an interview.

### The Solution: Two Pointers
Instead, we maintain a **write pointer** (`w`) and a **read pointer** (`r`). Both start at index `0`. As the read pointer scans the array:
1. If `nums[r]` is non-zero, we swap `nums[w]` and `nums[r]`, then increment `w`.
2. If `nums[r]` is zero, we do nothing and let `r` advance.

Because we only swap when we find a non-zero element, the write pointer always points to the first zero in the sequence (or is equal to `r` if no zeros have been encountered yet). This guarantees that the relative order of non-zero elements is preserved, and the array is updated in a single $O(N)$ pass with $O(1)$ auxiliary space.

**In `exercises.py`:** `move_zeros_to_end` is this section's target.

**Test it:** `python3 -m unittest test_exercises.TestMoveZerosToEnd -v`

---

## 2. Generalizing to Any Value

```python
>>> items = ["a", "b", "a", "c"]
>>> move_value_to_end(items, "a")
['b', 'c', 'a', 'a']
```

Once you solve "Move Zeros", an interviewer will often ask you to generalize the function to handle any arbitrary target value, not just `0`. This target could be an integer, a string, or even a custom object.

### The Pitfall
When generalizing, candidates often write `if nums[r] is not target:`. 

In Python, `is` checks for **object identity** (whether two variables point to the exact same memory address), whereas `==` checks for **equality** (whether their values are equivalent). If your target is `0.0` and the array contains `0`, or if you are comparing strings or large integers that aren't cached by the Python interpreter, `is` will fail to match them correctly. Always use `==` and `!=` for value comparisons unless you explicitly intend to check identity.

**In `exercises.py`:** `move_value_to_end` is this section's target.

**Test it:** `python3 -m unittest test_exercises.TestMoveValueToEnd -v`

---

## 3. Stable Partitioning by Predicate

```python
>>> nums = [1, 2, 3, 4, 5]
>>> is_even = lambda x: x % 2 == 0
>>> stable_partition(nums, is_even)
[1, 3, 5, 2, 4]
```

The next logical step is to partition the array based on a dynamic condition rather than a static value. We pass a **predicate function** (a function that returns `True` or `False`) to determine which elements should be sent to the back.

### The Pitfall
Doing a fully stable partition (where *both* the front and back partitions maintain their original relative order) in-place in $O(N)$ time and $O(1)$ auxiliary space is famously difficult and requires complex block-swapping algorithms. 

To keep the problem practical, we relax the constraint: only the elements that **do not** satisfy the predicate (the ones kept at the front) must preserve their relative order. The elements that **do** satisfy the predicate (moved to the end) do not need to maintain their relative order. This allows us to reuse our elegant two-pointer swap strategy: if `not predicate(nums[r])`, we swap and increment our write pointer.

**In `exercises.py`:** `stable_partition` is this section's target.

**Test it:** `python3 -m unittest test_exercises.TestStablePartition -v`

---

## 4. Minimum Swaps to Group Zeros

```python
>>> nums = [0, 1, 0, 0, 1, 1, 0]
>>> min_swaps_to_group(nums)
1
```

Now for the ultimate follow-up. Instead of moving all zeros to the end, what if we want to group all zeros together *anywhere* in the array, and we want to find the **minimum number of arbitrary swaps** required to do so? Note that we do not actually need to modify the array; we only need to return the minimum swap count.

### The Pitfall
Many candidates try to simulate the swaps using greedy heuristics, which leads to highly complex, bug-prone code that easily exceeds the $O(N)$ time limit.

### The Sliding Window Insight
Let's think about the properties of the final grouped state:
1. Suppose there are $K$ zeros in the array. In any valid final state, all $K$ zeros must occupy a contiguous window of size $K$.
2. If we choose any window of size $K$ in the array, the number of swaps needed to bring all zeros into this window is exactly equal to the number of **non-zero elements** currently inside that window. Why? Because each non-zero element inside our target window must be swapped with a zero from outside the window.
3. Therefore, to minimize the swaps, we must find the window of size $K$ that already contains the **maximum number of zeros**.
4. The minimum swaps required is simply: 
   $$\text{Min Swaps} = K - \text{Max Zeros in any window of size } K$$

This reduces the problem to a classic **Sliding Window** problem, which we can solve in $O(N)$ time and $O(1)$ auxiliary space!

**In `exercises.py`:** `min_swaps_to_group` is this section's target.

**Test it:** `python3 -m unittest test_exercises.TestMinSwapsToGroup -v`

---

## The exercises

| Function | What it teaches |
|---|---|
| `move_zeros_to_end` | Two-pointer in-place array partitioning |
| `move_value_to_end` | Generalizing comparisons and avoiding identity (`is`) pitfalls |
| `stable_partition` | Applying dynamic predicates to in-place partitioning |
| `min_swaps_to_group` | Combining sliding window with mathematical reduction |

## Check your understanding

1. **Why does the two-pointer swap approach preserve the relative order of non-zero elements?**
   <details><summary>Answer</summary>Because the read pointer scans the array strictly from left to right, and we write non-zero elements to the write pointer in the exact order they are encountered. Since the write pointer never overtakes the read pointer, elements are never moved backward relative to other non-zero elements.</details>

2. **What is the difference between `==` and `is` in Python?**
   <details><summary>Answer</summary><code>==</code> checks for value equality (whether two objects have equivalent data), while <code>is</code> checks for object identity (whether they point to the exact same object in memory). Always use <code>==</code> for value comparisons.</details>

3. **Why is modifying a list (like calling `list.remove()`) while iterating over it dangerous?**
   <details><summary>Answer</summary>Modifying the list changes its length and shifts indices. The internal iterator keeps track of the current index, so shifting elements left causes the iterator to skip the element immediately following the deleted one.</details>

4. **How does the sliding window approach for `min_swaps_to_group` achieve $O(1)$ auxiliary space?**
   <details><summary>Answer</summary>Instead of creating new sublists for each window, we maintain a running count of zeros. When the window slides right, we add the new element entering the window and subtract the element leaving the window, which only requires a few integer variables.</details>

5. **What happens in `min_swaps_to_group` if the array contains no zeros or only zeros?**
   <details><summary>Answer</summary>If there are no zeros ($K=0$) or only zeros ($K=N$), the elements are already grouped. The function should immediately return <code>0</code> without running the sliding window logic.</details>

## Interview questions this module answers

- *How do you partition an array in-place with $O(1)$ extra space?*
- *What is the difference between value equality and object identity in Python?*
- *How can you find the minimum swaps to group elements using a sliding window?*
- *Why should you avoid modifying a collection while iterating over it?*
