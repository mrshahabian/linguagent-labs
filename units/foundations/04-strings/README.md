# 04 — Strings and basic text processing

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/foundations/04-strings/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/foundations/strings?utm_source=github&utm_medium=labs)

*~35 minutes · 18 tests · Junior*

Text is everywhere — usernames, log lines, form input, file paths — and
almost every program spends some of its time slicing, cleaning, or
reshaping it. This module is the handful of string operations you'll reach
for constantly: splitting, joining, slicing, and the handful of methods
that cover most everyday text munging.

Work in `exercises.py`. Check your work with `python -m labkit check 04`, or open the notebook (`lab.ipynb`) in Colab.

Each section below ends with **Test it** — the exact command for just that
piece, run from `units/foundations/04-strings/`. No need to finish the
whole file before checking your work.

## What you'll build

Six functions in `exercises.py`, in this order. Keep this table open as you
go — each section below tells you which row it's building:

| Function | What it teaches | Shows up in real code as |
|---|---|---|
| `shout` | Basic string methods; concatenation | Formatting output for display |
| `count_words` | `.split()` with no arguments | Any "how many tokens" question over free text |
| `reverse_words` | `.split()` + `.join()` together | Reshaping text at the word level, not the character level |
| `is_palindrome` | Slicing with a step (`[::-1]`) | Reversing a sequence without a loop |
| `initials` | Iterating over the result of `.split()` | Deriving a short form from a longer piece of text |
| `slugify` | `.lower()` + `.replace()` | Turning a title into a URL-safe or filename-safe string |

---

## 1. Basic string methods

```python
>>> "hello".upper()
'HELLO'
>>> "hello".upper() + "!!!"
'HELLO!!!'
```

Strings have built-in methods for the obvious transformations —
`.upper()`, `.lower()`, and plenty more. `+` between two strings
concatenates them; there's no implicit conversion, so `"HELLO" +
3` raises `TypeError` — if you need a number in a string, convert it
first with `str(3)` or build the string with an f-string instead.

**In `exercises.py`:** `shout` is this section's target.

**Test it:** `python3 -m unittest test_exercises.TestShout -v`

## 2. `.split()` with no arguments

```python
>>> "the cat sat".split()
['the', 'cat', 'sat']
>>> "  the   cat  ".split()
['the', 'cat']
```

`.split()` with no arguments splits on whitespace **and** collapses runs
of it — extra spaces, tabs, or leading/trailing whitespace don't produce
empty strings in the result. `len(sentence.split())` is the standard way
to count words, and it's the version that survives messy real-world input
(`.split(" ")`, splitting on a literal single space, would not — it would
count the empty strings between consecutive spaces).

**In `exercises.py`:** `count_words` is this section's target.

**Test it:** `python3 -m unittest test_exercises.TestCountWords -v`

## 3. `.split()` and `.join()` together

```python
>>> words = "the cat sat".split()
>>> words
['the', 'cat', 'sat']
>>> words.reverse()
>>> " ".join(words)
'sat cat the'
```

`.split()` turns a string into a list of words; `.join()` does the
reverse — `"separator".join(list_of_strings)` glues them back together
with `separator` between each pair. `list.reverse()` reverses a list **in
place** (it returns `None`, not the reversed list — a common trap).
Split, transform the list, join back: this round trip is how you operate
on the *words* of a string rather than its raw characters.

**In `exercises.py`:** `reverse_words` is this section's target.

**Test it:** `python3 -m unittest test_exercises.TestReverseWords -v`

## 4. Slicing with a step: `[::-1]`

```python
>>> "hello"[::-1]
'olleh'
>>> "racecar"[::-1]
'racecar'
```

Slicing is `sequence[start:stop:step]` — leave `start` and `stop` empty to
mean "the whole thing," and a `step` of `-1` walks backwards, producing a
reversed copy. This works on any sequence (strings, lists, tuples), and
it's the standard, no-loop way to reverse one. A palindrome check just
compares a cleaned-up string to its own reverse.

**In `exercises.py`:** `is_palindrome` is this section's target — lowercase
and strip spaces first, so casing and spacing don't cause a false
mismatch.

**Test it:** `python3 -m unittest test_exercises.TestIsPalindrome -v`

## 5. Iterating over `.split()`'s result

```python
>>> full_name = "Ada Lovelace"
>>> [word[0] for word in full_name.split()]
['A', 'L']
>>> "".join(word[0].upper() for word in full_name.split())
'AL'
```

`full_name.split()` gives a list of words; `word[0]` is each word's first
character (indexing works the same on any string, long or short —
`word[0]` is just the first element). Wrapping that in a comprehension and
joining with `""` (an empty separator) concatenates the results directly
with nothing between them.

**In `exercises.py`:** `initials` is this section's target.

**Test it:** `python3 -m unittest test_exercises.TestInitials -v`

## 6. `.lower()` and `.replace()`

```python
>>> "Hello World".lower()
'hello world'
>>> "hello world".replace(" ", "-")
'hello-world'
```

`.replace(old, new)` swaps every occurrence of `old` for `new` — here,
every space becomes a hyphen. Chaining `.lower().replace(" ", "-")`
applies both transformations in sequence, left to right; since strings are
immutable, each method call returns a **new** string rather than modifying
the original, which is exactly what makes chaining them like this work.

**In `exercises.py`:** `slugify` is this section's target.

**Test it:** `python3 -m unittest test_exercises.TestSlugify -v`

---

## The exercises

| Function | What it teaches |
|---|---|
| `shout` | Basic string methods; concatenation |
| `count_words` | `.split()` with no arguments |
| `reverse_words` | `.split()` + `.join()` |
| `is_palindrome` | Slicing with a step |
| `initials` | Iterating over `.split()`'s result |
| `slugify` | `.lower()` + `.replace()`, chained |

## Check your understanding

1. **Why does `"hello".split(" ")` behave differently from
   `"hello   world".split()` on messy input with extra spaces?**
   <details><summary>Answer</summary><code>.split(" ")</code> splits on
   every single literal space, so consecutive spaces produce empty
   strings in the result. <code>.split()</code> with no argument splits on
   any whitespace and collapses runs of it, so extra spaces don't create
   empty entries.</details>

2. **What does `list.reverse()` return, and what's the common mistake
   people make with it?**
   <details><summary>Answer</summary>It returns <code>None</code> — it
   reverses the list in place. Writing <code>words =
   words.reverse()</code> throws away the list and leaves
   <code>words</code> as <code>None</code>.</details>

3. **What does `"abc"[::-1]` actually do, piece by piece?**
   <details><summary>Answer</summary><code>start</code> and
   <code>stop</code> are both omitted (meaning the whole string), and
   <code>step</code> is <code>-1</code>, meaning walk backwards one
   character at a time — the result is the string reversed.</details>

4. **Why does `is_palindrome` lowercase and strip spaces before comparing,
   instead of comparing the raw input to its reverse?**
   <details><summary>Answer</summary>Without normalizing, "Never Odd or
   Even" would not equal its own reverse, because the capitalization and
   spaces wouldn't line up the same way backwards as forwards, even
   though the phrase reads the same to a person.</details>

5. **Why does chaining `.lower().replace(" ", "-")` work, when strings are
   immutable?**
   <details><summary>Answer</summary>Each method returns a brand new
   string rather than modifying the original in place, so
   <code>.replace(...)</code> can be called directly on the string that
   <code>.lower()</code> just returned.</details>

## Interview questions this module answers

- *How would you reverse a string without writing a loop?*
- *How do you count the words in a sentence, robust to extra whitespace?*
- *What's the difference between `.split(" ")` and `.split()`?*
- *Why does chaining string methods work when strings are immutable?*
