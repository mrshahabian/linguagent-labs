"""Reference answers for 03 - core data structures."""


def first_and_last(items):
    """Return (first, last) from a non-empty list, using indexing - not
    len(items) - 1, use -1."""
    return (items[0], items[-1])


def unique_sorted(items):
    """Return the distinct values in items, sorted ascending."""
    return sorted(set(items))


def evens_only(numbers):
    """Return only the even numbers, in their original order, using a
    list comprehension."""
    return [n for n in numbers if n % 2 == 0]


def word_lengths(words):
    """Return {word: len(word)} for every word, using a dict
    comprehension."""
    return {word: len(word) for word in words}


def merge_dicts(a, b):
    """Merge two dicts into a new one. Where a key appears in both, b's
    value wins. Neither a nor b is modified."""
    merged = dict(a)
    merged.update(b)
    return merged


def group_by_first_letter(words):
    """Group words into {first_letter: [word, ...]}, preserving each
    word's original relative order within its group."""
    groups = {}
    for word in words:
        letter = word[0]
        if letter not in groups:
            groups[letter] = []
        groups[letter].append(word)
    return groups
