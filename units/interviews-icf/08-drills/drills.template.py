"""ICF warm-up drills.

Implement each function. Run ./drill to check, or ./drill A to check one block.
Standard library only. Aim for the times in the comments; if a drill takes
more than double its budget, that is your weak spot, not a hard problem.
"""


# ---------------------------------------------------------------- Block A
# Dictionaries as the default data structure.  Budget: 10 min total.

def word_count(words):
    """Return {word: number of occurrences} for a list of strings.
    word_count(["a", "b", "a"]) -> {"a": 2, "b": 1}
    """
    # TODO: your code here


def merge_counts(a, b):
    """Return a new dict summing two count dicts. Inputs unchanged.
    merge_counts({"a": 1}, {"a": 2, "b": 5}) -> {"a": 3, "b": 5}
    """
    # TODO: your code here


def invert(d):
    """Return {value: [keys]} with each key list sorted ascending.
    invert({"a": 1, "b": 2, "c": 1}) -> {1: ["a", "c"], 2: ["b"]}
    """
    # TODO: your code here

    #
    # result = {}
    # for key, value in d.items():
    #     result.setdefault(value, []).append(key)
    # for keys in result.values():
    #     keys.sort()
    # return result

def bump(d, key, amount=1):
    """Add amount to d[key], creating it at 0 if absent. Return the new value.
    Mutates d. This is the single most common line in an ICF solution.
    """
    # TODO: your code here


# ---------------------------------------------------------------- Block B
# Sorting with compound keys.  Budget: 15 min total. Learn these cold.

def by_score_desc(pairs):
    """Sort [(name, score)] by score descending, ties by name ascending.
    Return the sorted list of tuples.
    """
    # TODO: your code here


def top_k_formatted(counts, k):
    """Given {name: count}, return at most k strings "name(count)".
    Ordered by count descending, ties by name ascending.
    Entries with a count of 0 are excluded.
    top_k_formatted({"a": 1, "b": 2, "c": 0}, 5) -> ["b(2)", "a(1)"]
    """
    # TODO: your code here


def by_length_then_alpha(words):
    """Sort words by length ascending, ties alphabetically."""
    # TODO: your code here


def ranked(pairs):
    """Given [(name, score)], return ["1. name(score)", "2. ...", ...].

    Ordered by score descending, ties by name ascending. Positions start at 1.
    ranked([("b", 5), ("a", 5)]) -> ["1. a(5)", "2. b(5)"]
    """
    # TODO: your code here


# def second_largest(nums):
#     distinct = sorted(nums, reverse=True)
#     if len(distinct) < 2:
#         return None
#     return distinct[1]
# second_largest([("a", 1), ("b", 2), ("c", 0)])

# def second_largest(nums):
#     """Return the second largest distinct value, or None if there isn't one."""
#     sorted_nums = sorted(nums, key=lambda x: (-x[1], x[0]))
#     if len(sorted_nums) > 1:
#         return sorted_nums[1]
#     else:
#         return None
def second_largest(nums):
    """Return the second largest distinct value, or None if there isn't one."""
    # TODO: your code here


# ---------------------------------------------------------------- Block C
# Prefix matching and grouping.  Budget: 8 min total.

def with_prefix(names, prefix):
    """Return names starting with prefix, sorted ascending.

    An empty prefix matches everything.
    """
    # TODO: your code here


def group_by_first(names):
    """Return {first character: [names]}, each list sorted ascending.

    Names that are empty strings are skipped.
    """
    # TODO: your code here


def longest_common_prefix(names):
    """Return the longest string that every name starts with. "" if none."""
    # TODO: your code here


# ---------------------------------------------------------------- Block D
# Interval / TTL arithmetic. Half-open [start, start+ttl).  Budget: 12 min.

def is_alive(start, ttl, now):
    """True if a record created at start is alive at now.

    ttl of None means it never expires. A record is alive for
    start <= now < start + ttl.
    """
    # TODO: your code here


def remaining(start, ttl, now):
    """Remaining lifetime at now.

    None if ttl is None. 0 if already expired. Otherwise start + ttl - now.
    """
    # TODO: your code here


def filter_alive(records, now):
    """records is {name: (start, ttl)}. Return sorted list of alive names."""
    # TODO: your code here


def latest_before(events, now):
    """events is a list of (timestamp, value), unsorted.

    Return the value of the event with the largest timestamp <= now,
    or None if there is none.
    """
    # TODO: your code here


# ---------------------------------------------------------------- Block E
# Snapshot and rollback.  Budget: 10 min.

def snapshot(state):
    """state is {name: [item, ...]}. Return a copy that shares no lists.

    Mutating a list in the copy must not affect the original.
    """
    # TODO: your code here


def rebase(saved, now):
    """saved is {name: (value, remaining_or_None)} taken at some earlier time.

    Return {name: (value, expiry_or_None)} where expiry = now + remaining.
    A remaining of None stays None.
    """
    # TODO: your code here


# ---------------------------------------------------------------- Block F
# Small class design, including the delegation pattern ICF loves.
# Budget: 12 min.

class Ledger:
    """A tiny store you will build a hundred variations of.

    add(name, amount)        -> new total for name; creates at 0 if absent
    total(name)              -> current total, or None if name unknown
    remove(name)             -> True if it existed and was removed, else False
    top(n)                   -> at most n strings "name(total)", total desc,
                                ties by name ascending
    add_at(ts, name, amount) -> same as add, but records ts as the last
                                touched time for name
    last_touched(name)       -> the ts most recently passed to add_at for
                                name, or None if it has only ever seen add()

    add(name, amount) must behave exactly like add_at(0, name, amount) except
    that it does not update last_touched. Do not duplicate the logic.
    """

    # TODO: implement this class


# ---------------------------------------------------------------- Block G
# Formatting and parsing round-trips.  Budget: 8 min.

def format_entry(name, value):
    """Return "name(value)"."""
    # TODO: your code here


def parse_entry(entry):
    """Inverse of format_entry. Return (name, int value).

    parse_entry("web-utils(500)") -> ("web-utils", 500)
    Names may contain hyphens and dots but never parentheses.
    """
    # TODO: your code here


def summarise(counts):
    """Given {name: count}, return a single string:

    "name(count), name(count), ..." ordered by count desc then name asc,
    excluding zero counts. Empty input returns "".
    """
    # TODO: your code here
