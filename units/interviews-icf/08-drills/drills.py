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
    counts = {}
    for word in words:
        if word is not " ":
            counts[word] = counts.get(word,0) +1
    return counts


    counts = {}
    for word in words:
            counts[word] = counts.get(word,0) + 1

    return counts

def merge_counts(a, b):
    """Return a new dict summing two count dicts. Inputs unchanged.
    merge_counts({"a": 1}, {"a": 2, "b": 5}) -> {"a": 3, "b": 5}
    """

    merged= {}
    for  key, value in a.items():
        if key not in b.items():
            merged[key] = value
        else:
            merged[key] = value + b[key]
    return merged




    merged = {}
    for key, value in a.items():
        if key not in b:
            merged[key] = value
        else:
            merged[key] = value + b[key]
    for key, value in b.items():
        if key not in a:
            merged[key] = value
        else:
            merged[key] = value + a[key]
    return merged

def invert(d):
    """Return {value: [keys]} with each key list sorted ascending.
    invert({"a": 1, "b": 2, "c": 1}) -> {1: ["a", "c"], 2: ["b"]}
    """

    inverted = {}
    for key, value in d.items():
        if value not in inverted:
            inverted[value] = [key]
        else:
            inverted[value].apped(key)
    for keys in inverted.values():
        keys.sort()

    return inverted

    results = {}
    for key, value in d.items():
        if value not in results:
            results[value] = [key]
        else:
            results[value].append(key)
    for keys in results.values():
        keys.sort()

    return results
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
    d[key] = d.get(key, 0) + amount

    d[key] = d.get(key, 0) + amount
    return d[key]

# ---------------------------------------------------------------- Block B
# Sorting with compound keys.  Budget: 15 min total. Learn these cold.

def by_score_desc(pairs):
    """Sort [(name, score)] by score descending, ties by name ascending.
    Return the sorted list of tuples.
    """
    return sorted(pairs, key= lambda x: (-x[1], x[0]))

    return sorted(pairs, key= lambda x: (-x[1], x[0]))

def top_k_formatted(counts, k):
    """Given {name: count}, return at most k strings "name(count)".
    Ordered by count descending, ties by name ascending.
    Entries with a count of 0 are excluded.
    top_k_formatted({"a": 1, "b": 2, "c": 0}, 5) -> ["b(2)", "a(1)"]
    """

    list_name_count = [(name, count) for name, count in counts.items() if count > 0]
    list_name_count.sort(key=lambda x: (-x[1], x[0]))
    return [f"{name}({count})" for name, count in list_name_count[:k]]

def top_k_formatted(counts, k):
    alive = sorted(
        [(name, c) for name, c in counts.items() if c > 0],
        key=lambda x: (-x[1], x[0]),
    )
    return [f"{name}({c})" for name, c in alive[:k]]
#
    # def top_k_formatted(counts, k):
    #     """Given {name: count}, return at most k strings "name(count)"."""
    #     alive = [(name, c) for name, c in counts.items() if c > 0]
    #     alive.sort(key=lambda x: (-x[1], x[0]))
    #     return [f"{name}({c})" for name, c in alive[:k]]

    """
    Difference 1: return value. .sort() rearranges the list and returns None. sorted() leaves the input alone and returns a new list.
     This is the same trap that bit you on .append in Block A — mutating methods return None, so return alive.sort() returns nothing.

    Difference 2: what it works on. .sort() is a list method and only lists have it.
     counts.items() isn't a list, so .sort() doesn't exist on it — that AttributeError above.
      sorted() accepts any iterable, so it can eat .items() directly.
    """

def by_length_then_alpha(words):
    """Sort words by length ascending, ties alphabetically."""

    return sorted(words, key=lambda x: (len(x), x))



def ranked(pairs):
    """Given [(name, score)], return ["1. name(score)", "2. ...", ...].

    Ordered by score descending, ties by name ascending. Positions start at 1.
    ranked([("b", 5), ("a", 5)]) -> ["1. a(5)", "2. b(5)"]
    """
    sorted_pairs = sorted(pairs, key=lambda x: (-x[1], x[0]))
    return [f"{i}. {name}({count})" for i , (name, count) in enumerate(sorted_pairs, start = 1)]
    ordered = sorted(pairs, key=lambda x: (-x[1], x[0]))
    return [f'{i}. {name}({score})' for i, (name, score) in enumerate(ordered, start=1)]

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
    largest_sorted = sorted(set(nums), reverse = True)
    distinct = sorted(set(nums), reverse=True)
    if len(distinct) < 2:
        return None
    return distinct[1]

# ---------------------------------------------------------------- Block C
# Prefix matching and grouping.  Budget: 8 min total.

def with_prefix(names, prefix):
    """Return names starting with prefix, sorted ascending.

    An empty prefix matches everything.
    """
    return sorted(name for name in names if name.startswith(prefix))


def group_by_first(names):
    """Return {first character: [names]}, each list sorted ascending.

    Names that are empty strings are skipped.
    """
    out= {}
    for name in names:
        if name is not None and name != "":
            first_c = name[0]
            out[first_c] = list().append(name)
    return out




    out = {}
    for name in names:
        if name is not None and name != "":
            first_char = name[0]
            if first_char not in out:
                out[first_char] = []
            out[first_char].append(name)
    return out


def longest_common_prefix(names):
    """Return the longest string that every name starts with. "" if none."""
    if not names:
        return ""
    ordered = sorted(names)
    first, last = ordered[0], ordered[-1]
    i = 0
    while i < len(first) and i < len(last) and last[i] == first[i]:
        i += 1
    return first[:i]

# ---------------------------------------------------------------- Block D
# Interval / TTL arithmetic. Half-open [start, start+ttl).  Budget: 12 min.

def is_alive(start, ttl, now):
    """True if a record created at start is alive at now.

    ttl of None means it never expires. A record is alive for
    start <= now < start + ttl.
    """
    if now < start:
        return False
    if ttl is None:
        return  True
    return start <= now < start + ttl


def remaining(start, ttl, now):
    """Remaining lifetime at now.

    None if ttl is None. 0 if already expired. Otherwise start + ttl - now.
    """
    if ttl is None:
        return None
    if not is_alive(start, ttl, now):
        return 0
    return start + ttl - now


def filter_alive(records, now):
    """records is {name: (start, ttl)}. Return sorted list of alive names."""
    out = []
    for name, (start, ttl) in records.items():
        if is_alive(start, ttl, now):
            out.append(name)
    return sorted(out)


def latest_before(events, now):
    """events is a list of (timestamp, value), unsorted.

    Return the value of the event with the largest timestamp <= now,
    or None if there is none.
    """
    best_time, best_value = None, None
    for ts, value in events:
        if ts <= now and (best_time is None or ts > best_time):
            best_time, best_value = ts, value
    return best_value

# ---------------------------------------------------------------- Block E
# Snapshot and rollback.  Budget: 10 min.

def snapshot(state):
    """state is {name: [item, ...]}. Return a copy that shares no lists.

    Mutating a list in the copy must not affect the original.
    """
    out = {}
    for name, items in state.items():
        out[name] = items.copy()
    return out


def rebase(saved, now):
    """saved is {name: (value, remaining_or_None)} taken at some earlier time.

    Return {name: (value, expiry_or_None)} where expiry = now + remaining.
    A remaining of None stays None.
    """
    return {
        name: (value, None if remaining is None else now + remaining)
        for name, (value, remaining) in saved.items()
    }

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

    def __init__(self):
        self.totals = {}
        self.touched = {}


    def add(self, name, amount):
        return self.add_at(None, name, amount)

    def add_at(self, ts, name, amount):
        self.totals[name] = self.totals.get(name, 0) + amount
        if ts is not None:
            self.touched[name] = ts
        return self.totals[name]

    def total(self, name):
        return self.totals.get(name)
    def remove(self, name):
        if name in self.totals:
            del self.totals[name]
            return True
        else:
            return False

    def top(self, n):
        alive = sorted(
            [(name, c) for name, c in self.totals.items() if c > 0],
            key=lambda x: (-x[1], x[0]),
        )
        return [f"{name}({c})" for name, c in alive[:n]]

    def last_touched(self, name):
        return self.touched.get(name, None)


# ---------------------------------------------------------------- Block G
# Formatting and parsing round-trips.  Budget: 8 min.

def format_entry(name, value):
    """Return "name(value)"."""
    return f"{name}({value})"

def parse_entry(entry):
    """Inverse of format_entry. Return (name, int value).

    parse_entry("web-utils(500)") -> ("web-utils", 500)
    Names may contain hyphens and dots but never parentheses.
    """
    i = entry.find("(")
    name = entry[:i]
    number = int(entry[i+1:-1])
    return (name, int(number))

def summarise(counts):
    """Given {name: count}, return a single string:

    "name(count), name(count), ..." ordered by count desc then name asc,
    excluding zero counts. Empty input returns "".
    """
    pairs = [(n, c) for n, c in counts.items() if c!=0]  # drop zeros
    pairs.sort(key=lambda x: (-x[1], x[0]))  # you know this one
    return ", ".join(format_entry(n,c) for n, c in pairs)