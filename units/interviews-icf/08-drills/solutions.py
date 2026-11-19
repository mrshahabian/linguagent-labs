"""Reference answers for drills.py.

Look here after you have attempted a block, not before.

Where there is a shorter clever version and a clearer plain one, this file
picks the one you should be able to type correctly under time pressure.
"""


# ---------------------------------------------------------------- Block A
# Dictionaries as the default data structure.

def word_count(words):
    counts = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    return counts


def merge_counts(a, b):
    merged = dict(a)
    for key, value in b.items():
        merged[key] = merged.get(key, 0) + value
    return merged


def invert(d):
    result = {}
    for key, value in d.items():
        result.setdefault(value, []).append(key)
    for keys in result.values():
        keys.sort()
    return result


def bump(d, key, amount=1):
    d[key] = d.get(key, 0) + amount
    return d[key]


# ---------------------------------------------------------------- Block B
# Sorting with compound keys.  Learn these cold.

def by_score_desc(pairs):
    return sorted(pairs, key=lambda pair: (-pair[1], pair[0]))


def top_k_formatted(counts, k):
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [f"{name}({count})" for name, count in ordered if count > 0][:k]


def by_length_then_alpha(words):
    return sorted(words, key=lambda word: (len(word), word))


def ranked(pairs):
    ordered = sorted(pairs, key=lambda pair: (-pair[1], pair[0]))
    return [f"{i}. {name}({score})" for i, (name, score) in enumerate(ordered, 1)]


def second_largest(values):
    unique = sorted(set(values), reverse=True)
    return unique[1] if len(unique) > 1 else None


# ---------------------------------------------------------------- Block C
# Prefix matching and grouping.

def with_prefix(names, prefix):
    return sorted(name for name in names if name.startswith(prefix))


def group_by_first(names):
    groups = {}
    for name in names:
        if not name:
            continue
        groups.setdefault(name[0], []).append(name)
    for names_in_group in groups.values():
        names_in_group.sort()
    return groups


def longest_common_prefix(names):
    if not names:
        return ""
    shortest = min(names, key=len)
    for index, char in enumerate(shortest):
        for name in names:
            if name[index] != char:
                return shortest[:index]
    return shortest


# ---------------------------------------------------------------- Block D
# Interval / TTL arithmetic, half-open [start, start + ttl).

def is_alive(start, ttl, now):
    if ttl is None:
        return now >= start
    return start <= now < start + ttl


def remaining(start, ttl, now):
    if ttl is None:
        return None
    return max(0, start + ttl - now)


def filter_alive(records, now):
    return sorted(
        name for name, (start, ttl) in records.items() if is_alive(start, ttl, now)
    )


def latest_before(events, now):
    candidates = [(ts, value) for ts, value in events if ts <= now]
    return max(candidates)[1] if candidates else None


# ---------------------------------------------------------------- Block E
# Snapshot and rollback.

def snapshot(state):
    """A copy that shares no lists with the original."""
    return {name: list(items) for name, items in state.items()}


def rebase(saved, now):
    """Turn saved remaining-lifetimes back into absolute expiries at `now`."""
    return {
        name: (value, None if left is None else now + left)
        for name, (value, left) in saved.items()
    }


# ---------------------------------------------------------------- Block F
# Small class design + delegation.

class Ledger:
    def __init__(self):
        self._totals = {}
        self._last_touched = {}

    def _apply(self, name, amount):
        """The one place the add logic lives."""
        new_total = self._totals.get(name, 0) + amount
        self._totals[name] = new_total
        return new_total

    def add(self, name, amount):
        return self._apply(name, amount)

    def add_at(self, ts, name, amount):
        total = self._apply(name, amount)
        self._last_touched[name] = ts
        return total

    def total(self, name):
        return self._totals.get(name)

    def remove(self, name):
        if name in self._totals:
            del self._totals[name]
            self._last_touched.pop(name, None)
            return True
        return False

    def top(self, n):
        ordered = sorted(self._totals.items(), key=lambda kv: (-kv[1], kv[0]))
        return [f"{name}({total})" for name, total in ordered[:n]]

    def last_touched(self, name):
        return self._last_touched.get(name)


# ---------------------------------------------------------------- Block G
# Formatting and parsing.

def format_entry(name, value):
    return f"{name}({value})"


def parse_entry(entry):
    name, _, rest = entry.partition("(")
    return name, int(rest.rstrip(")"))


def summarise(counts):
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ", ".join(format_entry(name, c) for name, c in ordered if c > 0)
