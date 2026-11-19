"""Reference answers for drills.py.

Look here after you have attempted a block, not before. Where there is a
shorter clever version and a clearer plain one, this file picks the one you
should be able to type correctly under time pressure.
"""

from dataclasses import dataclass, field


# ------------------------------------------------------------------ Block A

@dataclass
class Plan:
    preempt: list = field(default_factory=list)
    decode: list = field(default_factory=list)
    prefill: list = field(default_factory=list)


def collect_ids(requests):
    return [r.id for r in requests]


def newest_first(admitted):
    return list(reversed(admitted))


def split_by_prefill(admitted):
    ready = [r for r in admitted if r.prefill_remaining == 0]
    started = [r for r in admitted if r.prefill_remaining > 0]
    return ready, started


# ------------------------------------------------------------------ Block B

def by_total_desc(pairs):
    return sorted(pairs, key=lambda pair: (-pair[1], pair[0]))


def top_k_formatted(totals, k):
    ordered = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
    return [f"{name}({total})" for name, total in ordered[:k]]


def by_priority_desc(requests):
    # One key, not two: sorted() is stable, so equal priorities keep the
    # input's (arrival) order for free.
    return sorted(requests, key=lambda r: -r.priority)


def lowest_priority_newest(admitted):
    rank = {r.id: i for i, r in enumerate(admitted)}
    return min(admitted, key=lambda r: (r.priority, -rank[r.id]))


# ------------------------------------------------------------------ Block C

def take_while_fits(costs, budget):
    taken = []
    for cost in costs:
        if cost > budget:
            break  # stop, don't skip ahead
        taken.append(cost)
        budget -= cost
    return taken


def take_greedily(costs, budget):
    taken = []
    for cost in costs:
        if cost > budget:
            continue  # the WRONG rule for this exercise
        taken.append(cost)
        budget -= cost
    return taken


def chunk_to_budget(needs, budget):
    taken = []
    for need in needs:
        if budget <= 0:
            break
        chunk = min(budget, need)
        taken.append(chunk)
        budget -= chunk
    return taken


def first_fit_index(costs, budget):
    for index, cost in enumerate(costs):
        if cost > budget:
            return index
        budget -= cost
    return None


# ------------------------------------------------------------------ Block D

def peak_slots(prompt_len, max_tokens):
    return prompt_len + max_tokens - 1


def slots_after_decodes(prompt_len, decodes_done):
    return prompt_len + decodes_done


def growth_need(kv_values):
    return sum(kv + 1 for kv in kv_values)


def evict_until_fits(kv_values, capacity):
    kept = list(kv_values)
    dropped = []
    while kept and growth_need(kept) > capacity:
        dropped.append(len(kept) - 1)
        kept.pop()
    return kept, dropped


def replay_cost(prompt_len, tokens_generated):
    return prompt_len + tokens_generated


# ------------------------------------------------------------------ Block E

def evictable(admitted, candidate_priority):
    return [r for r in admitted if r.priority < candidate_priority]


def enough_to_free(kv_values, shortfall):
    return sum(kv + 1 for kv in kv_values) >= shortfall


def evict_for_shortfall(kv_values, shortfall):
    evicted = []
    freed = 0
    for index, kv in enumerate(kv_values):
        if freed >= shortfall:
            break
        evicted.append(index)
        freed += kv + 1
    return evicted, freed


# ------------------------------------------------------------------ Block F

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


# ------------------------------------------------------------------ Block G

def rebase(saved, now):
    return {
        name: (value, None if remaining is None else now + remaining)
        for name, (value, remaining) in saved.items()
    }


def pyramid(n):
    return [" " * (n - i) + "*" * (2 * i - 1) for i in range(1, n + 1)]


def format_ops(ops):
    return ", ".join(f"{id}:{tokens}" for id, tokens in ops) if ops else "-"
