"""Warm-up drills for the scheduler exercise.

Implement each function. Run ./drills/drill to check, or ./drills/drill A for
one block. Standard library only.

These are not puzzles. Every one of them is a line you will write inside
plan_step, extracted so you can practise it without the surrounding problem.
Aim for the budgets in the block headers; if a block takes more than double
its budget, that is your weak spot.

Blocks A-C are the prerequisites. Blocks D-F are the scheduler's own logic,
pulled out of context. Block G is the session's warm-up questions.
"""

from dataclasses import dataclass, field


# ------------------------------------------------------------------ Block A
# Dataclasses and immutable views.  Budget: 8 min.

@dataclass
class Plan:
    """A stripped-down StepPlan.

    Give it three list fields - preempt, decode, prefill - that each default
    to a NEW empty list per instance. Getting this wrong (with `= []`) is the
    classic mutable-default bug; dataclasses will refuse to let you.
    """
    # TODO: your fields here


def collect_ids(requests):
    """Return the ids of `requests` as a list, without mutating the input.

    `requests` is a tuple of frozen objects with an `.id`, like view.admitted.
    collect_ids((R("a"), R("b"))) -> ["a", "b"]
    """


def newest_first(admitted):
    """Return `admitted` reversed as a list, without mutating it.

    admitted is in admission order (earliest first), so this is the Level 3b
    victim order.
    newest_first((a, b, c)) -> [c, b, a]
    """


def split_by_prefill(admitted):
    """Return (ready, started): those with prefill_remaining == 0, and those
    with prefill_remaining > 0, each in the original order.

    This is the Level 2 partition.
    """


# ------------------------------------------------------------------ Block B
# Compound sort keys.  Budget: 12 min. Over-practise this block.

def by_total_desc(pairs):
    """Sort (name, total) pairs by total descending, ties by name ascending.
    by_total_desc([("b", 5), ("a", 5), ("c", 9)]) -> [("c", 9), ("a", 5), ("b", 5)]
    """


def top_k_formatted(totals, k):
    """Return at most k strings "name(total)", total desc, ties name ascending.
    top_k_formatted({"a": 1, "b": 2, "c": 0}, 5) -> ["b(2)", "a(1)", "c(0)"]
    """


def by_priority_desc(requests):
    """Return `requests` sorted by priority descending, keeping the original
    relative order within each priority tier.

    This is the Level 3c admission order - and it needs exactly one sort key,
    because Python's sort is stable and the input is already in arrival order.
    """


def lowest_priority_newest(admitted):
    """Return the lowest-priority request in `admitted`, breaking ties towards
    the one admitted most recently.

    `admitted` is in admission order, so later index = more recent. This is the
    Level 3c victim rule.
    """


# ------------------------------------------------------------------ Block C
# Budget loops.  Budget: 12 min.

def take_while_fits(costs, budget):
    """Return the longest PREFIX of `costs` whose sum is <= budget.

    Stop at the first item that doesn't fit - do not skip ahead to a smaller
    later one. This is the Level 1 fairness rule.
    take_while_fits([4, 8, 1], 10) -> [4]
    take_while_fits([4, 3, 1], 10) -> [4, 3, 1]
    """


def take_greedily(costs, budget):
    """Return every item that fits, skipping ones that don't.

    The WRONG answer for this exercise - written here so you can feel the
    difference from take_while_fits.
    take_greedily([4, 8, 1], 10) -> [4, 1]
    """


def chunk_to_budget(needs, budget):
    """Spend `budget` across `needs` in order, taking min(remaining, need) each
    time, and stopping when the budget runs out. Return the amounts taken.

    This is Level 2's chunked prefill.
    chunk_to_budget([2, 13, 3], 8) -> [2, 6]
    chunk_to_budget([2, 3], 8) -> [2, 3]
    """


def first_fit_index(costs, budget):
    """Return the index of the first item that does NOT fit within budget when
    taken in order, or None if they all fit.
    first_fit_index([4, 3, 9], 10) -> 2
    """


# ------------------------------------------------------------------ Block D
# KV arithmetic.  Budget: 10 min.

def peak_slots(prompt_len, max_tokens):
    """Slots held on the request's final step (Level 3a).
    peak_slots(2, 2) -> 3
    peak_slots(7, 4) -> 10
    """


def slots_after_decodes(prompt_len, decodes_done):
    """Slots held after a completed prefill plus `decodes_done` decodes.
    slots_after_decodes(2, 0) -> 2
    slots_after_decodes(2, 2) -> 4
    """


def growth_need(kv_values):
    """Total slots needed for every one of these requests to decode this step.

    Each decoding request needs one MORE slot than it currently holds.
    growth_need([2, 4]) -> 8
    """


def evict_until_fits(kv_values, capacity):
    """Repeatedly drop the LAST element until the remaining ones can all decode
    within `capacity`. Return (kept, dropped_indices) with dropped_indices in
    the order they were dropped.

    This is Level 3b's growth-preemption loop.
    evict_until_fits([2, 4], 6) -> ([2], [1])
    evict_until_fits([2, 4], 8) -> ([2, 4], [])
    """


def replay_cost(prompt_len, tokens_generated):
    """prefill_remaining for a preempted request re-entering the queue.
    replay_cost(4, 1) -> 5
    """


# ------------------------------------------------------------------ Block E
# Eviction accounting.  Budget: 12 min. This is Level 3c's hard part.

def evictable(admitted, candidate_priority):
    """Those admitted requests with STRICTLY lower priority than the candidate.

    Note "strictly" - an equal-priority request is never evictable.
    """


def enough_to_free(kv_values, shortfall):
    """True if evicting all of these (each freeing kv + 1 slots) would cover
    `shortfall`. Level 3c's all-or-nothing check.
    enough_to_free([3, 3], 4) -> True
    enough_to_free([3], 8) -> False
    """


def evict_for_shortfall(kv_values, shortfall):
    """Evict from the front of `kv_values` (already in victim order), stopping
    as soon as the freed total covers `shortfall`. Each eviction frees kv + 1.

    Return (indices_evicted, total_freed). Evict as FEW as possible.
    evict_for_shortfall([3, 3], 4) -> ([0], 4)
    evict_for_shortfall([1, 3], 4) -> ([0, 1], 6)
    """


# ------------------------------------------------------------------ Block F
# Small class design + delegation.  Budget: 12 min.

class Ledger:
    """A tiny store you will build a hundred variations of.

    add(name, amount)        -> new total for name; creates at 0 if absent
    total(name)              -> current total, or None if name unknown
    remove(name)             -> True if it existed and was removed, else False
    top(n)                   -> at most n strings "name(total)", total desc,
                                ties by name ascending
    add_at(ts, name, amount) -> same as add, but records ts as the last touched
                                time for name
    last_touched(name)       -> the ts most recently passed to add_at for name,
                                or None if it has only ever seen add()

    add(name, amount) must behave exactly like add_at(0, name, amount) except
    that it does not update last_touched. Do not duplicate the logic.
    """


# ------------------------------------------------------------------ Block G
# Formatting, parsing, transformation.  Budget: 8 min.

def rebase(saved, now):
    """saved is {name: (value, remaining_or_None)} taken at some earlier time.

    Return {name: (value, expiry_or_None)} where expiry = now + remaining.
    A remaining of None stays None.
    """


def pyramid(n):
    """Return the n rows of an ASCII pyramid as a list of strings.

    Row i has 2i-1 asterisks, centred: left-padded so the base row has none.
    pyramid(3) -> ["  *", " ***", "*****"]
    """


def format_ops(ops):
    """Render prefill ops as "ID:tokens", joined by ", ", or "-" when empty.
    format_ops([("A", 6), ("B", 3)]) -> "A:6, B:3"
    format_ops([]) -> "-"
    """
