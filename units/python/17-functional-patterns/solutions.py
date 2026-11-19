"""Reference answers for 17 - functional patterns: itertools, functools, decorators."""

import functools
import itertools
import operator


def flatten(iterables):
    """Flatten one level of nested iterables into a single list.

    itertools.chain.from_iterable is lazy and does not build the
    intermediate flattened structure in memory the way
    `sum(iterables, [])` or nested list comprehensions do - it just
    chains iterators together.
    """
    return list(itertools.chain.from_iterable(iterables))


def group_adjacent(items):
    """Group CONSECUTIVE equal items, as (value, list) pairs.

    itertools.groupby only merges a run while the key keeps matching the
    PREVIOUS item - it does not look ahead across the whole sequence. So
    ["a", "a", "b", "a"] produces three groups, not two: the second "a"
    starts a new group because a "b" came between it and the first run.
    This is the single most common groupby mistake - forgetting the input
    must already be sorted/adjacent for the grouping you actually want.
    """
    return [(key, list(group)) for key, group in itertools.groupby(items)]


def group_all(items, key):
    """Group ALL items sharing a key, regardless of position.

    Unlike group_adjacent, this sorts by `key` first so every matching
    item becomes adjacent before groupby ever sees it - the standard fix
    for groupby's "consecutive only" behaviour. The trade-off: sorting is
    O(n log n) and the original relative order within a group is not
    preserved beyond what the sort itself does.
    """
    ordered = sorted(items, key=key)
    return {k: list(group) for k, group in itertools.groupby(ordered, key=key)}


def all_pairs(a, b):
    """Every (x, y) pair from the Cartesian product of a and b, in order.

    itertools.product generates this lazily instead of via a nested
    list comprehension - useful the moment either input is itself lazy
    or large enough that materialising both fully first would waste
    memory.
    """
    return list(itertools.product(a, b))


def running_total(numbers):
    """The cumulative sum after each number, as a list.

    itertools.accumulate defaults to addition but takes any two-argument
    function - accumulate(numbers, operator.mul) gives a running product
    with the exact same call shape.
    """
    return list(itertools.accumulate(numbers))


def product_of(numbers):
    """Multiply every number together; 1 for an empty sequence.

    functools.reduce folds a binary function across an iterable, keeping
    a running result - here operator.mul instead of a lambda, since
    `operator.mul` is both faster and clearer than `lambda a, b: a * b`.
    The initial value of 1 is what makes an empty input well-defined
    instead of raising TypeError.
    """
    return functools.reduce(operator.mul, numbers, 1)


def make_multiplier(n):
    """Return a callable that multiplies its argument by n.

    functools.partial pre-binds the first argument of operator.mul
    without writing a wrapping def or lambda - it returns a new callable
    object, not a closure, which is why partial objects are picklable
    where equivalent closures often are not.
    """
    return functools.partial(operator.mul, n)


@functools.singledispatch
def describe(obj):
    """Describe obj's type, dispatching on its runtime type.

    The base @singledispatch function is the fallback for any type with
    no registered implementation - this one covers "everything else."
    Each @describe.register below adds a specialised implementation
    without an if/isinstance chain inside a single function body, so
    adding a new type later means adding a new registration, not editing
    this one.
    """
    return f"a {type(obj).__name__}"


@describe.register
def _(obj: int):
    return f"the integer {obj}"


@describe.register
def _(obj: str):
    return f"a string of length {len(obj)}"


@describe.register
def _(obj: list):
    return f"a list of {len(obj)} items"


def retry(times):
    """A parameterised decorator: retry the wrapped call up to `times` times.

    `retry(3)` must itself return the actual decorator, because a
    decorator with arguments is really a decorator FACTORY - `@retry(3)`
    calls retry(3) first (giving back a decorator), then applies that
    decorator to the function, all before the function is ever called.
    functools.wraps preserves the wrapped function's __name__ and
    __doc__, which matters for introspection and for stacking with other
    decorators that also read those attributes.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(1, times + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as error:  # noqa: BLE001 - retry is generic on purpose
                    last_error = error
            raise last_error
        return wrapper
    return decorator


def call_counter(func):
    """A plain decorator (no arguments) that counts calls on func.calls.

    Unlike `retry`, this decorator takes the function directly - no outer
    factory layer, because it needs no configuration. The count lives on
    the wrapper function object itself, accessible as `wrapped.calls`
    after decoration.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        return func(*args, **kwargs)
    wrapper.calls = 0
    return wrapper
