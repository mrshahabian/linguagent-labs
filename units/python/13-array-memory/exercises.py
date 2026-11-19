"""13 - views, copies and dtypes

Implement each function so the tests in test_exercises.py pass. Read
README.md first; every exercise below is covered there.

    ../run 13            test your work
    ../run 13 --check    watch the reference solutions pass

The docstrings are the specification. Standard library only.
"""

import numpy as np


def is_view_of(child, parent):
    """True if `child` shares memory with `parent`.

    np.shares_memory is the honest answer. The rule of thumb worth knowing:
    BASIC slicing (x[1:3], x[:, 0], x.T, x.reshape(...)) returns a VIEW;
    FANCY indexing (x[[0, 2]], x[x > 0]) returns a COPY.
    """
    raise NotImplementedError("your code here")


def safe_scale(x, factor):
    """Return x * factor without modifying x.

    The trap: `x *= factor` mutates the caller's array in place, and if x is
    itself a view, it mutates whatever it is a view of.
    """
    raise NotImplementedError("your code here")


def scale_in_place(x, factor):
    """Multiply x by factor in place and return None.

    Using the *= form (or out=) avoids allocating a second array - the whole
    reason to do this in a memory-bound loop.
    """
    raise NotImplementedError("your code here")


def zero_negatives(x):
    """Return a copy of x with negative entries replaced by 0."""
    raise NotImplementedError("your code here")


def to_contiguous(x):
    """Return a C-contiguous version of x, copying only if it is not already.

    A transpose is a view with reversed strides, so it is NOT C-contiguous.
    Some libraries silently copy when handed such an array; doing it
    explicitly makes the cost visible.
    """
    raise NotImplementedError("your code here")


def sliding_window_view(x, size):
    """(n,) -> (n - size + 1, size) of overlapping windows, as a VIEW.

    No data is copied: the result reuses x's buffer with a fabricated stride
    pattern. That makes it free to build, but every window aliases the same
    memory - so writing into it corrupts the original.
    """
    raise NotImplementedError("your code here")


def safe_sum(x):
    """Sum x in a dtype wide enough not to overflow.

    np.sum on an int8 array accumulates in int8 and wraps around silently.
    Naming a wider accumulator dtype is the fix - the same reason float32
    tensors are often reduced in float32-with-float64-accumulate on GPUs.
    """
    raise NotImplementedError("your code here")


def downcast_safely(x, dtype):
    """Cast x to dtype, raising ValueError if any value would not survive.

    Silent wraparound on downcast is one of the quietest bugs in NumPy:
    np.array([300]).astype(np.uint8) gives 44, no warning.
    """
    raise NotImplementedError("your code here")


def memory_bytes(shape, dtype):
    """Bytes an array of this shape and dtype occupies, excluding overhead."""
    raise NotImplementedError("your code here")
