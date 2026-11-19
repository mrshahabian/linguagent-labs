"""Reference answers for 13 - views, copies and dtypes."""

import numpy as np


def is_view_of(child, parent):
    """True if `child` shares memory with `parent`.

    np.shares_memory is the honest answer. The rule of thumb worth knowing:
    BASIC slicing (x[1:3], x[:, 0], x.T, x.reshape(...)) returns a VIEW;
    FANCY indexing (x[[0, 2]], x[x > 0]) returns a COPY.
    """
    return np.shares_memory(child, parent)


def safe_scale(x, factor):
    """Return x * factor without modifying x.

    The trap: `x *= factor` mutates the caller's array in place, and if x is
    itself a view, it mutates whatever it is a view of.
    """
    return x * factor


def scale_in_place(x, factor):
    """Multiply x by factor in place and return None.

    Using the *= form (or out=) avoids allocating a second array - the whole
    reason to do this in a memory-bound loop.
    """
    x *= factor


def zero_negatives(x):
    """Return a copy of x with negative entries replaced by 0."""
    out = x.copy()
    out[out < 0] = 0
    return out


def to_contiguous(x):
    """Return a C-contiguous version of x, copying only if it is not already.

    A transpose is a view with reversed strides, so it is NOT C-contiguous.
    Some libraries silently copy when handed such an array; doing it
    explicitly makes the cost visible.
    """
    return np.ascontiguousarray(x)


def sliding_window_view(x, size):
    """(n,) -> (n - size + 1, size) of overlapping windows, as a VIEW.

    No data is copied: the result reuses x's buffer with a fabricated stride
    pattern. That makes it free to build, but every window aliases the same
    memory - so writing into it corrupts the original.
    """
    if size <= 0 or size > len(x):
        raise ValueError("size must be between 1 and len(x)")
    stride = x.strides[0]
    return np.lib.stride_tricks.as_strided(
        x, shape=(len(x) - size + 1, size), strides=(stride, stride))


def safe_sum(x):
    """Sum x in a dtype wide enough not to overflow.

    np.sum on an int8 array accumulates in int8 and wraps around silently.
    Naming a wider accumulator dtype is the fix - the same reason float32
    tensors are often reduced in float32-with-float64-accumulate on GPUs.
    """
    if np.issubdtype(x.dtype, np.integer):
        return np.sum(x, dtype=np.int64)
    return np.sum(x, dtype=np.float64)


def downcast_safely(x, dtype):
    """Cast x to dtype, raising ValueError if any value would not survive.

    Silent wraparound on downcast is one of the quietest bugs in NumPy:
    np.array([300]).astype(np.uint8) gives 44, no warning.
    """
    info = (np.iinfo(dtype) if np.issubdtype(dtype, np.integer)
            else np.finfo(dtype))
    if np.any(x < info.min) or np.any(x > info.max):
        raise ValueError(f"values do not fit in {np.dtype(dtype).name}")
    return x.astype(dtype)


def memory_bytes(shape, dtype):
    """Bytes an array of this shape and dtype occupies, excluding overhead."""
    count = 1
    for size in shape:
        count *= size
    return count * np.dtype(dtype).itemsize
