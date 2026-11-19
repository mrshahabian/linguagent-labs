"""18 - files, paths and serialization

Implement each function so the tests in test_exercises.py pass. Read
README.md first; every exercise below is covered there.

    ../run 18            test your work
    ../run 18 --check    watch the reference solutions pass

The docstrings are the specification. Standard library only.
"""

import contextlib
import json
import os
from pathlib import Path


def read_lines(path):
    """Return the file's lines as a list of strings, with no trailing newline.

    Path.read_text().splitlines() reads the whole file at once and never
    leaves a dangling '\\n' on each line the way iterating the open file
    object does - no need for `.rstrip("\\n")` on every line yourself.
    """
    raise NotImplementedError("your code here")


def list_python_files(directory):
    """Return every .py file under directory, recursively, sorted by path.

    Path.rglob("*.py") walks subdirectories automatically - the
    recursive-glob equivalent of os.walk plus a filename filter, without
    the manual os.path.join bookkeeping. Sorting makes the result
    deterministic; filesystem iteration order is not guaranteed.
    """
    raise NotImplementedError("your code here")


def ensure_dir(path):
    """Create path (and any missing parents) if it doesn't exist; return it as a Path.

    parents=True creates intermediate directories the way `mkdir -p`
    does. exist_ok=True makes this idempotent - calling it twice is not
    an error, unlike a bare Path.mkdir() which raises FileExistsError the
    second time.
    """
    raise NotImplementedError("your code here")


def save_json(data, path):
    """Write data to path as JSON, creating the file (or overwriting it)."""
    raise NotImplementedError("your code here")


def load_json(path):
    """Read and parse the JSON file at path."""
    raise NotImplementedError("your code here")


def merge_configs(base, override):
    """Recursively merge override into base; override wins on every leaf.

    A plain `{**base, **override}` only merges one level deep - a nested
    dict in override completely replaces the matching one in base instead
    of merging with it. This walks matching keys that are dicts on BOTH
    sides and merges those recursively too, which is the behaviour people
    actually mean by "merge these two config files."
    """
    raise NotImplementedError("your code here")


class SetEncoder(json.JSONEncoder):
    """A JSONEncoder that can serialise sets by turning them into sorted lists.

    json.dumps raises TypeError on a bare set - JSON has no set type.
    Overriding .default() is the documented extension point: it is only
    called for objects the base encoder does not already know how to
    handle, so dict/list/str/int/float/bool/None all still go through the
    normal fast path.
    """

    def default(self, obj):
        raise NotImplementedError("your code here")


def dumps_with_sets(data):
    """json.dumps, but able to serialise sets via SetEncoder."""
    raise NotImplementedError("your code here")


class AtomicWriter:
    """Write a file atomically: readers never see a partially-written file.

    Writes go to a sibling `<name>.tmp` file first. On a clean exit,
    os.replace() atomically swaps it into place - on POSIX and modern
    Windows this rename is a single filesystem operation, so a concurrent
    reader sees either the old complete file or the new complete file,
    never a half-written one. On an exception, the temp file is removed
    instead of being promoted, so the target path is left untouched.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        self._file = None

    def __enter__(self):
        raise NotImplementedError("your code here")

    def __exit__(self, exc_type, exc_value, traceback):
        raise NotImplementedError("your code here")


@contextlib.contextmanager
def temporary_cwd(path):
    """Change the working directory for the duration of the `with` block.

    The restore lives in `finally`, so the original directory comes back
    even if the block raises - a context manager's whole job is exactly
    this pairing of setup and guaranteed teardown, generalised from
    open()/close() to any resource, here the process's cwd.
    """
    raise NotImplementedError("your code here")


def chunked_lines(path, chunk_size):
    """Yield the file's lines in lists of at most chunk_size, streaming.

    Reads line by line rather than calling read_text() and slicing the
    result, so memory use stays proportional to chunk_size instead of the
    whole file - the difference between working on a log file that fits
    in memory and one that does not.
    """
    raise NotImplementedError("your code here")
