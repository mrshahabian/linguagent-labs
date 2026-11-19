# 18 — Files, paths and serialization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/python/18-files-and-serialization/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/python/files-and-serialization?utm_source=github&utm_medium=labs)

*~45 minutes · 25 tests*

Every real program eventually reads a config, writes a checkpoint, or logs
to a file it must not corrupt if the process dies mid-write. This module is
the mechanics behind all three: `pathlib` instead of string-joined paths,
`json` for the data, and context managers for the part people get wrong —
making sure cleanup happens whether or not the code in between raised.

Work in `exercises.py`. Check your work with `python -m labkit check 18`, or open the notebook (`lab.ipynb`) in Colab.

Each section below ends with **Test it** — the exact command for just that
piece, run from `units/python/18-files-and-serialization/`.

## What you'll build

| Name | You'll implement | Shows up in real code as |
|---|---|---|
| `read_lines` | `read_lines` | Reading a small file into lines without manual newline stripping |
| `list_python_files` | `list_python_files` | Recursive file discovery — `Path.rglob` instead of `os.walk` |
| `ensure_dir` | `ensure_dir` | The idempotent "make sure this output directory exists" call |
| `save_json`, `load_json` | both | The config/checkpoint-metadata round trip |
| `merge_configs` | `merge_configs` | Layering a user config over defaults — the real "deep merge" |
| `SetEncoder`, `dumps_with_sets` | `SetEncoder.default`, `dumps_with_sets` | Extending `json` for a type it doesn't know natively |
| `AtomicWriter` | `__enter__`, `__exit__` | A file write a reader never sees half-finished |
| `temporary_cwd` | `temporary_cwd` | A `@contextmanager` that guarantees cleanup, not just setup |
| `chunked_lines` | `chunked_lines` | Streaming a file that doesn't fit comfortably in memory |

---

## 1. `pathlib` over string paths

```python
Path(path).read_text().splitlines()
Path(directory).rglob("*.py")
```

`pathlib.Path` replaces `os.path.join`, `os.walk`, and manual
`open()`/`.read()`/`.close()` with methods on an object that represents
the path itself. `rglob("*.py")` is the recursive glob — it walks every
subdirectory looking for matches, the same job `os.walk` plus a filename
filter does, in one call. `read_text()` opens, reads, and closes the file
for you; `.splitlines()` on the result gives lines with no trailing `\n`
to strip, unlike iterating an open file object.

```python
>>> read_lines("notes.txt")
['line1', 'line2', 'line3']
>>> list_python_files("units/python/18-files-and-serialization")
['units/python/18-files-and-serialization/exercises.py', ...]
```

**In `exercises.py`:** `read_lines`, `list_python_files`.

**Test it:** `python3 -m unittest test_exercises.TestReadLines test_exercises.TestListPythonFiles -v`

## 2. Idempotent directory creation

```python
Path(path).mkdir(parents=True, exist_ok=True)
```

`parents=True` is `mkdir -p` — it creates any missing intermediate
directories instead of raising `FileNotFoundError` when a parent is
missing. `exist_ok=True` is what makes calling this twice safe: without
it, a bare `.mkdir()` raises `FileExistsError` the second time, which is
almost never what "make sure this directory exists" actually wants.

```python
>>> ensure_dir("output/run-42/checkpoints")
PosixPath('output/run-42/checkpoints')
>>> ensure_dir("output/run-42/checkpoints")   # calling again is a no-op
PosixPath('output/run-42/checkpoints')
```

**In `exercises.py`:** `ensure_dir`.

**Test it:** `python3 -m unittest test_exercises.TestEnsureDir -v`

## 3. JSON round trips and deep merges

```python
json.dumps(data, indent=2)
json.loads(text)
```

The straightforward half. The interesting exercise is `merge_configs`:

```python
def merge_configs(base, override):
    result = dict(base)
    for key, value in override.items():
        if (key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result
```

`{**base, **override}` only merges **one level deep** — if both have a
`"nested"` key that's itself a dict, the whole nested dict from `override`
replaces the one from `base`, instead of the two merging. That is almost
never what "layer a user config over the defaults" means in practice.
This walks every key that is a dict on **both** sides and recurses, so
only the leaf values that `override` actually specifies get replaced.

```python
>>> base = {"a": 1, "nested": {"x": 1, "y": 2}}
>>> override = {"nested": {"y": 99, "z": 3}, "b": 2}
>>> merge_configs(base, override)
{'a': 1, 'nested': {'x': 1, 'y': 99, 'z': 3}, 'b': 2}
```

**In `exercises.py`:** `save_json`, `load_json`, `merge_configs`.

**Test it:** `python3 -m unittest test_exercises.TestJsonRoundTrip test_exercises.TestMergeConfigs -v`

## 4. Extending `json` for a type it doesn't know

```python
>>> json.dumps({"tags": {"a", "b"}})
Traceback (most recent call last):
    ...
TypeError: Object of type set is not JSON serializable
```

JSON has no set type, so `json.dumps` raises on one. `json.JSONEncoder`
has a documented extension point for exactly this: override `.default()`,
which is called only for objects the base encoder doesn't already know
how to handle — everything already-serialisable (`dict`, `list`, `str`,
numbers, `bool`, `None`) still goes through the normal fast path
untouched.

```python
class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return sorted(obj)
        return super().default(obj)
```

The `super().default(obj)` fallback matters: it is what produces the
correct `TypeError` for a *genuinely* unsupported type (say, a socket),
instead of `.default()` silently returning `None` for it.

```python
>>> dumps_with_sets({"tags": {"b", "a", "c"}})
'{"tags": ["a", "b", "c"]}'
```

**In `exercises.py`:** `SetEncoder.default`, `dumps_with_sets`.

**Test it:** `python3 -m unittest test_exercises.TestSetEncoder -v`

## 5. Atomic writes: a class-based context manager

```python
class AtomicWriter:
    def __enter__(self):
        self._file = open(self.tmp_path, "w")
        return self._file

    def __exit__(self, exc_type, exc_value, traceback):
        self._file.close()
        if exc_type is None:
            os.replace(self.tmp_path, self.path)
        else:
            self.tmp_path.unlink(missing_ok=True)
        return False
```

Writing straight to the target path leaves a window where a concurrent
reader (or a crash mid-write) sees a **partial file**. `AtomicWriter`
writes to a sibling `.tmp` file instead, and only promotes it with
`os.replace()` — an atomic rename on POSIX and modern Windows — once the
`with` block exits cleanly. `__exit__`'s three arguments tell you whether
an exception happened; returning `False` (or nothing) lets that exception
keep propagating instead of being swallowed, which matters here since a
failed write should still fail loudly, just without corrupting the
target.

```python
>>> with AtomicWriter("checkpoint.json") as f:
...     f.write('{"step": 1000}')
>>> # if the block above raised instead, checkpoint.json is untouched
```

**In `exercises.py`:** `AtomicWriter.__enter__`, `AtomicWriter.__exit__`.

**Test it:** `python3 -m unittest test_exercises.TestAtomicWriter -v`

## 6. `@contextmanager`: cleanup that survives an exception

```python
@contextlib.contextmanager
def temporary_cwd(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)
```

Same idea as `AtomicWriter`, generator-decorator style instead of a
class: code before `yield` is setup (`__enter__`), code after is teardown
(`__exit__`). The `finally` is what makes the restore unconditional — if
the code inside the `with` block raises, `os.chdir(previous)` still runs
before the exception propagates out. Skipping the `try/finally` and just
writing `os.chdir(previous)` after the `yield` would leave the process in
the wrong directory forever the first time a caller's code inside the
block raises.

```python
>>> with temporary_cwd("/tmp"):
...     os.getcwd()
'/tmp'
>>> os.getcwd()   # back to normal after the block
'/original/dir'
```

**In `exercises.py`:** `temporary_cwd`.

**Test it:** `python3 -m unittest test_exercises.TestTemporaryCwd -v`

## 7. Streaming a file that doesn't fit in memory

```python
def chunked_lines(path, chunk_size):
    chunk = []
    with open(path) as f:
        for line in f:
            chunk.append(line.rstrip("\n"))
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
    if chunk:
        yield chunk
```

`path.read_text().splitlines()` — the approach from §1 — loads the
**entire file** before you can do anything with it. Fine for a config,
wrong for a log file too large to fit in memory. Iterating the open file
object one line at a time keeps memory proportional to `chunk_size`, not
to the file's size — the same shape as the batching exercise back in
module 15, applied to disk instead of a queue.

```python
>>> list(chunked_lines("access.log", chunk_size=3))
[['line1', 'line2', 'line3'], ['line4', 'line5', 'line6'], ['line7']]
```

**In `exercises.py`:** `chunked_lines`.

**Test it:** `python3 -m unittest test_exercises.TestChunkedLines -v`

---

## The exercises

| Function / class | What it teaches |
|---|---|
| `read_lines`, `list_python_files` | `pathlib` basics: `read_text`, `rglob` |
| `ensure_dir` | `mkdir(parents=True, exist_ok=True)` |
| `save_json`, `load_json` | The JSON round trip |
| `merge_configs` | A real deep merge vs. `{**a, **b}`'s one-level merge |
| `SetEncoder`, `dumps_with_sets` | Extending `json.JSONEncoder.default` |
| `AtomicWriter` | A class-based context manager; atomic rename |
| `temporary_cwd` | `@contextmanager`; `finally` guarantees teardown |
| `chunked_lines` | Streaming reads with bounded memory |

Every function in this table has a stub in `exercises.py` and test
coverage in `test_exercises.py` — a clean 1:1 mapping.

## Check your understanding

1. **Why does `{**base, **override}` fall short as a config-merging
   strategy?**
   <details><summary>Answer</summary>It only merges one level deep. If
   both dicts have a key whose value is itself a dict, the nested dict
   from <code>override</code> completely replaces the one from
   <code>base</code> rather than merging with it — any keys only present
   in <code>base</code>'s nested dict are silently lost.</details>

2. **What does `os.replace()` guarantee that writing directly to the
   target path does not?**
   <details><summary>Answer</summary>On POSIX (and modern Windows), a
   rename onto an existing path is a single atomic filesystem operation
   — any concurrent reader sees either the old complete file or the new
   complete file, never something half-written. Writing directly to the
   target leaves a window where a crash or a concurrent read catches the
   file mid-write.</details>

3. **Why does `AtomicWriter.__exit__` check `exc_type` before deciding
   whether to promote the temp file?**
   <details><summary>Answer</summary>If an exception happened inside the
   <code>with</code> block, the data written so far is presumed
   incomplete or invalid, so the temp file is discarded instead of being
   renamed into place — the target path is left exactly as it was
   before the write was attempted.</details>

4. **In a `@contextmanager` generator, what's the difference between
   putting the teardown code after `yield` versus after `yield` inside a
   `finally` block?**
   <details><summary>Answer</summary>Without <code>finally</code>, the
   teardown only runs if the code inside the <code>with</code> block
   completes without raising — an exception there skips straight past
   it, leaving the resource (a changed directory, an open connection,
   held lock) never cleaned up. <code>finally</code> makes the teardown
   run unconditionally, exception or not.</details>

5. **Why does `chunked_lines` iterate the open file object instead of
   calling `.read_text().splitlines()` and slicing the result?**
   <details><summary>Answer</summary><code>read_text()</code> loads the
   entire file into memory before any chunk can be produced, so peak
   memory use scales with the whole file's size. Iterating the file
   object line by line only ever holds one chunk's worth of lines at a
   time, which is the difference between working and OOMing on a file
   too large to fit in memory.</details>

## Interview questions this module answers

- *How would you merge a user config over a defaults dict, one level
  deep vs. properly nested?*
- *Why is `os.replace()` safer for writing output files than writing
  directly to the target path?*
- *What does `finally` guarantee in a context manager that comes after
  the `yield` — and what happens without it?*
- *This log file is bigger than memory — how do you process it?*
- *How do you make `json.dumps` handle a type it doesn't support
  natively?*
