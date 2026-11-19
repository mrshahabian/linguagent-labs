# Package Registry — Level 1

You are implementing an in-memory package registry. All state lives in the
`Registry` class in `registry.py`. No external libraries are needed.

Implement the following methods.

### `publish(name, size) -> bool`

Adds a package with the given name and size (an integer).

- Returns `True` if no package with that name existed.
- Returns `False` if a package with that name already existed. The size is
  overwritten with the new value in either case.

### `get(name) -> int | None`

Returns the size of the package, or `None` if no such package exists.

### `delete(name) -> bool`

Removes the package.

- Returns `True` if the package existed and was removed.
- Returns `False` if it did not exist.

---

Run `./run 1` until all Level 1 tests pass, then unlock Level 2.
