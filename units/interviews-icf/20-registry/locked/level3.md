# Package Registry — Level 3

The registry now supports **time**. Timestamps are non-negative integers and
are always supplied by the caller; calls arrive in non-decreasing timestamp
order within a single test, but do not rely on that.

Add timestamp-aware versions of the existing operations:

### `publish_at(timestamp, name, size, ttl=None) -> bool`

Publishes a package at `timestamp`.

- If `ttl` is `None`, the package never expires.
- If `ttl` is an integer, the package is **alive** for timestamps in the
  half-open interval `[timestamp, timestamp + ttl)`. At `timestamp + ttl` it
  is gone.
- Returns `True` if no **alive** package with that name existed at
  `timestamp`; `False` otherwise. Publishing over an expired package counts
  as new and returns `True`.
- Publishing over an alive package overwrites its size and resets its
  lifetime from `timestamp`.

### `get_at(timestamp, name) -> int | None`

Returns the size if the package is alive at `timestamp`, else `None`.

### `delete_at(timestamp, name) -> bool`

Removes the package if it is alive at `timestamp` and returns `True`.
Returns `False` if it is absent or already expired.

### `search_at(timestamp, prefix) -> list[str]`

Same rules as `search`, but only considers packages alive at `timestamp`.

---

**Backwards compatibility.** The Level 1 and Level 2 methods must still work.
Treat each of them as its timestamp-aware equivalent called at
`timestamp = 0` with no TTL.

---

Run `./run 3` until Levels 1–3 pass, then unlock Level 4.
