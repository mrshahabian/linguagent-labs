# Package Registry — Level 2

All Level 1 behaviour must continue to work.

### `search(prefix) -> list[str]`

Returns **at most 5** packages whose name starts with `prefix`.

- Ordered by size, largest first.
- Ties broken by name, ascending (standard string ordering).
- Each entry is formatted as `"name(size)"` — for example `"web-utils(500)"`.
- Returns an empty list if nothing matches.

### `total_size() -> int`

Returns the sum of the sizes of all packages currently in the registry.
Returns `0` if the registry is empty.

---

Run `./run 2` until all Level 1 and Level 2 tests pass, then unlock Level 3.
