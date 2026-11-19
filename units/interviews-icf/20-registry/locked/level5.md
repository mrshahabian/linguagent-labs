# Package Registry — Level 5

All previous behaviour must continue to work.

The registry now tracks downloads.

### `download_at(timestamp, name) -> int | None`

Records one download of the package.

- If the package is alive at `timestamp`, increments its download count and
  returns the **new total**.
- If the package is absent or expired, returns `None` and records nothing.

Download counts are attached to the **name**, not to any particular
publication of it. They survive deletion, expiry, republication, and
`restore`. Nothing ever decreases a download count.

### `top_downloaded(n) -> list[str]`

Returns at most `n` names, ordered by download count, highest first, ties
broken by name ascending.

- Names with zero downloads are excluded entirely.
- Each entry is formatted as `"name(count)"` — for example `"web-utils(12)"`.
- Names appear regardless of whether the package is currently alive.

---

Run `./run 5`. When Levels 1–5 pass, you are done — stop the clock.
