# Package Registry — Level 4

All previous behaviour must continue to work.

The registry can now snapshot and roll back its contents.

### `backup(timestamp) -> int`

Saves the state of every package **alive** at `timestamp`, and returns the
number of packages saved.

For each saved package, the **remaining lifetime** is what is recorded — not
the absolute expiry time. A package with no TTL is saved as never-expiring.

Multiple backups may be taken at different timestamps. A backup at a
timestamp that already has one replaces it.

### `restore(timestamp, timestamp_to_restore) -> None`

Replaces the current set of packages with the snapshot taken at exactly
`timestamp_to_restore`. You may assume such a backup exists.

- Packages that were deleted or expired since the backup come back.
- Packages published since the backup are discarded.
- Sizes revert to the saved values.
- Each restored package becomes alive at `timestamp`, with the remaining
  lifetime it had when the backup was taken. So a package with 6 units of
  life left when backed up, restored at timestamp 100, is alive across
  `[100, 106)`.
- Backups themselves are not affected by a restore.

---

Run `./run 4` until Levels 1–4 pass, then unlock Level 5.
