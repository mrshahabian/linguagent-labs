# Level 2 — histogram, outlier station, reproducible audit sample

Write your answer in `levels/level2.py`. Check with `./run 2`. Uses the same
matrix as level 1 — `temperature_histogram` and `most_anomalous_station`
both take the `(stations, days)` array directly, `NaN`s and all.

## The spec

**`temperature_histogram(matrix, low, high) -> np.ndarray`**

Take every valid (non-`NaN`) reading, floor it to an integer degree, and
count it into bin `[low, low+1)`, ..., `[high-1, high)`. A reading outside
`[low, high)` is dropped, not clamped. Return an array of length
`high - low`.

**`empty_bins(histogram) -> np.ndarray`**

The indices, ascending, of every bin with zero readings.

**`most_anomalous_station(matrix, station_ids) -> str`**

The station whose mean valid reading is furthest — by total absolute
difference from every *other* station's mean — from the rest of the
network. Ties go to whichever station comes first in `station_ids`. A
station with zero valid readings has mean `0.0` (level 1's rule, reused).

**`audit_sample(station_ids, k, seed) -> list[str]`**

`k` station ids chosen without replacement, in ascending station-index
order, deterministically for a given `seed` — same `(station_ids, k, seed)`
always returns the same list. Use `np.random.RandomState(seed)`, not the
global `np.random`.

## Worked example

Same matrix as level 1. `temperature_histogram(matrix, 0, 15)`:

```
degree:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14
count:   0  1  3  1  0  1  0  1  0  1  1  1  1  1  0
```

(`south` reports `2.0` three times — degree `2`'s count of 3 is not a typo.)
`empty_bins` on that array: `[0, 4, 6, 8, 14]`.

Means: `north` 4.0, `south` 2.0, `east` 0.0 (no valid readings), `west` 11.0.
`west`'s total deviation from the other three (`7 + 9 + 11 = 27`) is the
largest, so `most_anomalous_station` returns `"west"` — note `east`'s
fabricated `0.0` mean pulls on this calculation too, exactly as it would if
it were a real reading of 0°.

## Deriving the code

**`temperature_histogram`.** Mask `NaN` (same rule as level 1), restrict to
`[low, high)`, then shift down by `low` before counting — `bincount` needs
non-negative integers and doesn't know what `low` means:

```python
valid = matrix[~np.isnan(matrix)]
in_range = valid[(valid >= low) & (valid < high)]
bins = np.floor(in_range).astype(int) - low
return np.bincount(bins, minlength=high - low)
```

`minlength=high - low` matters even though it looks redundant: without it,
`bincount` only returns as many entries as the *largest* bin index plus one,
so if nothing lands in the top few bins the array comes back shorter than
the spec's `high - low`, and `empty_bins` silently can't see those bins at
all.

**`empty_bins`.** `np.where(histogram == 0)[0]` (or `np.nonzero` — see
`02-prereq-numpy.md` for why they're interchangeable here). The `[0]` is not
optional: `np.where(...)` on its own is a one-tuple, and `assertEqual` against
a plain list will fail on the wrapper, not the values.

**`most_anomalous_station`.** One mean per station (masking `NaN`, defaulting
to `0.0` for an empty row — the same branch as level 1), then broadcast:

```python
diffs = np.abs(means[:, None] - means[None, :])
total_deviation = np.sum(diffs, axis=1)
winner_row = np.nonzero(total_deviation == np.max(total_deviation))[0][0]
return station_ids[int(winner_row)]
```

`axis=1` sums each *row* of the pairwise grid — row `i` is "station `i`
against everyone" — so you get one number per station without a Python loop
over pairs. The final `[0][0]` picks the first (lowest-index) station
achieving the max, which is the tie-break rule.

**`audit_sample`.** `RandomState(seed).choice(len(station_ids), size=k,
replace=False)` returns `k` distinct row indices in an arbitrary order —
sort them yourself, the spec asks for ascending station-index order, not
generation order.

## Pitfalls

- **`np.random.seed(seed)` then `np.random.choice(...)`.** Reproducible
  *within one call*, but reseeds the global generator every level 2 call
  makes — if `audit_sample` runs more than once in a process (as it does in
  the tests), later calls are affected by whatever ran between them. Use a
  fresh `RandomState(seed)` per call instead.
- **Forgetting `minlength`.** Your histogram is the right shape on the
  worked example (where the top bins aren't empty) and silently too short on
  any input where they are — an easy thing to not notice until `empty_bins`
  or a length assertion fails somewhere that looks unrelated.
- **Clamping instead of dropping out-of-range readings.** "Outside
  `[low, high)`" means excluded from the count entirely, not folded into the
  boundary bin.
- **Looping over station pairs in Python for `most_anomalous_station`.**
  Works, but it's the thing this level exists to make you not do — if you
  find yourself writing a nested `for` loop here, look at `.T`, `[:, None]`
  and `[None, :]` in `02-prereq-numpy.md` again.
