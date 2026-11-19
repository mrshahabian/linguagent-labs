# 63 — A streaming log pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/interview-drills/63-log-pipeline/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/interview-drills/log-pipeline?utm_source=github&utm_medium=labs)

*~45 minutes · 21 tests · Interview*

In production systems, logs are the primary source of truth for system health. A single high-traffic service can easily generate gigabytes of logs per hour. If you attempt to load these logs into memory all at once—for instance, by reading a file into a list of strings—your application will quickly run out of memory and crash.

To process large datasets efficiently, we must build a **streaming pipeline**. Using Python's generators and lazy evaluation, we can parse, filter, aggregate, and analyze logs of arbitrary size using a constant, minimal memory footprint. This is the foundation of modern telemetry systems like Fluentd, Logstash, and Prometheus.

This module covers the core patterns of streaming data processing: lazy parsing, pipeline chaining, time-bucketed aggregation, and sliding-window alerting. These techniques are highly prized in software engineering and system design interviews, where candidates are frequently asked to process massive datasets under strict memory constraints.

## What you'll build

Six components in `exercises.py`, in this order. Keep this table open as you go — each section below tells you which row it's building:

| Function / class | You'll implement | Shows up in real code as |
|---|---|---|
| `LogRecord` & `parse_line` | Parsing raw string logs into structured dataclasses, handling malformed lines gracefully | Log forwarders (Fluentd, Logstash) parsing raw text into structured JSON |
| `records` | A generator yielding valid `LogRecord`s from an iterable of lines | Streaming file readers or socket consumers that don't load the whole file into memory |
| `filter_level` | A generator filtering records by severity level | Log routing and filtering rules in telemetry pipelines |
| `errors_per_minute` | Aggregating error counts bucketed by 1-minute intervals | Time-series aggregation engines (Prometheus, InfluxDB) |
| `top_endpoints` | Finding the most frequent endpoints in a stream | Analytics dashboards and rate-limiting monitors |
| `error_rate_alerts` | A sliding-window generator alerting when error rate exceeds a threshold | Real-time alerting systems (PagerDuty, Datadog monitors) |

---

## 1. Parsing Raw Logs with Dataclasses and Regular Expressions

```python
>>> from dataclasses import dataclass
>>> import re
>>> 
>>> @dataclass(frozen=True)
... class SimpleRecord:
...     level: str
...     message: str
...
>>> pattern = re.compile(r"^(\w+): (.+)$")
>>> match = pattern.match("INFO: System boot")
>>> if match:
...     level, msg = match.groups()
...     record = SimpleRecord(level, msg)
...
>>> record
SimpleRecord(level='INFO', message='System boot')
```

Using a `dataclass` with `frozen=True` provides a lightweight, immutable, and type-safe representation of our structured data. In real-world log processing, raw log lines are strings that must be parsed using regular expressions. 

Our target log format is:
`[YYYY-MM-DD HH:MM:SS] LEVEL ENDPOINT STATUS`

For example:
`[2023-10-27 12:34:56] INFO /index.html 200`

**The Pitfall:** Real-world log files are messy. They contain partial writes, stack traces, and malformed lines. If your parser raises an unhandled exception on a malformed line, the entire long-running pipeline will crash. A robust parser must catch parsing errors and return `None` (or a specialized null object) to let the pipeline discard junk and keep running.

**In `exercises.py`:** Implement the `LogRecord` dataclass and the `parse_line` function. If the line does not match the expected format or contains an invalid date/time, return `None`.

**Test it:** `python3 -m unittest test_exercises.TestParseLine -v`

---

## 2. Lazy Streaming with Generators

```python
>>> def count_up(n):
...     for i in range(n):
...         yield i
...
>>> generator = count_up(3)
>>> next(generator)
0
>>> next(generator)
1
```

Generators allow us to process data lazily. Instead of reading all lines from a file into a list, we can process them one line at a time. The `yield` keyword pauses the function's execution and returns a value to the caller. When the caller asks for the next value, the function resumes exactly where it left off.

**The Pitfall:** Materializing the generator. It is incredibly easy to accidentally destroy the memory benefits of generators by converting them to lists. For example, writing `list(lines)` inside your generator loads the entire file into memory, defeating the purpose of lazy evaluation. Always iterate over the input stream lazily.

**In `exercises.py`:** Implement the `records` generator. It should take an iterable of raw string lines, parse each line using `parse_line`, and yield only the valid `LogRecord` objects, skipping any malformed lines.

**Test it:** `python3 -m unittest test_exercises.TestRecords -v`

---

## 3. Chaining Generators for Filtering

```python
>>> def evens(numbers):
...     for n in numbers:
...         if n % 2 == 0:
...             yield n
...
>>> list(evens(count_up(5)))
[0, 2, 4]
```

Generators can be chained together to form a processing pipeline, much like Unix pipes (`cat file | grep ERROR`). Each step in the pipeline takes an iterator, performs a lazy transformation or filter, and returns a new iterator. The data is pulled through the pipeline on demand, keeping memory usage constant regardless of the stream's size.

**The Pitfall:** Reusing generators. A generator is an iterator, which means it can only be traversed once. Once it is exhausted, attempting to read from it again yields nothing. If you need to perform multiple passes over the data, you must recreate the generator chain or design your pipeline to process data in a single pass.

**In `exercises.py`:** Implement the `filter_level` generator. It should take an iterable of `LogRecord`s and a severity level string (e.g., `"ERROR"`), and yield only the records that match that level. The match should be case-insensitive.

**Test it:** `python3 -m unittest test_exercises.TestFilterLevel -v`

---

## 4. Time-Bucketed Aggregation

```python
>>> from datetime import datetime
>>> dt = datetime(2023, 10, 27, 12, 34, 56)
>>> dt.replace(second=0, microsecond=0)
datetime.datetime(2023, 10, 27, 12, 34)
```

To detect spikes in errors, we often group logs into fixed time buckets (e.g., 1-minute intervals). If we assume the input stream is sorted chronologically, we can perform this aggregation on-the-fly with $O(1)$ auxiliary space. We keep track of the current active bucket. As soon as we see a record belonging to a later bucket, we yield the aggregated count for the completed bucket and start a new one.

An "error" in our pipeline is defined as any record where the level is `"ERROR"` or the HTTP status code is $\ge 500$.

**The Pitfall:** Forgetting to yield the final bucket. When the input stream ends, the loop terminates, but the last active bucket has not been yielded yet. This is a classic off-by-one bug in streaming aggregation. Always check if there is an active bucket remaining after the loop finishes and yield it.

**In `exercises.py`:** Implement the `errors_per_minute` generator. It takes an iterable of `LogRecord`s (guaranteed to be sorted by timestamp) and yields tuples of `(minute_datetime, error_count)` for any minute bucket that contains at least one error.

**Test it:** `python3 -m unittest test_exercises.TestErrorsPerMinute -v`

---

## 5. Finding Top-K Elements

```python
>>> from collections import Counter
>>> counts = Counter(["/index.html", "/api", "/index.html"])
>>> counts.most_common(1)
[('/index.html', 2)]
```

To find the most popular endpoints, we must count occurrences across the entire stream. While we cannot do this with $O(1)$ memory because we must remember the count of every unique endpoint, we can still process the stream lazily to build our frequency map.

**The Pitfall:** Incorrect tie-breaking. When sorting the endpoints by frequency, we want the most frequent endpoints first. But if two endpoints have the exact same count, we want to sort them alphabetically. A naive sort like `sorted(counts.items(), key=lambda x: x[1], reverse=True)` will not guarantee alphabetical order for ties. We must construct a sorting key that handles both directions: descending for counts, and ascending for names.

**In `exercises.py`:** Implement `top_endpoints`. It consumes the entire stream of `LogRecord`s and returns a list of the top `k` endpoints as `(endpoint, count)` tuples, sorted descending by count, with ties broken alphabetically.

**Test it:** `python3 -m unittest test_exercises.TestTopEndpoints -v`

---

## 6. Sliding-Window Alerting

```python
>>> from collections import deque
>>> window = deque()
>>> window.append(1)
>>> window.append(2)
>>> window.popleft()
1
```

Real-time alerting systems monitor error rates over a sliding window (e.g., the last 5 minutes). For every incoming record, we add it to our window and evict any records that have fallen outside the window's time range. We then calculate the error rate (errors divided by total records in the window). If the rate exceeds a threshold, we raise an alert.

Using a `collections.deque` allows us to append and pop from both ends in $O(1)$ time, making it the perfect data structure for a sliding window.

**The Pitfall:** Inefficient eviction. If you only evict one expired record per incoming record, your window will grow indefinitely if there are gaps in the log stream. You must use a `while` loop to evict *all* expired records from the front of the queue before calculating the error rate.

**In `exercises.py`:** Implement the `error_rate_alerts` generator. It takes an iterable of `LogRecord`s (sorted by timestamp), a `window_seconds` integer, and a `threshold` float (e.g., `0.1` for 10%). For each record processed, it updates the sliding window, evicts expired records, and yields `(current_timestamp, error_rate)` if the error rate in the window is strictly greater than the threshold.

**Test it:** `python3 -m unittest test_exercises.TestErrorRateAlerts -v`

---

## The exercises

| Function / class | What it teaches |
|---|---|
| `LogRecord` | Structured data representation using immutable dataclasses |
| `parse_line` | Robust string parsing and error handling with regular expressions |
| `records` | Building a lazy generator to stream structured data with $O(1)$ memory |
| `filter_level` | Chaining generators to build modular, lazy processing pipelines |
| `errors_per_minute` | State-machine based streaming aggregation over sorted time-series data |
| `top_endpoints` | Frequency tracking and custom multi-key sorting with tie-breakers |
| `error_rate_alerts` | Implementing a sliding-window algorithm using `collections.deque` |

## Check your understanding

1. **Why does `parse_line` return `None` instead of raising an exception when it encounters a malformed line?**
   <details><summary>Answer</summary>In a production streaming pipeline, a single malformed line (e.g., a corrupted log write) should not crash the entire application. Returning <code>None</code> allows the pipeline to gracefully skip invalid data and continue processing subsequent logs.</details>

2. **What is the memory complexity of the `records` generator?**
   <details><summary>Answer</summary>The memory complexity is $O(1)$ auxiliary space. It only holds a single line and its parsed <code>LogRecord</code> in memory at any given time, regardless of whether the input stream contains ten lines or ten billion lines.</details>

3. **What happens if you attempt to iterate over the generator returned by `filter_level` twice?**
   <details><summary>Answer</summary>The second iteration will yield nothing. Generators are iterators; they are stateful and can only be consumed once. To iterate again, you must recreate the generator chain from the source.</details>

4. **In `errors_per_minute`, why is it safe to yield a bucket as soon as we see a record with a timestamp in a later minute?**
   <details><summary>Answer</summary>Because the input stream is guaranteed to be sorted chronologically. Once we see a record from a later minute, we are guaranteed that no more records from the previous minute will arrive, making it safe to finalize and yield that bucket's count.</details>

5. **Why is `collections.deque` preferred over a standard Python `list` for implementing a sliding window?**
   <details><summary>Answer</summary>A <code>deque</code> (double-ended queue) supports $O(1)$ appends and pops from both ends. A standard Python <code>list</code> has $O(N)$ complexity when removing elements from the front (index 0) because all remaining elements must be shifted in memory.</details>

## Interview questions this module answers

- *How do you process a 100GB log file in Python without running out of memory?*
- *What are generators, and how do they differ from standard functions returning lists?*
- *How do you implement a streaming sliding-window algorithm with $O(1)$ appends and evictions?*
- *How do you sort a collection by multiple keys with different sorting directions (e.g., descending count, ascending name)?*
- *How do you build a modular, testable data pipeline using generator delegation and chaining?*
