"""63 — A streaming log pipeline

Implement each function and class so the tests in test_exercises.py pass.
Read README.md first; every exercise below is covered there.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Iterator, Optional
import re
from collections import Counter, deque
LOG_PATTERN = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] (\w+) (\S+) (\d+)$")


@dataclass(frozen=True)
class LogRecord:
    """An immutable representation of a single parsed log record."""
    timestamp: datetime
    level: str
    endpoint: str
    status: int


def parse_line(line: str) -> Optional[LogRecord]:
    """Parse a raw log line into a LogRecord.

    Expected format:
    [YYYY-MM-DD HH:MM:SS] LEVEL ENDPOINT STATUS

    Example:
    [2023-10-27 12:34:56] INFO /index.html 200

    If the line is malformed, has an invalid date, or does not match the
    format, return None instead of raising an exception.
    """
    raise NotImplementedError("your code here")


def records(lines: Iterable[str]) -> Iterator[LogRecord]:
    """A generator that yields valid LogRecord objects from an iterable of raw lines.

    Malformed lines should be silently skipped.
    """
    raise NotImplementedError("your code here")


def filter_level(records: Iterable[LogRecord], level: str) -> Iterator[LogRecord]:
    """Yield only records matching the specified severity level (case-insensitive)."""
    raise NotImplementedError("your code here")


def errors_per_minute(records: Iterable[LogRecord]) -> Iterator[tuple[datetime, int]]:
    """Aggregate error counts bucketed by 1-minute intervals.

    An error is defined as a record where level == "ERROR" or status >= 500.
    The input records are guaranteed to be sorted chronologically.

    Yields tuples of (minute_datetime, error_count) for any minute bucket
    that contains at least one error, as soon as the stream moves past that minute
    or ends.
    """
    raise NotImplementedError("your code here")


def top_endpoints(records: Iterable[LogRecord], k: int) -> list[tuple[str, int]]:
    """Consume the stream and return the top k endpoints by request count.

    Returns a list of (endpoint, count) tuples, sorted descending by count.
    Ties must be broken alphabetically by endpoint name.
    """
    raise NotImplementedError("your code here")


def error_rate_alerts(
    records: Iterable[LogRecord],
    window_seconds: int,
    threshold: float
) -> Iterator[tuple[datetime, float]]:
    """Monitor the error rate over a sliding window of window_seconds.

    An error is defined as a record where level == "ERROR" or status >= 500.
    The input records are guaranteed to be sorted chronologically.

    For each record processed:
      1. Add it to the sliding window.
      2. Evict any records from the window that are older than (current_time - window_seconds).
         A record is expired if record.timestamp <= current_time - window_seconds.
      3. Calculate the error rate: (errors in window) / (total records in window).
      4. If the error rate is strictly greater than the threshold, yield (current_time, error_rate).
    """
    raise NotImplementedError("your code here")
