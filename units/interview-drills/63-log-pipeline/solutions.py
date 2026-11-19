"""63 — A streaming log pipeline

Reference solutions for the streaming log pipeline exercises.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Iterator, Optional
import re
from collections import Counter, deque


@dataclass(frozen=True)
class LogRecord:
    timestamp: datetime
    level: str
    endpoint: str
    status: int


# Regex pattern to match: [YYYY-MM-DD HH:MM:SS] LEVEL ENDPOINT STATUS
LOG_PATTERN = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] (\w+) (\S+) (\d+)$")


def parse_line(line: str) -> Optional[LogRecord]:
    match = LOG_PATTERN.match(line.strip())
    if not match:
        return None
    
    ts_str, level, endpoint, status_str = match.groups()
    try:
        timestamp = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        status = int(status_str)
        return LogRecord(timestamp=timestamp, level=level, endpoint=endpoint, status=status)
    except ValueError:
        return None


def records(lines: Iterable[str]) -> Iterator[LogRecord]:
    for line in lines:
        record = parse_line(line)
        if record is not None:
            yield record


def filter_level(records: Iterable[LogRecord], level: str) -> Iterator[LogRecord]:
    target_level = level.upper()
    for record in records:
        if record.level.upper() == target_level:
            yield record


def errors_per_minute(records: Iterable[LogRecord]) -> Iterator[tuple[datetime, int]]:
    current_minute = None
    error_count = 0

    for record in records:
        rec_minute = record.timestamp.replace(second=0, microsecond=0)
        is_error = (record.level == "ERROR" or record.status >= 500)

        if current_minute is None:
            current_minute = rec_minute
            error_count = 1 if is_error else 0
        elif rec_minute == current_minute:
            if is_error:
                error_count += 1
        else:
            if error_count > 0:
                yield (current_minute, error_count)
            current_minute = rec_minute
            error_count = 1 if is_error else 0

    if current_minute is not None and error_count > 0:
        yield (current_minute, error_count)


def top_endpoints(records: Iterable[LogRecord], k: int) -> list[tuple[str, int]]:
    counts = Counter(record.endpoint for record in records)
    # Sort by count descending, then alphabetically by endpoint name ascending
    sorted_endpoints = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return sorted_endpoints[:k]


def error_rate_alerts(
    records: Iterable[LogRecord],
    window_seconds: int,
    threshold: float
) -> Iterator[tuple[datetime, float]]:
    window = deque()
    error_count = 0

    for record in records:
        current_time = record.timestamp
        is_error = (record.level == "ERROR" or record.status >= 500)

        window.append(record)
        if is_error:
            error_count += 1

        cutoff = current_time - timedelta(seconds=window_seconds)
        while window and window[0].timestamp <= cutoff:
            expired = window.popleft()
            expired_is_error = (expired.level == "ERROR" or expired.status >= 500)
            if expired_is_error:
                error_count -= 1

        total = len(window)
        if total > 0:
            rate = error_count / total
            if rate > threshold:
                yield (current_time, rate)
