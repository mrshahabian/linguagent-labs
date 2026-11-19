"""Tests for 63 — A streaming log pipeline."""

import importlib
import os
import unittest
from datetime import datetime

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestParseLine(unittest.TestCase):
    def test_valid_info_log(self):
        line = "[2023-10-27 12:34:56] INFO /index.html 200"
        record = m.parse_line(line)
        self.assertIsNotNone(record)
        self.assertEqual(record.timestamp, datetime(2023, 10, 27, 12, 34, 56))
        self.assertEqual(record.level, "INFO")
        self.assertEqual(record.endpoint, "/index.html")
        self.assertEqual(record.status, 200)

    def test_valid_error_log(self):
        line = "[2023-10-27 12:35:00] ERROR /api/login 500"
        record = m.parse_line(line)
        self.assertIsNotNone(record)
        self.assertEqual(record.level, "ERROR")
        self.assertEqual(record.status, 500)

    def test_malformed_line_returns_none(self):
        bad_lines = [
            "invalid log format",
            "[2023-10-27 12:34:56] INFO /index.html",  # Missing status
            "2023-10-27 12:34:56 INFO /index.html 200",  # Missing brackets
            "[2023-10-27 12:34:56] INFO 200",  # Missing endpoint
        ]
        for line in bad_lines:
            self.assertIsNone(m.parse_line(line), f"Failed to reject: {line}")

    def test_invalid_date_returns_none(self):
        line = "[2023-02-30 12:00:00] INFO /index.html 200"  # Feb 30th is invalid
        self.assertIsNone(m.parse_line(line))


class TestRecords(unittest.TestCase):
    def test_empty_stream(self):
        self.assertEqual(list(m.records([])), [])

    def test_mix_of_valid_and_invalid_lines(self):
        lines = [
            "[2023-10-27 12:34:56] INFO /index.html 200",
            "corrupted line here",
            "[2023-10-27 12:34:57] ERROR /api/v1 500",
        ]
        results = list(m.records(lines))
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].endpoint, "/index.html")
        self.assertEqual(results[1].endpoint, "/api/v1")

    def test_is_generator_lazy(self):
        # Verify that it yields items lazily and does not consume the whole iterator
        def infinite_lines():
            while True:
                yield "[2023-10-27 12:34:56] INFO /index.html 200"

        gen = m.records(infinite_lines())
        first = next(gen)
        self.assertEqual(first.endpoint, "/index.html")
        second = next(gen)
        self.assertEqual(second.endpoint, "/index.html")


class TestFilterLevel(unittest.TestCase):
    def setUp(self):
        self.records = [
            m.LogRecord(datetime(2023, 10, 27, 12, 0), "INFO", "/", 200),
            m.LogRecord(datetime(2023, 10, 27, 12, 1), "ERROR", "/", 500),
            m.LogRecord(datetime(2023, 10, 27, 12, 2), "info", "/", 200),
        ]

    def test_filter_exact_match(self):
        filtered = list(m.filter_level(self.records, "ERROR"))
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].level, "ERROR")

    def test_filter_case_insensitive(self):
        filtered = list(m.filter_level(self.records, "info"))
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0].level, "INFO")
        self.assertEqual(filtered[1].level, "info")

    def test_filter_no_matches(self):
        filtered = list(m.filter_level(self.records, "WARNING"))
        self.assertEqual(filtered, [])


class TestErrorsPerMinute(unittest.TestCase):
    def test_empty_stream(self):
        self.assertEqual(list(m.errors_per_minute([])), [])

    def test_single_minute_with_errors(self):
        records = [
            m.LogRecord(datetime(2023, 10, 27, 12, 34, 10), "INFO", "/", 200),
            m.LogRecord(datetime(2023, 10, 27, 12, 34, 15), "ERROR", "/", 500),
            m.LogRecord(datetime(2023, 10, 27, 12, 34, 45), "INFO", "/", 500),  # status >= 500 is error
        ]
        results = list(m.errors_per_minute(records))
        self.assertEqual(results, [(datetime(2023, 10, 27, 12, 34), 2)])

    def test_multiple_minutes_some_without_errors(self):
        records = [
            m.LogRecord(datetime(2023, 10, 27, 12, 34, 10), "ERROR", "/", 500),  # Min 34: 1 error
            m.LogRecord(datetime(2023, 10, 27, 12, 35, 15), "INFO", "/", 200),   # Min 35: 0 errors
            m.LogRecord(datetime(2023, 10, 27, 12, 36, 0), "ERROR", "/", 500),  # Min 36: 1 error
            m.LogRecord(datetime(2023, 10, 27, 12, 36, 30), "ERROR", "/", 500),  # Min 36: 2 errors total
        ]
        results = list(m.errors_per_minute(records))
        expected = [
            (datetime(2023, 10, 27, 12, 34), 1),
            (datetime(2023, 10, 27, 12, 36), 2),
        ]
        self.assertEqual(results, expected)


class TestTopEndpoints(unittest.TestCase):
    def test_empty_stream(self):
        self.assertEqual(m.top_endpoints([], 5), [])

    def test_typical_counts(self):
        records = [
            m.LogRecord(datetime(2023, 10, 27, 12, 0), "INFO", "/index.html", 200),
            m.LogRecord(datetime(2023, 10, 27, 12, 0), "INFO", "/api", 200),
            m.LogRecord(datetime(2023, 10, 27, 12, 0), "INFO", "/index.html", 200),
        ]
        self.assertEqual(m.top_endpoints(records, 1), [("/index.html", 2)])

    def test_tie_breaker_sorting(self):
        # Both have count of 2. Alphabetical order should put /api before /index.html
        records = [
            m.LogRecord(datetime(2023, 10, 27, 12, 0), "INFO", "/index.html", 200),
            m.LogRecord(datetime(2023, 10, 27, 12, 0), "INFO", "/api", 200),
            m.LogRecord(datetime(2023, 10, 27, 12, 0), "INFO", "/index.html", 200),
            m.LogRecord(datetime(2023, 10, 27, 12, 0), "INFO", "/api", 200),
        ]
        self.assertEqual(m.top_endpoints(records, 2), [("/api", 2), ("/index.html", 2)])

    def test_k_larger_than_unique_endpoints(self):
        records = [
            m.LogRecord(datetime(2023, 10, 27, 12, 0), "INFO", "/index.html", 200),
        ]
        self.assertEqual(m.top_endpoints(records, 10), [("/index.html", 1)])


class TestErrorRateAlerts(unittest.TestCase):
    def test_no_errors_no_alerts(self):
        records = [
            m.LogRecord(datetime(2023, 10, 27, 12, 0, 0), "INFO", "/", 200),
            m.LogRecord(datetime(2023, 10, 27, 12, 0, 30), "INFO", "/", 200),
        ]
        alerts = list(m.error_rate_alerts(records, window_seconds=60, threshold=0.5))
        self.assertEqual(alerts, [])

    def test_all_errors_triggers_alerts(self):
        records = [
            m.LogRecord(datetime(2023, 10, 27, 12, 0, 0), "ERROR", "/", 500),
        ]
        alerts = list(m.error_rate_alerts(records, window_seconds=60, threshold=0.5))
        # 1 error out of 1 record = 100% error rate > 50%
        self.assertEqual(alerts, [(datetime(2023, 10, 27, 12, 0, 0), 1.0)])

    def test_sliding_window_eviction(self):
        records = [
            m.LogRecord(datetime(2023, 10, 27, 12, 0, 0), "ERROR", "/", 500),  # Window: [t=0 (Err)]. Rate = 1.0. Alert!
            m.LogRecord(datetime(2023, 10, 27, 12, 0, 30), "INFO", "/", 200),   # Window: [t=0 (Err), t=30 (OK)]. Rate = 0.5. No alert.
            m.LogRecord(datetime(2023, 10, 27, 12, 1, 1), "INFO", "/", 200),   # t=0 expired. Window: [t=30 (OK), t=61 (OK)]. Rate = 0.0. No alert.
        ]
        alerts = list(m.error_rate_alerts(records, window_seconds=60, threshold=0.5))
        self.assertEqual(alerts, [(datetime(2023, 10, 27, 12, 0, 0), 1.0)])

    def test_threshold_boundary(self):
        # Rate must be strictly greater than threshold
        records = [
            m.LogRecord(datetime(2023, 10, 27, 12, 0, 0), "ERROR", "/", 500),
            m.LogRecord(datetime(2023, 10, 27, 12, 0, 10), "INFO", "/", 200),
        ]
        alerts = list(m.error_rate_alerts(records, window_seconds=60, threshold=0.5))
        # Rate is exactly 0.5, threshold is 0.5 -> no alert
        self.assertEqual(alerts, [(datetime(2023, 10, 27, 12, 0, 0), 1.0)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
