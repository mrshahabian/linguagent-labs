"""Tests for 62 — Retries and a circuit breaker."""

import importlib
import os
import random
import unittest

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class FakeClock:
    def __init__(self, start=0.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, amount: float):
        self.now += amount


class TestIsRetryable(unittest.TestCase):
    def test_basic_matching(self):
        exc = ValueError("invalid value")
        self.assertTrue(m.is_retryable(exc, (ValueError, TypeError)))
        self.assertFalse(m.is_retryable(exc, (TypeError, KeyError)))

    def test_subclass_matching(self):
        exc = KeyError("missing key")
        # KeyError is a subclass of LookupError
        self.assertTrue(m.is_retryable(exc, (LookupError,)))

    def test_never_retry_takes_precedence(self):
        exc = KeyError("missing key")
        # Even though it matches LookupError, KeyError is in never_retry
        self.assertFalse(
            m.is_retryable(exc, retry_on=(LookupError,), never_retry=(KeyError,))
        )

    def test_never_retry_subclass_exclusion(self):
        exc = KeyError("missing key")
        # Excluding LookupError should exclude its subclass KeyError
        self.assertFalse(
            m.is_retryable(exc, retry_on=(Exception,), never_retry=(LookupError,))
        )


class TestExponentialDelays(unittest.TestCase):
    def test_no_jitter_delays(self):
        delays = m.exponential_delays(base=1.0, factor=2.0, cap=10.0, attempts=4)
        # 3 delays expected for 4 attempts
        # 1.0 * 2^0 = 1.0
        # 1.0 * 2^1 = 2.0
        # 1.0 * 2^2 = 4.0
        self.assertEqual(delays, [1.0, 2.0, 4.0])

    def test_cap_applied(self):
        delays = m.exponential_delays(base=2.0, factor=3.0, cap=10.0, attempts=4)
        # 2.0 * 3^0 = 2.0
        # 2.0 * 3^1 = 6.0
        # 2.0 * 3^2 = 18.0 -> capped at 10.0
        self.assertEqual(delays, [2.0, 6.0, 10.0])

    def test_attempts_edge_cases(self):
        self.assertEqual(m.exponential_delays(1.0, 2.0, 10.0, attempts=1), [])
        self.assertEqual(m.exponential_delays(1.0, 2.0, 10.0, attempts=0), [])

    def test_with_jitter(self):
        rng = random.Random(42)
        delays = m.exponential_delays(base=10.0, factor=2.0, cap=100.0, attempts=4, rng=rng)
        self.assertEqual(len(delays), 3)
        # Each delay must be strictly between 0 and the un-jittered limit
        self.assertTrue(0 <= delays[0] <= 10.0)
        self.assertTrue(0 <= delays[1] <= 20.0)
        self.assertTrue(0 <= delays[2] <= 40.0)


class TestRetry(unittest.TestCase):
    def test_immediate_success(self):
        calls = 0
        def fn():
            nonlocal calls
            calls += 1
            return "success"

        sleeps = []
        result = m.retry(fn, attempts=3, retry_on=(ValueError,), sleep=sleeps.append, delays=[1, 2])
        self.assertEqual(result, "success")
        self.assertEqual(calls, 1)
        self.assertEqual(sleeps, [])

    def test_success_after_retries(self):
        calls = 0
        def fn():
            nonlocal calls
            calls += 1
            if calls < 3:
                raise ValueError("temporary error")
            return "success"

        sleeps = []
        result = m.retry(fn, attempts=3, retry_on=(ValueError,), sleep=sleeps.append, delays=[1.0, 2.0])
        self.assertEqual(result, "success")
        self.assertEqual(calls, 3)
        self.assertEqual(sleeps, [1.0, 2.0])

    def test_exhaustion_raises_last_error(self):
        calls = 0
        def fn():
            nonlocal calls
            calls += 1
            raise ValueError(f"error {calls}")

        sleeps = []
        with self.assertRaises(ValueError) as ctx:
            m.retry(fn, attempts=3, retry_on=(ValueError,), sleep=sleeps.append, delays=[1.0, 2.0])
        
        self.assertEqual(str(ctx.exception), "error 3")
        self.assertEqual(calls, 3)
        self.assertEqual(sleeps, [1.0, 2.0])

    def test_non_retryable_error_propagates_immediately(self):
        calls = 0
        def fn():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise TypeError("fatal error")
            return "success"

        sleeps = []
        with self.assertRaises(TypeError):
            m.retry(fn, attempts=3, retry_on=(ValueError,), sleep=sleeps.append, delays=[1.0, 2.0])
        
        self.assertEqual(calls, 1)
        self.assertEqual(sleeps, [])


class TestCircuitBreaker(unittest.TestCase):
    def test_initial_state_closed(self):
        clock = FakeClock()
        breaker = m.CircuitBreaker(failure_threshold=3, recovery_timeout=5.0, clock=clock)
        self.assertEqual(breaker.state, "CLOSED")
        self.assertTrue(breaker.allow_request())

    def test_trips_to_open_on_failures(self):
        clock = FakeClock()
        breaker = m.CircuitBreaker(failure_threshold=3, recovery_timeout=5.0, clock=clock)
        
        breaker.record_failure()
        self.assertEqual(breaker.state, "CLOSED")
        breaker.record_failure()
        self.assertEqual(breaker.state, "CLOSED")
        
        # Third failure trips the circuit
        breaker.record_failure()
        self.assertEqual(breaker.state, "OPEN")
        
        with self.assertRaises(m.CircuitOpenError):
            breaker.allow_request()

    def test_success_resets_failure_count(self):
        clock = FakeClock()
        breaker = m.CircuitBreaker(failure_threshold=2, recovery_timeout=5.0, clock=clock)
        
        breaker.record_failure()
        breaker.record_success()
        breaker.record_failure()
        
        # Failure count was reset, so 1 failure does not trip it
        self.assertEqual(breaker.state, "CLOSED")

    def test_recovery_timeout_transitions_to_half_open(self):
        clock = FakeClock(start=100.0)
        breaker = m.CircuitBreaker(failure_threshold=2, recovery_timeout=10.0, clock=clock)
        
        breaker.record_failure()
        breaker.record_failure()
        self.assertEqual(breaker.state, "OPEN")
        
        # Advance clock, but not enough
        clock.advance(9.9)
        self.assertEqual(breaker.state, "OPEN")
        
        # Advance past recovery timeout
        clock.advance(0.2)
        self.assertEqual(breaker.state, "HALF-OPEN")
        self.assertTrue(breaker.allow_request())

    def test_half_open_success_closes_circuit(self):
        clock = FakeClock(start=100.0)
        breaker = m.CircuitBreaker(failure_threshold=2, recovery_timeout=10.0, clock=clock)
        
        breaker.record_failure()
        breaker.record_failure()
        clock.advance(15.0)
        
        self.assertEqual(breaker.state, "HALF-OPEN")
        breaker.record_success()
        self.assertEqual(breaker.state, "CLOSED")

    def test_half_open_failure_trips_immediately(self):
        clock = FakeClock(start=100.0)
        breaker = m.CircuitBreaker(failure_threshold=5, recovery_timeout=10.0, clock=clock)
        
        breaker.record_failure() # trips because threshold is 1? No, let's trip it.
        # Let's force trip by hitting threshold
        for _ in range(5): 
            breaker.record_failure()
        self.assertEqual(breaker.state, "OPEN")
        
        clock.advance(15.0)
        self.assertEqual(breaker.state, "HALF-OPEN")
        
        # A single failure in HALF-OPEN trips it immediately without needing 5 failures
        breaker.record_failure()
        self.assertEqual(breaker.state, "OPEN")


class TestGuardedCall(unittest.TestCase):
    def test_guarded_call_success(self):
        clock = FakeClock()
        breaker = m.CircuitBreaker(failure_threshold=2, recovery_timeout=5.0, clock=clock)
        sleeps = []
        
        result = m.guarded_call(
            lambda: "ok",
            breaker,
            attempts=3,
            retry_on=(ValueError,),
            sleep=sleeps.append,
            delays=[1.0, 2.0]
        )
        self.assertEqual(result, "ok")
        self.assertEqual(breaker.state, "CLOSED")

    def test_guarded_call_transient_failure_succeeds(self):
        clock = FakeClock()
        breaker = m.CircuitBreaker(failure_threshold=2, recovery_timeout=5.0, clock=clock)
        sleeps = []
        
        calls = 0
        def fn():
            nonlocal calls
            calls += 1
            if calls < 3:
                raise ValueError("transient")
            return "ok"
            
        result = m.guarded_call(
            fn,
            breaker,
            attempts=3,
            retry_on=(ValueError,),
            sleep=sleeps.append,
            delays=[1.0, 2.0]
        )
        self.assertEqual(result, "ok")
        # Breaker should still be CLOSED because the high-level call succeeded
        self.assertEqual(breaker.state, "CLOSED")
        self.assertEqual(sleeps, [1.0, 2.0])

    def test_guarded_call_exhaustion_trips_breaker(self):
        clock = FakeClock()
        breaker = m.CircuitBreaker(failure_threshold=2, recovery_timeout=5.0, clock=clock)
        sleeps = []
        
        def fn():
            raise ValueError("persistent")
            
        # First high-level failure
        with self.assertRaises(ValueError):
            m.guarded_call(fn, breaker, attempts=2, retry_on=(ValueError,), sleep=sleeps.append, delays=[1.0])
        self.assertEqual(breaker.state, "CLOSED") # threshold is 2
        
        # Second high-level failure
        with self.assertRaises(ValueError):
            m.guarded_call(fn, breaker, attempts=2, retry_on=(ValueError,), sleep=sleeps.append, delays=[1.0])
        self.assertEqual(breaker.state, "OPEN")

    def test_guarded_call_blocked_when_open(self):
        clock = FakeClock()
        breaker = m.CircuitBreaker(failure_threshold=1, recovery_timeout=5.0, clock=clock)
        breaker.record_failure() # trip it
        
        calls = 0
        def fn():
            nonlocal calls
            calls += 1
            return "ok"
            
        with self.assertRaises(m.CircuitOpenError):
            m.guarded_call(fn, breaker, attempts=3, retry_on=(ValueError,), sleep=print, delays=[1.0, 2.0])
            
        self.assertEqual(calls, 0) # fn was never called


if __name__ == "__main__":
    unittest.main(verbosity=2)
