"""A local stand-in for the `timeout_decorator` package.

The CodeSignal environment provides the real package (`pip install
timeout-decorator`). This shim exists only so the platform's test files run
unmodified on your machine, with no network and no install step.

It uses a real interval timer, so an infinite loop is interrupted rather than
hanging your test run. On platforms without SIGALRM (Windows) it degrades to
measuring elapsed time after the fact.
"""

import functools
import signal
import time


class TimeoutError(Exception):
    """Raised when a decorated call exceeds its budget."""


def timeout(seconds):
    """Fail a test that takes longer than `seconds`."""

    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def on_alarm(signum, frame):
                raise TimeoutError(f"timed out after {seconds}s")

            try:
                previous = signal.signal(signal.SIGALRM, on_alarm)
                signal.setitimer(signal.ITIMER_REAL, seconds)
            except (AttributeError, ValueError):
                # No SIGALRM here (Windows, or not the main thread).
                started = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - started
                if elapsed > seconds:
                    raise TimeoutError(f"took {elapsed:.3f}s, budget {seconds}s")
                return result

            try:
                return func(*args, **kwargs)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous)

        return wrapper

    return decorate
