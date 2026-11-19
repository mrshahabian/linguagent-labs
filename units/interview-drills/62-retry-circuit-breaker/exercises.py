"""62 — Retries and a circuit breaker

Implement the exception filtering, exponential backoff, retry loop, and
circuit breaker state machine to build a resilient execution client.
"""

import random
from typing import Callable, Iterable, List, Optional, Type


class CircuitOpenError(Exception):
    """Raised when the circuit breaker is OPEN and blocks a request."""
    pass


def is_retryable(
    exc: Exception,
    retry_on: Iterable[Type[Exception]],
    never_retry: Optional[Iterable[Type[Exception]]] = None,
) -> bool:
    """Return True if the exception instance is an instance of any class in
    retry_on, unless it is an instance of any class in never_retry.

    The never_retry exclusion takes absolute precedence.
    """
    raise NotImplementedError("your code here")


def exponential_delays(
    base: float,
    factor: float,
    cap: float,
    attempts: int,
    rng: Optional[random.Random] = None,
) -> List[float]:
    """Generate a list of delay durations for (attempts - 1) retries.

    For attempt i (from 0 to attempts - 2):
        un_jittered_delay = min(base * (factor ** i), cap)
        if rng is provided:
            delay = uniform random value between 0 and un_jittered_delay
        else:
            delay = un_jittered_delay

    If attempts <= 1, return an empty list.
    """
    raise NotImplementedError("your code here")


def retry(
    fn: Callable[[], any],
    attempts: int,
    retry_on: Iterable[Type[Exception]],
    sleep: Callable[[float], None],
    delays: List[float],
    never_retry: Optional[Iterable[Type[Exception]]] = None,
) -> any:
    """Execute fn up to `attempts` times.

    If fn raises an exception:
      - If it is not retryable (using is_retryable), raise it immediately.
      - If it is retryable and attempts remain, call sleep(delays[attempt_index])
        and try again.
      - If attempts are exhausted, re-raise the last exception.
    """
    raise NotImplementedError("your code here")


class CircuitBreaker:
    def __init__(
        self, 
        failure_threshold: int, 
        recovery_timeout: float, 
        clock: Callable[[], float]
    ):
        """Initialize the circuit breaker.

        States are: "CLOSED", "OPEN", "HALF-OPEN".
        Starts in "CLOSED" state with 0 consecutive failures.
        """
        raise NotImplementedError("your code here")

    @property
    def state(self) -> str:
        """Return the current state: "CLOSED", "OPEN", or "HALF-OPEN".

        This state must be evaluated dynamically based on the current clock()
        and the recovery_timeout if the state is currently "OPEN".
        """
        raise NotImplementedError("your code here")

    def allow_request(self) -> bool:
        """Check if a request is allowed.

        If the state is "OPEN", raise CircuitOpenError.
        If the state is "HALF-OPEN" or "CLOSED", return True.
        """
        raise NotImplementedError("your code here")

    def record_success(self) -> None:
        """Record a successful operation.

        If the state was "HALF-OPEN", transition to "CLOSED" and reset failures.
        If the state was "CLOSED", reset consecutive failures to 0.
        """
        raise NotImplementedError("your code here")

    def record_failure(self) -> None:
        """Record a failed operation.

        If the state is "CLOSED":
            Increment consecutive failures.
            If consecutive failures >= failure_threshold, transition to "OPEN"
            and record the time of opening.
        If the state is "HALF-OPEN":
            Transition back to "OPEN" immediately and record the time of opening.
        """
        raise NotImplementedError("your code here")


def guarded_call(
    fn: Callable[[], any],
    breaker: CircuitBreaker,
    attempts: int,
    retry_on: Iterable[Type[Exception]],
    sleep: Callable[[float], None],
    delays: List[float],
    never_retry: Optional[Iterable[Type[Exception]]] = None,
) -> any:
    """Execute fn guarded by both the circuit breaker and the retry loop.

    1. Check if the breaker allows the request. If not, let CircuitOpenError propagate.
    2. Execute fn inside the retry loop.
    3. If the retry loop succeeds, record success on the breaker and return the result.
    4. If the retry loop fails (exhausts retries or hits a non-retryable error),
       record failure on the breaker and re-raise the exception.
    """
    raise NotImplementedError("your code here")
