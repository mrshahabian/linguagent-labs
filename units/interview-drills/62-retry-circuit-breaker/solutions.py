"""62 — Retries and a circuit breaker

Reference implementation of the exception filtering, exponential backoff,
retry loop, and circuit breaker state machine.
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
    if never_retry is not None and isinstance(exc, tuple(never_retry)):
        return False
    return isinstance(exc, tuple(retry_on))


def exponential_delays(
    base: float,
    factor: float,
    cap: float,
    attempts: int,
    rng: Optional[random.Random] = None,
) -> List[float]:
    if attempts <= 1:
        return []
    
    delays = []
    for i in range(attempts - 1):
        un_jittered = min(base * (factor ** i), cap)
        if rng is not None:
            delay = rng.uniform(0, un_jittered)
        else:
            delay = un_jittered
        delays.append(delay)
    return delays


def retry(
    fn: Callable[[], any],
    attempts: int,
    retry_on: Iterable[Type[Exception]],
    sleep: Callable[[float], None],
    delays: List[float],
    never_retry: Optional[Iterable[Type[Exception]]] = None,
) -> any:
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
        
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if not is_retryable(exc, retry_on, never_retry) or attempt == attempts:
                raise exc
            
            delay = delays[attempt - 1]
            sleep(delay)


class CircuitBreaker:
    def __init__(
        self, 
        failure_threshold: int, 
        recovery_timeout: float, 
        clock: Callable[[], float]
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.clock = clock
        
        self._state = "CLOSED"
        self._failures = 0
        self._last_state_change = 0.0

    def _update_state(self) -> None:
        if self._state == "OPEN":
            if self.clock() - self._last_state_change >= self.recovery_timeout:
                self._state = "HALF-OPEN"

    @property
    def state(self) -> str:
        self._update_state()
        return self._state

    def allow_request(self) -> bool:
        if self.state == "OPEN":
            raise CircuitOpenError("Circuit is open")
        return True

    def record_success(self) -> None:
        self._update_state()
        if self._state in ("HALF-OPEN", "CLOSED"):
            self._state = "CLOSED"
            self._failures = 0

    def record_failure(self) -> None:
        self._update_state()
        if self._state == "CLOSED":
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._state = "OPEN"
                self._last_state_change = self.clock()
        elif self._state == "HALF-OPEN":
            self._state = "OPEN"
            self._last_state_change = self.clock()


def guarded_call(
    fn: Callable[[], any],
    breaker: CircuitBreaker,
    attempts: int,
    retry_on: Iterable[Type[Exception]],
    sleep: Callable[[float], None],
    delays: List[float],
    never_retry: Optional[Iterable[Type[Exception]]] = None,
) -> any:
    breaker.allow_request()
    try:
        result = retry(fn, attempts, retry_on, sleep, delays, never_retry)
        breaker.record_success()
        return result
    except Exception as exc:
        breaker.record_failure()
        raise exc
