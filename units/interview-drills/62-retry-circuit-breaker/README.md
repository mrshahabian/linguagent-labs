# 62 — Retries and a circuit breaker

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/interview-drills/62-retry-circuit-breaker/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/interview-drills/retry-circuit-breaker?utm_source=github&utm_medium=labs)

*~45 minutes · 22 tests · Interview*

In a distributed system, failure is a constant. Network hiccups, temporary database overloads, and rolling restarts of downstream microservices will cause occasional, transient errors. If your application simply propagates these errors to the user, you build a fragile, frustrating product. 

To make systems resilient, we use two primary patterns: **Retries with Exponential Backoff** and **Circuit Breakers**. However, implementing these incorrectly can make outages *worse*. Retrying too quickly or without randomness creates a "retry storm" that acts as a self-inflicted Distributed Denial of Service (DDoS) attack on your own database or API. Conversely, retrying a non-retryable error (like a `400 Bad Request` or a syntax error) is a waste of CPU and network bandwidth. 

This module teaches you how to build a production-grade resilience layer. You will implement precise exception filtering, exponential backoff with full jitter, a stateful circuit breaker, and a guarded execution wrapper that coordinates them safely.

## What you'll build

| Function / class | You'll implement | Shows up in real code as |
|---|---|---|
| `is_retryable` | Exception filtering | Deciding which errors (like network timeouts) to retry, and which (like validation errors) to fail immediately. |
| `exponential_delays` | Backoff with Jitter | Spacing out retries exponentially and adding randomness to prevent synchronized retry storms. |
| `retry` | Retry loop | Executing a function repeatedly on failure, sleeping between attempts, and re-raising the final exception. |
| `CircuitBreaker` | State machine | Tracking consecutive failures to temporarily block requests to a broken dependency. |
| `guarded_call` | Resilient execution | Combining retries and the circuit breaker to handle transient and systemic failures safely. |

---

## 1. Exception filtering

```python
try:
    perform_network_call()
except (ConnectionResetError, TimeoutError) as e:
    # These are safe to retry
    retry_call()
except ValueError as e:
    # This is a programming bug; retrying will never make it succeed!
    raise e
```

Not all exceptions are created equal. When a database query fails because of a lock conflict, retrying is a great idea. When it fails because of a syntax error in your SQL, retrying is useless. 

To handle this, we need a helper function `is_retryable(exc, retry_on, never_retry)` that inspects an exception instance. It checks if the exception is an instance of any class in `retry_on`, while explicitly excluding any classes in `never_retry`. The exclusion list takes absolute precedence: if an exception is a subclass of both a retryable type and a non-retryable type, it must not be retried.

**In `exercises.py`:** Implement `is_retryable`.

**Test it:** `python3 -m unittest test_exercises.TestIsRetryable -v`

---

## 2. Exponential backoff and full jitter

If a downstream service is struggling, retrying immediately puts more load on it, pushing it further into failure. Instead, we back off: we wait longer between each successive attempt.

The standard formula for exponential backoff is:
$$\text{delay} = \min(\text{base} \times \text{factor}^{\text{attempt}}, \text{cap})$$

However, if 100 containers all fail at the same time and use this exact formula, they will all back off and retry at the exact same moments. This is called the **thundering herd** problem. To break this synchronization, we introduce **Full Jitter**. Instead of sleeping for the exact calculated delay, we sleep for a random duration chosen uniformly between `0` and the calculated delay:
$$\text{sleep\_duration} = \text{random}(0, \text{delay})$$

**In `exercises.py`:** Implement `exponential_delays` to generate a list of delays for `attempts - 1` retries. It must support both deterministic backoff and randomized full jitter using an injected `random.Random` instance.

**Test it:** `python3 -m unittest test_exercises.TestExponentialDelays -v`

---

## 3. The retry loop

```python
attempt = 0
while True:
    try:
        return fn()
    except Exception as e:
        attempt += 1
        if attempt >= max_attempts:
            raise e
        sleep(delays[attempt - 1])
```

With filtering and delay generation ready, we can build the core `retry` loop. The function accepts a target function `fn`, a maximum number of attempts, a tuple of retryable exception classes, a `sleep` callback, and a pre-calculated list of delays.

**The Pitfall:** Candidates often write retry loops that swallow the original exception and raise a generic `RuntimeError("Failed after X attempts")`. This destroys the traceback and makes debugging in production impossible. Always re-raise the *last* actual exception raised by the target function, preserving its original type and traceback.

**In `exercises.py`:** Implement `retry`.

**Test it:** `python3 -m unittest test_exercises.TestRetry -v`

---

## 4. The Circuit Breaker state machine

Retries are perfect for short, transient hiccups. But what if the downstream service goes down completely for 10 minutes? Retrying every single request will exhaust your application thread pool, slow down your users, and hammer the dead service, preventing it from recovering.

A **Circuit Breaker** solves this by wrapping calls in a state machine:
1. **CLOSED**: Normal operation. All requests flow through. If `failure_threshold` consecutive failures occur, the circuit trips and transitions to **OPEN**.
2. **OPEN**: Fast-fail mode. All requests are blocked immediately by raising a `CircuitOpenError` without calling the underlying service. It stays open for a `recovery_timeout` period.
3. **HALF-OPEN**: After the timeout elapses, the breaker allows a single trial request. If it succeeds, the circuit closes again. If it fails, it trips back to **OPEN** immediately.

```
       +---------+  consecutive failures  +--------+
       |         | ---------------------> |        |
  +--> | CLOSED  |                        |  OPEN  |
  |    |         | <--------------------- |        |
  |    +---------+     trial success      +--------+
  |                                           |
  |                                           | timeout elapsed
  |            +-----------+                  |
  +------------| HALF-OPEN | <----------------+
               +-----------+
```

To make this testable without waiting for real-world wall clock time, we inject a `clock` function (which returns a float timestamp) into the constructor.

**In `exercises.py`:** Implement the `CircuitBreaker` class and the custom `CircuitOpenError` exception.

**Test it:** `python3 -m unittest test_exercises.TestCircuitBreaker -v`

---

## 5. Guarding the call

Now, we combine our stateful `CircuitBreaker` and our `retry` loop into a single, highly resilient execution wrapper: `guarded_call`.

When a request comes in, `guarded_call` performs these steps:
1. Checks if the circuit breaker allows the request. If the breaker is `OPEN`, it raises `CircuitOpenError` immediately (no attempts are made, no retries run).
2. If allowed, it executes the operation using the retry logic.
3. If the operation ultimately succeeds (even if it took a couple of retries), we record a success on the breaker and return the result.
4. If the operation ultimately fails (either by exhausting all retries or encountering a non-retryable exception), we record a failure on the breaker and propagate the exception.

**The Pitfall:** Do not record a failure on the circuit breaker for *every* individual transient retry attempt. If a call fails twice but succeeds on the third retry, the overall operation was successful! The circuit breaker should only count it as a failure if the entire, retried call sequence fails.

**In `exercises.py`:** Implement `guarded_call`.

**Test it:** `python3 -m unittest test_exercises.TestGuardedCall -v`

---

## The exercises

| Function / Class | What it teaches |
|---|---|
| `is_retryable` | Precise exception filtering and subclass matching. |
| `exponential_delays` | Generating exponential backoff intervals with optional full jitter. |
| `retry` | Writing a robust retry loop that preserves tracebacks. |
| `CircuitBreaker` | Implementing a state machine with dynamic time-based transitions. |
| `guarded_call` | Coordinating retries and circuit breakers without double-counting failures. |

## Check your understanding

1. **Why does `never_retry` take precedence over `retry_on` in `is_retryable`?**
   <details><summary>Answer</summary>Because you might want to retry a general class of errors (like <code>OSError</code>) but explicitly avoid retrying a specific subclass (like <code>ConnectionRefusedError</code>) if you know it indicates a permanent configuration problem.</details>

2. **What is the difference between "Equal Jitter" and "Full Jitter"?**
   <details><summary>Answer</summary>Equal Jitter keeps a portion of the exponential backoff constant and adds a small random amount. Full Jitter chooses a random value uniformly between 0 and the maximum exponential backoff value. Full Jitter is proven to prevent thundering herds most effectively because it spreads the load completely across the entire backoff window.</details>

3. **Why must the `CircuitBreaker` state transition to HALF-OPEN be evaluated dynamically?**
   <details><summary>Answer</summary>If we relied on a background thread or timer to transition the state from OPEN to HALF-OPEN, we would introduce unnecessary resource overhead and synchronization bugs. By checking the current timestamp from <code>clock()</code> inside <code>allow_request()</code>, we can lazily and safely transition the state on-demand.</details>

4. **If a call succeeds on its 3rd retry attempt, what does the Circuit Breaker record?**
   <details><summary>Answer</summary>It records a single success. The internal transient failures are managed and absorbed by the retry loop. The circuit breaker only cares about the final outcome of the high-level operation.</details>

## Interview questions this module answers

- *How do you prevent a "thundering herd" or "retry storm" when calling a flaky service?*
- *What is a Circuit Breaker, and what are its three states?*
- *How do you implement a state machine that transitions based on elapsed time without using background threads?*
- *Why should you avoid recording a failure on a circuit breaker for every transient retry?*
