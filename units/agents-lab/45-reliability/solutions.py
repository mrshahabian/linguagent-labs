"""45 - Guardrails, budgets and evaluation

Reference implementation.
"""

import re
import time
import random
from typing import Callable, Dict, List, Any, Optional


class BudgetExceeded(Exception):
    """Raised when an agent exceeds its step, token, or monetary budget."""
    pass


def retry_with_backoff(
    func: Callable[[], Any],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    sleep_fn: Optional[Callable[[float], None]] = None,
    rng: Optional[Callable[[float, float], float]] = None
) -> Any:
    if sleep_fn is None:
        sleep_fn = time.sleep
    if rng is None:
        rng = random.uniform

    for attempt in range(1, max_retries + 2):
        try:
            return func()
        except Exception as e:
            if attempt > max_retries:
                raise e
            delay = initial_delay * (backoff_factor ** (attempt - 1))
            if jitter:
                delay = rng(0.0, delay)
            sleep_fn(delay)


class Budget:
    def __init__(
        self,
        max_steps: Optional[int] = None,
        max_tokens: Optional[int] = None,
        max_cost: Optional[float] = None,
        token_cost_map: Optional[Dict[str, float]] = None
    ):
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.max_cost = max_cost
        self.token_cost_map = token_cost_map or {"prompt": 0.0, "completion": 0.0}
        self.steps = 0
        self.tokens = 0
        self.cost = 0.0

    def add_step(self) -> None:
        self.steps += 1
        if self.max_steps is not None and self.steps > self.max_steps:
            raise BudgetExceeded(f"Step budget exceeded: {self.steps} > {self.max_steps}")

    def add_tokens(self, prompt_tokens: int, completion_tokens: int) -> None:
        added_tokens = prompt_tokens + completion_tokens
        self.tokens += added_tokens

        prompt_cost = prompt_tokens * self.token_cost_map.get("prompt", 0.0)
        completion_cost = completion_tokens * self.token_cost_map.get("completion", 0.0)
        self.cost += (prompt_cost + completion_cost)

        if self.max_tokens is not None and self.tokens > self.max_tokens:
            raise BudgetExceeded(f"Token budget exceeded: {self.tokens} > {self.max_tokens}")
        if self.max_cost is not None and self.cost > self.max_cost:
            raise BudgetExceeded(f"Cost budget exceeded: ${self.cost:.6f} > ${self.max_cost:.6f}")


def redact_pii(text: str) -> str:
    # 1. Card regex: 13 to 16 digits, with optional spaces or hyphens
    card_re = r"\b\d{4}[- ]\d{4}[- ]\d{4}[- ]\d{4}\b|\b\d{13,16}\b"
    text = re.sub(card_re, "[CARD]", text)

    # 2. Email regex: standard email matching, avoiding trailing dot consumption
    email_re = r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]*[a-zA-Z0-9]\b"
    text = re.sub(email_re, "[EMAIL]", text)

    # 3. Phone regex: matches US, international, and simple local formats
    phone_re = r"(?:\+?\d{1,3}[-.\s]+)?\(?\d{3}\)?[-.\s]*\d{3}[-.\s]+\d{4}\b|\b\d{3}[-.\s]\d{4}\b"
    text = re.sub(phone_re, "[PHONE]", text)

    return text


class ToolPolicy:
    def __init__(
        self,
        allowed_tools: Optional[List[str]] = None,
        denied_tools: Optional[List[str]] = None,
        require_approval_tools: Optional[List[str]] = None
    ):
        self.allowed_tools = set(allowed_tools) if allowed_tools is not None else None
        self.denied_tools = set(denied_tools) if denied_tools is not None else set()
        self.require_approval_tools = set(require_approval_tools) if require_approval_tools is not None else set()

    def check_tool(self, tool_name: str) -> str:
        if tool_name in self.denied_tools:
            return "deny"
        if self.allowed_tools is not None and tool_name not in self.allowed_tools:
            return "deny"
        if tool_name in self.require_approval_tools:
            return "approve"
        return "allow"


def evaluate_agent(
    agent_runner: Callable[[str], Dict[str, Any]],
    test_cases: List[Dict[str, Any]]
) -> Dict[str, Any]:
    if not test_cases:
        return {
            "success_rate": 0.0,
            "mean_steps": 0.0,
            "total_tokens": 0,
            "failures": {}
        }

    total_cases = len(test_cases)
    successful_runs = 0
    total_steps = 0
    total_tokens = 0
    failures = {}

    for case in test_cases:
        prompt = case["prompt"]
        expected = case.get("expected")

        try:
            result = agent_runner(prompt)
        except Exception as e:
            reason = type(e).__name__
            failures[reason] = failures.get(reason, 0) + 1
            continue

        steps = result.get("steps", 0)
        tokens = result.get("tokens", 0)
        success = result.get("success", False)
        output = result.get("output", "")
        error = result.get("error")

        total_steps += steps
        total_tokens += tokens

        is_success = success
        reason = None

        if not is_success:
            reason = error or "unknown_error"
        elif expected is not None and expected not in output:
            is_success = False
            reason = "output_mismatch"

        if is_success:
            successful_runs += 1
        else: 
            failures[reason] = failures.get(reason, 0) + 1

    return {
        "success_rate": successful_runs / total_cases,
        "mean_steps": total_steps / total_cases,
        "total_tokens": total_tokens,
        "failures": failures
    }
