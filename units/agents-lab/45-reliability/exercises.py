"""45 - Guardrails, budgets and evaluation

Implement each function and class so the tests in test_exercises.py pass.
Read README.md first; every exercise below is covered there.
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
    """Retry a function with exponential backoff and optional jitter.

    Args:
        func: The function to call.
        max_retries: Maximum number of retries (excluding the first attempt).
        initial_delay: Delay after the first failure.
        backoff_factor: Multiplier for the delay on subsequent failures.
        jitter: If True, add random jitter (delay is uniformly random between 0 and calculated delay).
        sleep_fn: Injected sleep function (defaults to time.sleep).
        rng: Injected random uniform function (defaults to random.uniform).
    """
    raise NotImplementedError("your code here")


class Budget:
    """Tracks steps, tokens, and cost, raising BudgetExceeded if limits are breached."""

    def __init__(
        self,
        max_steps: Optional[int] = None,
        max_tokens: Optional[int] = None,
        max_cost: Optional[float] = None,
        token_cost_map: Optional[Dict[str, float]] = None
    ):
        """Initialize the budget tracker.

        Args:
            max_steps: Maximum allowed steps.
            max_tokens: Maximum allowed total tokens (prompt + completion).
            max_cost: Maximum allowed total cost in USD.
            token_cost_map: Dict mapping "prompt" and "completion" to cost per token.
        """
        raise NotImplementedError("your code here")

    def add_step(self) -> None:
        """Increment step count and check step budget.

        Raises:
            BudgetExceeded: If steps exceed max_steps.
        """
        raise NotImplementedError("your code here")

    def add_tokens(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Add tokens, calculate cost, and check token and cost budgets.

        Raises:
            BudgetExceeded: If tokens exceed max_tokens or cost exceeds max_cost.
        """
        raise NotImplementedError("your code here")


def redact_pii(text: str) -> str:
    """Redact emails, phone numbers, and credit card-like numbers from text.

    Replaces:
        - Emails with '[EMAIL]'
        - Phone numbers with '[PHONE]'
        - Card numbers (13 to 16 digits, with optional spaces/hyphens) with '[CARD]'
    """
    raise NotImplementedError("your code here")


class ToolPolicy:
    """Enforces allow/deny lists and approval gates for agent tools."""

    def __init__(
        self,
        allowed_tools: Optional[List[str]] = None,
        denied_tools: Optional[List[str]] = None,
        require_approval_tools: Optional[List[str]] = None
    ):
        """Initialize the tool policy.

        Args:
            allowed_tools: If provided, only these tools are allowed.
            denied_tools: These tools are explicitly denied.
            require_approval_tools: These tools require human approval.
        """
        raise NotImplementedError("your code here")

    def check_tool(self, tool_name: str) -> str:
        """Check if a tool is allowed, denied, or requires approval.

        Returns:
            'allow', 'deny', or 'approve'
        """
        raise NotImplementedError("your code here")


def evaluate_agent(
    agent_runner: Callable[[str], Dict[str, Any]],
    test_cases: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Run evaluation over a suite of test cases and return aggregated metrics.

    Args:
        agent_runner: A callable taking a prompt and returning a dict with keys:
                      'success' (bool), 'output' (str), 'steps' (int), 'tokens' (int), 'error' (str or None).
        test_cases: A list of dicts, each containing 'prompt' (str) and optionally 'expected' (str).

    Returns:
        A dict with:
            'success_rate': float (0.0 to 1.0)
            'mean_steps': float
            'total_tokens': int
            'failures': dict mapping failure reasons to counts.
    """
    raise NotImplementedError("your code here")
