"""Tests for 45 - Guardrails, budgets and evaluation."""

import importlib
import os
import unittest

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestRetryWithBackoff(unittest.TestCase):
    def test_immediate_success(self):
        calls = 0
        def target():
            nonlocal calls
            calls += 1
            return "success"
        res = m.retry_with_backoff(target)
        self.assertEqual(res, "success")
        self.assertEqual(calls, 1)

    def test_success_after_retries(self):
        calls = 0
        def target():
            nonlocal calls
            calls += 1
            if calls < 3:
                raise ValueError("temporary error")
            return "success"

        sleeps = []
        def mock_sleep(t):
            sleeps.append(t)

        res = m.retry_with_backoff(
            target,
            max_retries=3,
            initial_delay=1.0,
            backoff_factor=2.0,
            jitter=False,
            sleep_fn=mock_sleep
        )
        self.assertEqual(res, "success")
        self.assertEqual(calls, 3)
        self.assertEqual(sleeps, [1.0, 2.0])

    def test_max_retries_exceeded(self):
        calls = 0
        def target():
            nonlocal calls
            calls += 1
            raise ValueError(f"error {calls}")

        sleeps = []
        def mock_sleep(t):
            sleeps.append(t)

        with self.assertRaises(ValueError) as ctx:
            m.retry_with_backoff(
                target,
                max_retries=2,
                initial_delay=1.0,
                backoff_factor=2.0,
                jitter=False,
                sleep_fn=mock_sleep
            )
        self.assertEqual(str(ctx.exception), "error 3")
        self.assertEqual(calls, 3)
        self.assertEqual(sleeps, [1.0, 2.0])

    def test_jitter_applied(self):
        calls = 0
        def target():
            nonlocal calls
            calls += 1
            raise ValueError("fail")

        sleeps = []
        def mock_sleep(t):
            sleeps.append(t)

        def mock_rng(a, b):
            return b * 0.5

        with self.assertRaises(ValueError):
            m.retry_with_backoff(
                target,
                max_retries=2,
                initial_delay=2.0,
                backoff_factor=3.0,
                jitter=True,
                sleep_fn=mock_sleep,
                rng=mock_rng
            )
        self.assertEqual(sleeps, [1.0, 3.0])

    def test_zero_retries(self):
        calls = 0
        def target():
            nonlocal calls
            calls += 1
            raise ValueError("fail")

        with self.assertRaises(ValueError):
            m.retry_with_backoff(target, max_retries=0)
        self.assertEqual(calls, 1)


class TestBudget(unittest.TestCase):
    def test_step_budget_ok(self):
        b = m.Budget(max_steps=3)
        b.add_step()
        b.add_step()
        b.add_step()
        self.assertEqual(b.steps, 3)

    def test_step_budget_exceeded(self):
        b = m.Budget(max_steps=2)
        b.add_step()
        b.add_step()
        with self.assertRaises(m.BudgetExceeded) as ctx:
            b.add_step()
        self.assertIn("Step budget exceeded", str(ctx.exception))

    def test_token_budget_exceeded(self):
        b = m.Budget(max_tokens=100)
        b.add_tokens(50, 40)
        with self.assertRaises(m.BudgetExceeded) as ctx:
            b.add_tokens(10, 5)
        self.assertIn("Token budget exceeded", str(ctx.exception))

    def test_cost_budget_exceeded(self):
        cost_map = {"prompt": 0.002, "completion": 0.015}
        b = m.Budget(max_cost=0.05, token_cost_map=cost_map)
        b.add_tokens(10, 2)
        self.assertAlmostEqual(b.cost, 0.05)
        with self.assertRaises(m.BudgetExceeded) as ctx:
            b.add_tokens(1, 0)
        self.assertIn("Cost budget exceeded", str(ctx.exception))

    def test_no_limits(self):
        b = m.Budget()
        for _ in range(100):
            b.add_step()
        b.add_tokens(10000, 10000)
        self.assertEqual(b.steps, 100)
        self.assertEqual(b.tokens, 20000)
        self.assertEqual(b.cost, 0.0)


class TestRedactPII(unittest.TestCase):
    def test_redact_emails(self):
        text = "My email is test@example.com, write to support@corp.co.uk."
        expected = "My email is [EMAIL], write to [EMAIL]."
        self.assertEqual(m.redact_pii(text), expected)

    def test_redact_phones(self):
        text = "Call me at +1-555-123-4567 or (555) 987-6543 or 555-4321."
        expected = "Call me at [PHONE] or [PHONE] or [PHONE]."
        self.assertEqual(m.redact_pii(text), expected)

    def test_redact_cards(self):
        text = "Visa: 1234-5678-1234-5678, Amex: 123456789012345."
        expected = "Visa: [CARD], Amex: [CARD]."
        self.assertEqual(m.redact_pii(text), expected)

    def test_mixed_pii(self):
        text = "User john@gmail.com (phone 555-123-4567) paid with card 1111222233334444."
        expected = "User [EMAIL] (phone [PHONE]) paid with card [CARD]."
        self.assertEqual(m.redact_pii(text), expected)


class TestToolPolicy(unittest.TestCase):
    def test_default_policy(self):
        policy = m.ToolPolicy()
        self.assertEqual(policy.check_tool("any_tool"), "allow")

    def test_allowed_list(self):
        policy = m.ToolPolicy(allowed_tools=["search", "calc"])
        self.assertEqual(policy.check_tool("search"), "allow")
        self.assertEqual(policy.check_tool("calc"), "allow")
        self.assertEqual(policy.check_tool("delete_db"), "deny")

    def test_denied_list(self):
        policy = m.ToolPolicy(denied_tools=["delete_db", "format"])
        self.assertEqual(policy.check_tool("search"), "allow")
        self.assertEqual(policy.check_tool("delete_db"), "deny")

    def test_require_approval(self):
        policy = m.ToolPolicy(
            allowed_tools=["search", "send_email"],
            require_approval_tools=["send_email"]
        )
        self.assertEqual(policy.check_tool("search"), "allow")
        self.assertEqual(policy.check_tool("send_email"), "approve")


class TestEvaluateAgent(unittest.TestCase):
    def test_all_success(self):
        def runner(prompt):
            return {"success": True, "output": f"Result for {prompt}", "steps": 2, "tokens": 40, "error": None}

        cases = [
            {"prompt": "Case 1", "expected": "Result"},
            {"prompt": "Case 2", "expected": "Result"}
        ]
        res = m.evaluate_agent(runner, cases)
        self.assertEqual(res["success_rate"], 1.0)
        self.assertEqual(res["mean_steps"], 2.0)
        self.assertEqual(res["total_tokens"], 80)
        self.assertEqual(res["failures"], {})

    def test_partial_success_with_mismatch(self):
        def runner(prompt):
            if "fail" in prompt:
                return {"success": False, "output": "", "steps": 1, "tokens": 10, "error": "API_TIMEOUT"}
            return {"success": True, "output": "wrong output", "steps": 3, "tokens": 30, "error": None}

        cases = [
            {"prompt": "fail prompt"},
            {"prompt": "good prompt", "expected": "correct"}
        ]
        res = m.evaluate_agent(runner, cases)
        self.assertEqual(res["success_rate"], 0.0)
        self.assertEqual(res["mean_steps"], 2.0)
        self.assertEqual(res["total_tokens"], 40)
        self.assertEqual(res["failures"], {"API_TIMEOUT": 1, "output_mismatch": 1})

    def test_runner_exception_handling(self):
        def runner(prompt):
            if "crash" in prompt:
                raise RuntimeError("System crash")
            return {"success": True, "output": "ok", "steps": 1, "tokens": 5, "error": None}

        cases = [
            {"prompt": "crash prompt"},
            {"prompt": "normal prompt"}
        ]
        res = m.evaluate_agent(runner, cases)
        self.assertEqual(res["success_rate"], 0.5)
        self.assertEqual(res["failures"], {"RuntimeError": 1})

    def test_empty_test_cases(self):
        def runner(prompt):
            return {"success": True}
        res = m.evaluate_agent(runner, [])
        self.assertEqual(res["success_rate"], 0.0)
        self.assertEqual(res["mean_steps"], 0.0)
        self.assertEqual(res["total_tokens"], 0)
        self.assertEqual(res["failures"], {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
