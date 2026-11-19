# 45 — Guardrails, budgets and evaluation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/agents-lab/45-reliability/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/agents-lab/reliability?utm_source=github&utm_medium=labs)

*~45 minutes · 22 tests · Intermediate*

Building an AI agent that works on your local machine with a few hand-crafted prompts is a great milestone. However, taking that agent into production requires a completely different engineering mindset. In the wild, agents face unpredictable user inputs, transient network failures, rate limits, and the risk of infinite loops that can drain your API budget in minutes. Without robust guardrails, your agent is a liability.

This unit focuses on the critical engineering practices needed to make AI agents safe, cost-controlled, and measurable. You will build the core safety infrastructure that wraps around an agent's LLM client and tool execution loop. This includes exponential backoff with jitter to handle rate limits, resource budgets to prevent runaway execution, PII redaction to protect user privacy, tool execution policies to secure system actions, and an automated evaluation harness to measure performance scientifically.

Every component in this unit runs offline: the 'model' is a deterministic fake (a scripted or rule-based object with the same interface a real LLM client has), so the tests pin the agent's engineering, not a model's mood. In production, a real LLM client (like OpenAI or Anthropic) simply plugs in where the fake is used.

## What you'll build

Five core components in `exercises.py` that form the safety and measurement suite for any production-grade agent:

| Function / class | You'll implement | Shows up in real code as |
|---|---|---|
| `retry_with_backoff` | Exponential backoff with random jitter | Resilient API client wrappers that handle rate limits (HTTP 429) gracefully |
| `Budget` | Step, token, and cost tracking with circuit breakers | Financial and operational guardrails to prevent runaway loops |
| `redact_pii` | Regex-based data sanitization | Compliance layers that scrub sensitive data before it leaves your network |
| `ToolPolicy` | Allow/deny lists and approval gates | Security middleware restricting what actions an agent can perform |
| `evaluate_agent` | Automated evaluation and metrics aggregator | CI/CD testing pipelines that measure agent accuracy and performance |

---

## 1. Retries with exponential backoff and jitter

When calling external LLM APIs, transient failures are inevitable. You will encounter network timeouts, temporary server overloads, and rate limits (HTTP 429). A naive retry loop that retries immediately will only worsen the server's load and exhaust your retry quota instantly.

To handle this, production systems use **exponential backoff**. The delay after failure $i$ is calculated as:

$$\text{delay} = \text{initial\_delay} \times \text{backoff\_factor}^{(i-1)}$$

However, if multiple client instances fail at the same time, they will calculate the exact same backoff delays and retry simultaneously. This is known as the **thundering herd problem**. To prevent this, we introduce **jitter**—randomizing the delay uniformly between `0` and the calculated maximum delay.

```python
# Concept: Calculating backoff with jitter
base_delay = initial_delay * (backoff_factor ** (attempt - 1))
actual_delay = random.uniform(0.0, base_delay) if jitter else base_delay
```

To make this function fully testable and deterministic, you will inject `sleep_fn` (defaulting to `time.sleep`) and `rng` (defaulting to `random.uniform`).

**Pitfall:** A common mistake is forgetting that `max_retries` is the number of *retries*, meaning the function can be called up to `max_retries + 1` times in total. Another pitfall is applying jitter incorrectly (e.g., adding jitter to the backoff instead of scaling the backoff, or not handling the case where `jitter` is False).

**In `exercises.py`:** Implement `retry_with_backoff` to wrap any callable and retry it upon exceptions.

**Test it:** `python3 -m unittest test_exercises.TestRetryWithBackoff -v`

---

## 2. Token, step, and cost budgeting

AI agents typically run in an autonomous loop: they observe, plan, call tools, and repeat. If an agent gets confused—for example, if a tool returns an unexpected error format—the agent might get stuck in an infinite loop, repeatedly calling the same tool or generating the same invalid response.

Because LLM APIs charge per token, a runaway agent loop can consume millions of tokens and cost hundreds of dollars in a matter of minutes. You must implement a **Budget** class that acts as a circuit breaker.

```python
# Concept: Budget tracking
budget = Budget(max_steps=5, max_cost=0.05, token_cost_map={"prompt": 0.00001, "completion": 0.00003})
try:
    for step in range(10):
        budget.add_step()
        # ... call LLM ...
        budget.add_tokens(prompt_tokens=1000, completion_tokens=500)
except BudgetExceeded:
    print("Halted runaway agent!")
```

Your `Budget` class will track steps, total tokens, and total cost, raising a custom `BudgetExceeded` exception the moment any limit is crossed.

**Pitfall:** Allowing the agent to exceed the budget slightly before raising the exception, or not calculating the cost correctly based on the token cost map. The check must happen *during* or *immediately after* the resource is consumed, raising the exception immediately to halt execution.

**In `exercises.py`:** Implement the `Budget` class and the `BudgetExceeded` exception.

**Test it:** `python3 -m unittest test_exercises.TestBudget -v`

---

## 3. PII Redaction

Data privacy is a non-negotiable requirement for enterprise AI applications. Sending Personally Identifiable Information (PII) like customer emails, phone numbers, or credit card numbers to third-party LLM providers can violate regulations like GDPR, HIPAA, or PCI-DSS.

Before sending any text to an external model, you should pass it through a redaction guardrail. Regular expressions (regex) provide a fast, lightweight, and offline way to scrub sensitive patterns.

```python
# Concept: Regex redaction
email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
redacted_text = re.sub(email_pattern, "[EMAIL]", raw_text)
```

In this section, you will build a utility to redact emails, phone numbers, and credit card-like numbers (13 to 16 continuous digits, or blocks separated by spaces/hyphens). To avoid partial matching conflicts (e.g., matching a credit card number as a phone number), you must carefully order your redactions and use word boundaries.

**Pitfall:** Order of operations and greedy matching. If you try to match phone numbers before credit cards, or if your regex is too greedy, you might redact parts of a credit card as a phone number, or vice versa. Also, word boundaries (`\b`) are crucial to avoid matching random parts of larger strings (like a serial number matching a phone number pattern).

**In `exercises.py`:** Implement `redact_pii` using Python's `re` module.

**Test it:** `python3 -m unittest test_exercises.TestRedactPII -v`

---

## 4. Tool execution policies

Agents are powerful because they can execute tools to interact with the physical or digital world. However, giving an agent unrestricted access to all tools is extremely dangerous. A prompt injection attack could trick the agent into executing destructive actions, such as deleting database tables or sending unauthorized emails.

To secure tool execution, you must implement a `ToolPolicy` that acts as a gatekeeper. It checks tool names against:
1. An **allowlist** (if provided, only these tools can run).
2. A **denylist** (tools that are explicitly forbidden).
3. An **approval list** (tools that require human-in-the-loop authorization before running).

```python
# Concept: Policy checking
policy = ToolPolicy(allowed_tools=["search", "send_email"], require_approval_tools=["send_email"])
policy.check_tool("send_email")  # Returns "approve"
policy.check_tool("delete_db")   # Returns "deny"
```

**Pitfall:** Incorrectly handling default policies (e.g., if `allowed_tools` is `None`, any tool not explicitly denied should be allowed, but if `allowed_tools` is a list, only those tools are allowed). Also, order of checks matters: denylist must take precedence over allowlist, which must take precedence over approval list.

**In `exercises.py`:** Implement the `ToolPolicy` class.

**Test it:** `python3 -m unittest test_exercises.TestToolPolicy -v`

---

## 5. Agent evaluation harness

When you modify an agent's prompt, system instructions, or tool definitions, how do you know if you made the agent better or worse? Relying on "vibe checks" (manually testing 2 or 3 prompts) is highly subjective, error-prone, and doesn't scale.

Instead, you must run automated evaluations over a suite of test cases. An evaluation harness executes the agent over a set of prompts, compares the outputs to expected answers, tracks resource consumption (steps and tokens), and groups failures by reason.

```python
# Concept: Evaluation metrics
results = evaluate_agent(agent_runner, test_cases)
print(f"Success Rate: {results['success_rate']:.2%}")
print(f"Average Steps: {results['mean_steps']:.1f}")
```

Your evaluation harness will run a deterministic mock agent runner over test cases, aggregating metrics and categorizing errors (like API timeouts or output mismatches) so developers can pinpoint regressions.

**Pitfall:** Not handling exceptions raised by the agent runner itself (which should be caught and recorded as a failure with the exception class name as the reason), or failing to detect output mismatches when an expected answer is provided but missing from the agent's output.

**In `exercises.py`:** Implement the `evaluate_agent` function.

**Test it:** `python3 -m unittest test_exercises.TestEvaluateAgent -v`

---

## The exercises

| Function / Class | What it teaches |
|---|---|
| `retry_with_backoff` | Resilient API interaction with exponential delays and jitter |
| `Budget` | Resource tracking and circuit-breaking with custom exceptions |
| `redact_pii` | Regex-based data sanitization and compliance guardrails |
| `ToolPolicy` | Security policies and human-in-the-loop approval gates |
| `evaluate_agent` | Quantitative agent evaluation and regression testing |

## Check your understanding

1. **Why is random jitter crucial in exponential backoff algorithms?**
   <details><summary>Answer</summary>Without jitter, multiple clients that fail at the same time will retry at the exact same intervals, causing synchronized bursts of traffic (the thundering herd problem) that keep the server overloaded. Jitter spreads the retries smoothly over time.</details>

2. **What is the purpose of raising a custom `BudgetExceeded` exception instead of just returning a failure status?**
   <details><summary>Answer</summary>Raising an exception immediately halts execution and unwinds the call stack, acting as an absolute circuit breaker. This prevents any subsequent code or nested loops from continuing to consume resources.</details>

3. **Why is regex-based PII redaction considered a first line of defense rather than a complete solution?**
   <details><summary>Answer</summary>Regex is fast, offline, and highly effective for structured patterns (emails, credit cards). However, it can miss unstructured PII (like names or addresses in conversational context) and can produce false positives on non-sensitive numbers.</details>

4. **What is the difference between a denied tool and a tool requiring approval in a `ToolPolicy`?**
   <details><summary>Answer</summary>A denied tool is blocked immediately, returning an error to the agent. A tool requiring approval halts execution to ask a human operator for permission before proceeding, enabling safe execution of high-risk actions.</details>

5. **Why should an evaluation harness track metrics like `mean_steps` and `total_tokens` alongside `success_rate`?**
   <details><summary>Answer</summary>An agent might maintain a 100% success rate but become twice as expensive or twice as slow due to a prompt change. Tracking steps and tokens ensures you measure efficiency and cost alongside accuracy.</details>

## Interview questions this module answers

- *How do you implement exponential backoff and jitter to handle rate limits?*
- *How do you prevent autonomous agents from getting stuck in infinite, expensive loops?*
- *What strategies do you use to ensure user PII is not leaked to third-party LLM providers?*
- *How do you design a security policy for agentic tool execution?*
- *How do you quantitatively evaluate an LLM agent's performance and cost?*
