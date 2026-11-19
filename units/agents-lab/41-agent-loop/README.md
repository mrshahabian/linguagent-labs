# 41 — The agent loop

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/agents-lab/41-agent-loop/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/agents-lab/agent-loop?utm_source=github&utm_medium=labs)

*~45 minutes · 21 tests · Intermediate*

At the heart of every autonomous AI agent is a simple, relentless loop: the agent receives a goal, thinks, decides to call a tool, observes the tool's output, and repeats until it decides it has reached the goal. While modern frameworks hide this loop behind layers of complex abstractions, building it yourself from scratch is the only way to truly understand how agents work, how they fail, and how to keep them from running up massive API bills.

In production, an unguided agent loop can easily get stuck in infinite cycles, crash on minor tool errors, or run indefinitely. This unit teaches you how to build a robust, production-grade agent loop from scratch. You will implement deterministic fakes to test your agent offline, handle tool errors gracefully without crashing, and build safety guardrails to stop runaway loops.

## What you'll build

| Function / class | You'll implement | Shows up in real code as |
|---|---|---|
| Message Helpers | Functions to format system, user, assistant, and tool messages | Standardizing payloads for OpenAI/Anthropic Chat Completion APIs |
| `ScriptedModel` | A deterministic mock LLM client that records inputs and yields queued responses | Unit testing LLM agents reliably and offline without API keys |
| `run_agent` | The core execution loop with step limits and loop detection | The central runner of any autonomous agent framework (e.g., LangChain, CrewAI) |

---

## 1. Structuring messages for the loop

To run an agent loop, we must maintain a conversation history that the LLM can read and append to. Modern chat APIs expect messages to follow a strict schema of roles:
- `system`: Sets the agent's persona and instructions.
- `user`: Represents the human's goal or query.
- `assistant`: Represents the agent's thoughts, final answers, or requests to call tools.
- `tool`: Represents the output (observation) of a tool execution.

```python
# Example of a tool call and its corresponding observation message
assistant_message = {
    "role": "assistant",
    "content": "I need to look up the weather.",
    "tool_calls": [
        {"id": "call_123", "name": "get_weather", "arguments": {"location": "Chicago"}}
    ]
}

tool_response_message = {
    "role": "tool",
    "tool_call_id": "call_123",
    "content": "Rainy, 55°F"
}
```

**The Pitfall:** A common mistake is forgetting to link the `tool` message back to the assistant's request. Chat APIs will reject your payload if a `tool` message does not contain a `tool_call_id` matching an active tool call from the preceding assistant message. Additionally, some APIs require the `content` field of an assistant message to exist (even as `None`) when `tool_calls` are present.

**In `exercises.py`:** Implement the message helper functions (`system_msg`, `user_msg`, `assistant_msg`, `tool_msg`) to standardize message creation.

**Test it:** `python3 -m unittest test_exercises.TestMessageHelpers -v`

---

## 2. Mocking the LLM: The ScriptedModel

Testing an agent loop with a live LLM is slow, expensive, and non-deterministic. If the model changes its mind or the API has latency, your tests will fail randomly. To build a robust loop, we need a deterministic fake model that behaves exactly like an LLM but returns pre-programmed responses.

Our `ScriptedModel` will take a list of predefined `AgentResponse` objects. Each time the agent loop calls the model, it pops the next response from the queue and returns it. Crucially, it must also record the exact history of messages it was sent, allowing us to assert that our agent is passing the correct context.

```python
# Conceptual usage of a scripted model
model = ScriptedModel([
    AgentResponse(content="I will check the database.", tool_calls=[...]),
    AgentResponse(content="The database says we have 10 items in stock.")
])
```

**The Pitfall:** When recording the history of messages sent to the model, you must save a *deep copy* of the list. If your agent loop later mutates the active history list (which it will, by appending new messages), your recorded history will mutate too, ruining your test assertions and making it impossible to verify what the model saw at each step.

**In `exercises.py`:** Implement the `ScriptedModel` class.

**Test it:** `python3 -m unittest test_exercises.TestScriptedModel -v`

---

## 3. The Agent Loop: run_agent

Now, we assemble the pieces into `run_agent(model, registry, goal, max_steps)`. This function orchestrates the entire lifecycle:
1. Initialize the history with a standard system prompt and the user's goal.
2. In a loop, pass the history to the model to get its next response.
3. Append the model's response to the history.
4. If the model has no tool calls, it has reached a final answer. Stop and return.
5. If the model requests tool calls, execute them using the `ToolRegistry`.
6. Append each tool's output back to the history as a `tool` message, then repeat.

```python
# A simple loop structure
for step in range(max_steps):
    response = model(history)
    history.append(assistant_msg(response.content, response.tool_calls))
    if not response.tool_calls:
        return {"status": "final_answer", "history": history, "steps": step + 1}
    # ... execute tools and append observations ...
```

To make this production-ready, we must add two critical guardrails:
- **Graceful Error Handling:** If a tool raises an exception, the agent shouldn't crash. Instead, catch the error, format it as a string (e.g., `"Error: <message>"`), and feed it back to the model as a tool observation. This allows the model to try a different approach or report the error.
- **Infinite Loop Prevention:** If the model requests the exact same tool call with the exact same arguments twice in the same run, it is stuck in a loop. Detect this and stop immediately with a status of `"repeated_tool_call"`.

**In `exercises.py`:** Implement `run_agent` with all stop conditions and error handling.

**Test it:** `python3 -m unittest test_exercises.TestRunAgent -v`

---

## The exercises

| Function / Class | What it teaches |
|---|---|
| `system_msg`, `user_msg`, `assistant_msg`, `tool_msg` | Standardizing message schemas for Chat Completion APIs |
| `ScriptedModel` | Mocking LLM clients deterministically for offline testing |
| `run_agent` | Orchestrating tool execution, handling errors, and enforcing loop guardrails |

## Check your understanding

1. **Why do tool messages require a `tool_call_id`?**
   <details><summary>Answer</summary>LLMs can generate multiple tool calls in parallel. The <code>tool_call_id</code> allows the model to map each observation back to the specific tool call that requested it.</details>

2. **Why should tool execution errors be caught and fed back to the model instead of crashing?**
   <details><summary>Answer</summary>Tools interact with the real world (databases, APIs, file systems) and will occasionally fail. Feeding the error back to the model as an observation allows the agent to self-correct, try different arguments, or explain the failure to the user.</details>

3. **What is the purpose of tracking repeated tool calls?**
   <details><summary>Answer</summary>If an LLM receives a tool output that doesn't satisfy its expectations, it might try calling the exact same tool with the exact same arguments again. Without a guardrail, this creates an infinite loop that drains your API budget.</details>

4. **How does `ScriptedModel` help with testing?**
   <details><summary>Answer</summary>It makes the LLM's behavior 100% deterministic, runs instantly without network latency, costs nothing, and records the exact messages sent to it so you can assert your agent's prompt engineering is correct.</details>

5. **What happens if the agent loop hits `max_steps`?**
   <details><summary>Answer</summary>The loop terminates safely with a status of <code>"max_steps"</code>, returning the transcript so far. This prevents runaway execution if the model fails to reach a final answer.</details>

## Interview questions this module answers

- *How do you design a resilient execution loop for an autonomous AI agent?*
- *How do you prevent AI agents from getting stuck in infinite loops?*
- *How do you write deterministic unit tests for LLM-based applications without calling live APIs?*
- *How should an agent handle tool execution failures?*
