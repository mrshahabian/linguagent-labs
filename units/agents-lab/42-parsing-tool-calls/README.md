# 42 — Parsing and repairing tool calls

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/agents-lab/42-parsing-tool-calls/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/agents-lab/parsing-tool-calls?utm_source=github&utm_medium=labs)

*~45 minutes · 20 tests · Intermediate*

In an ideal world, Large Language Models (LLMs) would always return perfectly formatted JSON that matches your exact schema. In the real world, models—especially smaller, faster, or open-source ones—frequently make minor formatting errors. They wrap JSON in markdown code fences, include conversational prose, use single quotes instead of double quotes, leave trailing commas, or use Python-specific literals like `None` and `True` instead of JSON's `null` and `true`.

If your agent loop crashes every time a model makes a minor syntax slip, your application will be fragile and frustrating. Instead of wasting API tokens and latency on re-prompting the model to "fix its formatting," you can build a robust parser that extracts, repairs, and validates tool calls deterministically.

In this unit, we build an offline, deterministic parser. Because we are testing the software engineering of the agent loop rather than the mood of a live model, we use structured text inputs to verify that our parser handles messy outputs gracefully. In production, this parser sits directly between your LLM client and your tool execution engine.

## What you'll build

One exception class and four parsing functions in `exercises.py` that turn raw, messy model outputs into structured, validated tool calls:

| Function / class | You'll implement | Shows up in real code as |
|---|---|---|
| `ToolCallError` | A custom exception class | Standardizing error handling when a model's output cannot be parsed or validated |
| `extract_json` | Extracting JSON from markdown fences and conversational filler | Pre-processing raw LLM text to isolate the structured payload |
| `repair_json` | Auto-correcting trailing commas, single quotes, and Python literals | A deterministic, zero-latency recovery layer for minor syntax errors |
| `parse_tool_call` | Validating tool names and arguments against allowed lists | The secure gatekeeper that ensures only valid, safe tools are executed |
| `parse_react` | Parsing Thought/Action/Action Input/Final Answer blocks | The core parser for ReAct (Reasoning + Acting) style agent loops |

---

## 1. Custom exceptions for robust handling

When building an agent, you must distinguish between **system errors** (like a database being down) and **agent errors** (like the model calling a tool that doesn't exist, or providing invalid arguments). If you use generic exceptions like `ValueError` or `KeyError` for both, your agent loop won't know whether to retry, re-prompt the model, or crash.

```python
# Example of a custom exception pattern
class ToolCallError(Exception):
    """Raised when a tool call cannot be parsed or validated."""
    pass
```

By raising a dedicated `ToolCallError`, your outer agent loop can catch this specific exception, format the error message, and feed it back to the model (e.g., *"Error: Tool 'calculate_tax' requires argument 'income'. Please try again."*). This allows the agent to self-correct without crashing the entire application.

**In `exercises.py`:** Implement the `ToolCallError` exception class.

**Test it:** `python3 -m unittest test_exercises.TestToolCallError -v`

---

## 2. Extracting JSON from messy prose

LLMs love to talk. Even when explicitly instructed to "return only JSON," they often wrap the JSON in markdown code blocks or add conversational filler:

```text
Sure! Here is the tool call you requested:
```json
{
  "name": "get_weather",
  "arguments": {"city": "Paris"}
}
```
Hope this helps!
```

To handle this, your parser must locate and extract the JSON payload. The standard strategy is to first look for markdown code blocks (specifically ` ```json ` or ` ``` `), and if none are found, fall back to finding the first opening brace `{` and the last closing brace `}`.

```python
# A simple extraction strategy
def extract_json_simple(text):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON found")
    return text[start:end+1]
```

**The Pitfall:** A naive substring search can fail if the model outputs multiple markdown blocks or if there are braces inside conversational text before the actual JSON block. Always prioritize explicit markdown code blocks first before falling back to raw brace matching.

**In `exercises.py`:** Implement `extract_json` to robustly extract the first JSON object string from surrounding text.

**Test it:** `python3 -m unittest test_exercises.TestExtractJson -v`

---

## 3. Repairing common LLM JSON mistakes

Even after extracting the JSON string, `json.loads()` will fail if the model made minor syntax errors. The three most common mistakes are:
1. **Trailing commas:** `{"a": 1, "b": 2,}` (valid in Python and JavaScript, but illegal in standard JSON).
2. **Single quotes:** `{'name': 'get_weather'}` (JSON strictly requires double quotes `"`).
3. **Python literals:** Using `True`, `False`, or `None` instead of `true`, `false`, or `null`.

Instead of failing, we can use regular expressions to repair these common issues before parsing:

```python
# Example of repairing trailing commas
import re
repaired = re.sub(r",\s*([\]}])", r"\1", "{\"a\": 1,}")
# repaired is now '{"a": 1}'
```

**The Pitfall:** Blindly replacing all single quotes with double quotes will break contractions inside text values (e.g., `'don't'` becomes `"don"t"`, which is invalid). A robust repairer must only replace single quotes that act as string delimiters—specifically, those next to braces, brackets, colons, commas, or whitespace.

**In `exercises.py`:** Implement `repair_json` to clean up trailing commas, Python literals, and delimiter-style single quotes.

**Test it:** `python3 -m unittest test_exercises.TestRepairJson -v`

---

## 4. Validating and parsing tool calls

Once you have extracted and repaired the JSON, you must validate its structure. A valid tool call must be a JSON object containing a tool name and its arguments. 

To make your agent robust, your parser should accept both standard formats:
- `{"name": "tool_name", "arguments": {...}}`
- `{"tool": "tool_name", "args": {...}}`

Additionally, you must validate that the requested tool is actually in your list of allowed tools. Executing arbitrary tool names provided by an LLM is a major security and stability risk.

```python
# Example validation check
if allowed_tools is not None and tool_name not in allowed_tools:
    raise ToolCallError(f"Tool '{tool_name}' is not allowed.")
```

**The Pitfall:** Assuming `arguments` is always present. Sometimes a model calls a tool with no arguments and omits the key entirely. Your parser should handle this gracefully by defaulting `arguments` to an empty dictionary `{}` rather than crashing.

**In `exercises.py`:** Implement `parse_tool_call` to extract, repair, load, and validate a tool call.

**Test it:** `python3 -m unittest test_exercises.TestParseToolCall -v`

---

## 5. Parsing ReAct (Reasoning and Acting) loops

ReAct is a popular agent pattern where the model interleaves reasoning steps ("Thought") with tool executions ("Action" and "Action Input") before returning a "Final Answer". The format is typically plain text with structured headers:

```text
Thought: I need to find the population of Paris.
Action: search_wikipedia
Action Input: {"query": "Paris population"}
```

Your parser needs to extract each of these blocks into a structured dictionary. Because these blocks can span multiple lines, simple line-by-line splitting is fragile. Instead, use regular expressions with lookaheads to capture everything between a header and the next block header.

```python
# Example lookahead pattern for ReAct blocks
pattern = r"Thought:\s*(.*?)(?=\n(?:Action|Final Answer):|$)"
```

**The Pitfall:** Assuming the model always outputs all blocks. In the final step of a loop, the model will output a `Thought` and a `Final Answer` but no `Action` or `Action Input`. Your parser must handle missing blocks gracefully, returning `None` for any block that wasn't generated.

**In `exercises.py`:** Implement `parse_react` to parse ReAct blocks into a structured dictionary.

**Test it:** `python3 -m unittest test_exercises.TestParseReact -v`

---

## The exercises

| Function / class | What it teaches |
|---|---|
| `ToolCallError` | Creating domain-specific exceptions for agent control flow |
| `extract_json` | Isolating JSON payloads from conversational text and markdown fences |
| `repair_json` | Using regular expressions to fix syntax errors like trailing commas and Python literals |
| `parse_tool_call` | Combining extraction, repair, and schema validation into a safe execution gate |
| `parse_react` | Parsing multi-line, structured text blocks in ReAct agent loops |

## Check your understanding

1. **Why should we raise a custom `ToolCallError` instead of a standard Python `ValueError`?**
   <details><summary>Answer</summary>A custom exception allows the calling agent loop to catch tool-specific failures specifically. This lets the system handle formatting errors or invalid tool names gracefully (e.g., by prompting the model to correct itself) while letting actual system bugs crash and raise alerts.</details>

2. **What is the risk of using a simple `text.find('{')` and `text.rfind('}')` as the primary JSON extraction strategy?**
   <details><summary>Answer</summary>If the model outputs conversational text containing braces before or after the JSON block (for example, "Here is your dictionary: {key: val}"), a simple brace-search will extract the wrong boundaries and fail to parse. Prioritizing explicit markdown fences (<code>```json</code>) is much safer.</details>

3. **How does `repair_json` avoid breaking contractions like "don't" when converting single quotes to double quotes?**
   <details><summary>Answer</summary>By using lookarounds or boundary checks to only replace single quotes that are adjacent to JSON structural characters (like <code>{</code>, <code>}</code>, <code>:</code>, <code>,</code>, <code>[</code>, <code>]</code>) or whitespace, leaving apostrophes inside words untouched.</details>

4. **Why should `parse_tool_call` default missing arguments to an empty dictionary?**
   <details><summary>Answer</summary>Models frequently omit the <code>arguments</code> or <code>args</code> key entirely when calling a tool that requires no parameters. Treating this as a fatal error breaks the agent unnecessarily; defaulting to <code>{}</code> is much more robust.</details>

5. **Why do we use lookaheads in the ReAct block parsing regular expressions?**
   <details><summary>Answer</summary>Lookaheads allow us to capture all text up to the next block header (like <code>Action:</code>) without consuming that header. This ensures each block matches its content fully, even if the content spans multiple lines.</details>

## Interview questions this module answers

- *How do you handle unstructured or semi-structured LLM outputs in a production pipeline?*
- *How can you make a JSON parser resilient to common syntax mistakes without re-prompting the model?*
- *What security considerations must you keep in mind when executing tool calls generated by an AI?*
- *How do you parse and structure multi-line text blocks from a ReAct agent loop?*
