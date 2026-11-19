# 40 — Tools and their schemas

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/agents-lab/40-tool-schemas/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/agents-lab/tool-schemas?utm_source=github&utm_medium=labs)

*~45 minutes · 20 tests · Intermediate*

In an AI agent system, the Large Language Model (LLM) does not execute code directly. Instead, it acts as a reasoning engine that decides *which* tool to call and *what* arguments to pass to it. To make this decision, the model needs to know what tools are available, what they do, and what arguments they expect. We provide this metadata using schemas—typically a subset of JSON Schema.

However, LLMs are notorious for making mistakes when generating tool calls. They might hallucinate arguments, pass strings instead of numbers (e.g., `"5"` instead of `5`), forget required fields, or pass invalid values for enum constraints. If you pass these raw, unchecked inputs directly to your Python functions, your application will crash, raise unhandled exceptions, or worse, execute dangerous operations with corrupted data.

In this unit, you will build a robust tool execution and validation layer. This layer acts as a guardrail: it describes tools clearly to the model, intercepts and repairs common LLM formatting mistakes, and strictly validates arguments before they ever reach your execution code.

*Note on Offline Testing:* In this course, all tests run offline. We do not make real API calls to OpenAI, Anthropic, or any other provider. Instead, we use a deterministic fake model or test the tool execution layer directly. This ensures your tests are fast, cheap, and test your engineering logic rather than the model's mood. However, we will show you exactly where a real LLM client plugs in.

## What you'll build

| Function / class | You'll implement | Shows up in real code as |
|---|---|---|
| `Tool` | A class representing an agent tool | The core definition of a tool, binding metadata, schemas, and executable code |
| `validate_args` | A strict JSON Schema-like validator | A guardrail that rejects invalid tool calls before they execute |
| `coerce_args` | A robust type-coercion helper | A parser that fixes common LLM mistakes like passing numbers as strings |
| `ToolRegistry` | A central tool manager and prompt generator | The registry that exposes tools to the LLM and executes them safely |

---

## 1. Defining Tools with Schemas

```python
class Tool:
    def __init__(self, name: str, description: str, parameters: dict, func: Callable):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.func = func
```

To let an LLM know a tool exists, we must package the executable Python function with its metadata. The `parameters` dictionary is a JSON Schema object describing the expected arguments. For example, a tool that calculates tax might have a schema like this:

```python
{
    "type": "object",
    "properties": {
        "price": {"type": "number"},
        "state": {"type": "string", "enum": ["NY", "CA", "TX"]}
    },
    "required": ["price"]
}
```

By packaging these elements together, we create a self-documenting unit of execution. The LLM reads the `description` and `parameters` to understand *when* and *how* to use the tool, while our application uses the `func` to actually run it.

**In `exercises.py`:** Implement the `Tool` class constructor.

**Test it:** `python3 -m unittest test_exercises.TestToolClass -v`

---

## 2. Validating Arguments Strictly

```python
def validate_args(schema: dict, args: dict) -> None:
    # Check required fields, types, and enums
    ...
```

Before running any tool, we must verify that the arguments provided by the LLM match the schema. If validation fails, we raise a descriptive `ValueError`. This error can be caught and sent back to the LLM, allowing the agent to self-correct and try again.

### The Pitfall: Python's Boolean Type Trap
In Python, `bool` is a subclass of `int`. This means `isinstance(True, int)` evaluates to `True`! If your schema expects an `integer` or a `number`, and the LLM passes `True`, a naive type check will let it pass. You must explicitly check for and reject booleans when validating numeric types.

```python
# The trap:
>>> isinstance(True, int)
True

# The fix:
>>> isinstance(True, int) and not isinstance(True, bool)
False
```

**In `exercises.py`:** Implement `validate_args` to check for:
1. Missing required fields.
2. Unknown arguments (arguments passed but not defined in `properties`).
3. Type mismatches (supporting `string`, `number`, `integer`, `boolean`, and `array`).
4. Enum constraints.

**Test it:** `python3 -m unittest test_exercises.TestValidateArgs -v`

---

## 3. Coercing LLM Outputs Safely

```python
>>> coerce_args(schema, {"price": "42.5", "state": "NY"})
{'price': 42.5, 'state': 'NY'}
```

LLMs frequently output numeric values as strings (e.g., `"42"` or `"3.14"`) or booleans as strings (e.g., `"true"`). Instead of failing immediately, a robust agent framework attempts to **coerce** these values to their correct types if they are safely convertible.

If coercion fails (e.g., trying to convert `"hello"` to an integer), we leave the value as-is and let the strict validation step catch and report the error. This keeps our coercion logic safe and non-destructive. Crucially, coercion must not mutate the original dictionary; it should return a new one.

**In `exercises.py`:** Implement `coerce_args` to return a new dictionary with safely coerced values for `integer`, `number`, and `boolean` types.

**Test it:** `python3 -m unittest test_exercises.TestCoerceArgs -v`

---

## 4. The Tool Registry and System Prompts

```python
class ToolRegistry:
    def __init__(self):
        self.tools = {}

    def register(self, tool: Tool):
        ...
```

A `ToolRegistry` acts as the central coordinator. It has three main jobs:
1. **Manage tools**: Register and retrieve tools by name.
2. **Execute safely**: Retrieve a tool, coerce its arguments, validate them, and then execute the underlying function.
3. **Generate prompts**: Render a clean, human-readable text description of all tools and their schemas. This text is injected directly into the system prompt so the LLM knows how to call them.

To ensure deterministic output, the `describe()` method must sort the tools alphabetically by name, and sort each tool's parameters alphabetically by parameter name.

### Connecting to a Real LLM Client
In a production application, you would use this registry to generate system instructions and execute the resulting calls. Here is how a real client (like OpenAI) would plug into your registry:

```python
# 1. Generate the system prompt with tool descriptions
system_prompt = f"""You are a helpful assistant with access to these tools:

{registry.describe()}

To call a tool, output a JSON block like: {{"tool": "tool_name", "args": {{...}}}}"""

# 2. Send to the LLM
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "What is the tax on 100 dollars in NY?"}
    ]
)

# 3. Parse and execute
tool_call = parse_json(response.choices[0].message.content)
result = registry.call(tool_call["tool"], tool_call["args"])
```

**In `exercises.py`:** Implement `ToolRegistry` with `register`, `get`, `call`, and `describe` methods.

**Test it:** `python3 -m unittest test_exercises.TestToolRegistry -v`

---

## The exercises

| Function / Class | What it teaches |
|---|---|
| `Tool` | Packaging executable code with JSON Schema metadata |
| `validate_args` | Strict validation of types, enums, and required fields |
| `coerce_args` | Resilient parsing of stringified LLM outputs |
| `ToolRegistry` | Centralized tool execution and prompt generation |

## Check your understanding

1. **Why does `isinstance(True, int)` return `True` in Python, and how does it affect tool validation?**
   <details><summary>Answer</summary>Historically, Python did not have a boolean type, and used <code>0</code> and <code>1</code>. When <code>bool</code> was introduced, it was subclassed from <code>int</code> for backwards compatibility. If you do not explicitly check for booleans, a tool expecting an integer will accept <code>True</code> or <code>False</code>, leading to unexpected behavior.</details>

2. **Why should we coerce arguments before validating them?**
   <details><summary>Answer</summary>LLMs frequently output numeric values as strings (e.g., <code>"10"</code>). If we validate first, the validation will fail because a string is not an integer. By coercing first, we convert <code>"10"</code> to <code>10</code>, allowing validation to pass and the tool to run successfully.</details>

3. **What happens if coercion fails (e.g., converting <code>"abc"</code> to an integer)?**
   <details><summary>Answer</summary>Coercion should fail silently and leave the value as-is. This allows the subsequent strict validation step to raise a clear, standardized <code>ValueError</code> explaining exactly what type was expected.</details>

4. **Why do we sort tools and parameters alphabetically in the registry's `describe()` method?**
   <details><summary>Answer</summary>LLM prompts should be deterministic. If the order of tools or parameters changes randomly between runs, it can cause subtle variations in the LLM's behavior and make debugging or caching responses incredibly difficult.</details>

5. **Why is it important that `coerce_args` does not mutate the input dictionary?**
   <details><summary>Answer</summary>Immutability prevents side effects. If the original arguments are needed later (for example, to log the raw LLM output or to send the exact failed payload back to the model for debugging), mutating them in-place would destroy that original context.</details>

## Interview questions this module answers

- *How do you build a secure execution environment for an LLM agent's tools?*
- *Why is type validation critical when handling structured outputs from generative models?*
- *How do you handle common formatting errors (like stringified numbers) in LLM-generated JSON?*
- *How do you dynamically expose Python functions to an LLM as system instructions?*
