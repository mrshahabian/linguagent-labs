"""42 — Parsing and repairing tool calls

Reference solutions for the exercises.
"""

import json
import re
from typing import Any, Dict, List, Tuple, Optional


class ToolCallError(Exception): 
    """Raised when a tool call cannot be parsed or validated."""
    pass


def extract_json(text: str) -> str:
    # Try to extract from markdown code blocks first
    code_block_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1).strip()

    # Fallback to brace matching
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in text")
    
    return text[start:end+1].strip()


def repair_json(text: str) -> str:
    # 1. Replace Python literals with JSON equivalents
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(r"\bNone\b", "null", text)

    # 2. Replace single quotes with double quotes for delimiters
    text = re.sub(r"(?<=[\{\s,\[:])'|'(?=[\}\s,\]:])|^'|'$", '"', text)

    # 3. Replace escaped single quotes \' with ' since it's now double-quoted
    text = text.replace(r"\'", "'")

    # 4. Remove trailing commas before closing braces or brackets
    text = re.sub(r",\s*([\]}])", r"\1", text)

    return text


def parse_tool_call(text: str, allowed_tools: Optional[List[str]] = None) -> Tuple[str, Dict[str, Any]]:
    try:
        extracted = extract_json(text)
    except ValueError as e:
        raise ToolCallError(f"Could not extract JSON: {str(e)}")

    repaired = repair_json(extracted)

    try:
        data = json.loads(repaired)
    except json.JSONDecodeError as e:
        raise ToolCallError(f"Invalid JSON syntax: {str(e)}")

    if not isinstance(data, dict):
        raise ToolCallError("Tool call must be a JSON object")

    # Support both standard formats without using falsy 'or'
    name = data.get("name")
    if name is None:
        name = data.get("tool")

    arguments = data.get("arguments")
    if arguments is None:
        arguments = data.get("args")

    if name is None:
        raise ToolCallError("Missing tool name (expected 'name' or 'tool')")
    if not isinstance(name, str):
        raise ToolCallError("Tool name must be a string")

    if arguments is None:
        arguments = {}
    elif not isinstance(arguments, dict):
        raise ToolCallError("Arguments must be a JSON object (dict)")

    if allowed_tools is not None and name not in allowed_tools:
        raise ToolCallError(f"Tool '{name}' is not allowed")

    return name, arguments


def parse_react(text: str) -> Dict[str, Optional[str]]:
    headers = r"(?:Thought|Action|Action Input|Final Answer):"
    
    thought_match = re.search(rf"Thought:\s*(.*?)(?=\n\s*{headers}|$)", text, re.DOTALL | re.IGNORECASE)
    action_match = re.search(rf"Action:\s*(.*?)(?=\n\s*{headers}|$)", text, re.DOTALL | re.IGNORECASE)
    action_input_match = re.search(rf"Action Input:\s*(.*?)(?=\n\s*{headers}|$)", text, re.DOTALL | re.IGNORECASE)
    final_answer_match = re.search(rf"Final Answer:\s*(.*?)(?=\n\s*{headers}|$)", text, re.DOTALL | re.IGNORECASE)

    return {
        "thought": thought_match.group(1).strip() if thought_match else None,
        "action": action_match.group(1).strip() if action_match else None,
        "action_input": action_input_match.group(1).strip() if action_input_match else None,
        "final_answer": final_answer_match.group(1).strip() if final_answer_match else None,
    }
