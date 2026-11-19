"""Tests for 42 — Parsing and repairing tool calls"""

import importlib
import os
import unittest

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestToolCallError(unittest.TestCase):
    def test_exception_inheritance(self):
        """Verify ToolCallError inherits from Exception."""
        self.assertTrue(issubclass(m.ToolCallError, Exception))

    def test_exception_message(self):
        """Verify ToolCallError accepts and stores a message."""
        err = m.ToolCallError("Something went wrong")
        self.assertEqual(str(err), "Something went wrong")


class TestExtractJson(unittest.TestCase):
    def test_clean_json(self):
        """Verify extraction of a perfectly clean JSON string."""
        text = '{"name": "test"}'
        self.assertEqual(m.extract_json(text), '{"name": "test"}')

    def test_markdown_json_fence(self):
        """Verify extraction from a ```json block."""
        text = "Here is the call:\n```json\n{\"name\": \"test\"}\n```\nHope this helps!"
        self.assertEqual(m.extract_json(text), '{"name": "test"}')

    def test_markdown_plain_fence(self):
        """Verify extraction from a plain ``` block."""
        text = "```\n{\"name\": \"test\"}\n```"
        self.assertEqual(m.extract_json(text), '{"name": "test"}')

    def test_conversational_fallback(self):
        """Verify fallback to brace matching when no fences exist."""
        text = "Sure, I will call {\"name\": \"test\"} right away."
        self.assertEqual(m.extract_json(text), '{"name": "test"}')

    def test_no_json_raises_value_error(self):
        """Verify ValueError is raised when no braces exist."""
        with self.assertRaises(ValueError):
            m.extract_json("There is no JSON here!")


class TestRepairJson(unittest.TestCase):
    def test_python_literals(self):
        """Verify Python literals are converted to JSON equivalents."""
        text = '{"is_active": True, "is_admin": False, "role": None}'
        repaired = m.repair_json(text)
        self.assertIn('"is_active": true', repaired)
        self.assertIn('"is_admin": false', repaired)
        self.assertIn('"role": null', repaired)

    def test_trailing_commas(self):
        """Verify trailing commas are stripped from objects and arrays."""
        text = '{"list": [1, 2,], "dict": {"a": 1,},}'
        repaired = m.repair_json(text)
        self.assertEqual(repaired, '{"list": [1, 2], "dict": {"a": 1}}')

    def test_single_quotes_to_double_quotes(self):
        """Verify single quotes are converted to double quotes for keys and values."""
        text = "{'name': 'get_weather', 'args': {'city': 'Paris'}}"
        repaired = m.repair_json(text)
        self.assertEqual(repaired, '{"name": "get_weather", "args": {"city": "Paris"}}')

    def test_contractions_preserved(self):
        """Verify contractions like don't are not broken during single-quote repair."""
        text = "{'message': 'Please don\'t do that'}"
        repaired = m.repair_json(text)
        self.assertIn('"message"', repaired)
        self.assertIn("don't", repaired.replace("\\", ""))


class TestParseToolCall(unittest.TestCase):
    def test_valid_standard_call(self):
        """Verify parsing of a standard tool call format."""
        text = '{"name": "calculate", "arguments": {"x": 5}}'
        name, args = m.parse_tool_call(text)
        self.assertEqual(name, "calculate")
        self.assertEqual(args, {"x": 5})

    def test_valid_alternative_call(self):
        """Verify parsing of the alternative (tool/args) format."""
        text = '{"tool": "calculate", "args": {"x": 5}}'
        name, args = m.parse_tool_call(text)
        self.assertEqual(name, "calculate")
        self.assertEqual(args, {"x": 5})

    def test_missing_arguments_defaults_to_empty_dict(self):
        """Verify missing arguments key defaults to an empty dictionary."""
        text = '{"name": "reset_system"}'
        name, args = m.parse_tool_call(text)
        self.assertEqual(name, "reset_system")
        self.assertEqual(args, {})

    def test_invalid_json_raises_tool_call_error(self):
        """Verify syntax errors raise ToolCallError."""
        text = '{"name": "calculate", "arguments": {invalid}'
        with self.assertRaises(m.ToolCallError):
            m.parse_tool_call(text)

    def test_missing_name_raises_tool_call_error(self):
        """Verify missing tool name raises ToolCallError."""
        text = '{"arguments": {"x": 5}}'
        with self.assertRaises(m.ToolCallError):
            m.parse_tool_call(text)

    def test_allowed_tools_validation(self):
        """Verify validation against allowed tools list."""
        text = '{"name": "delete_database", "arguments": {}}'
        # Should succeed when allowed
        name, args = m.parse_tool_call(text, allowed_tools=["delete_database", "search"])
        self.assertEqual(name, "delete_database")

        # Should fail when not allowed
        with self.assertRaises(m.ToolCallError):
            m.parse_tool_call(text, allowed_tools=["search"])


class TestParseReact(unittest.TestCase):
    def test_complete_react_block(self):
        """Verify parsing of a complete ReAct block."""
        text = """
        Thought: I need to check the weather.
        Action: get_weather
        Action Input: {"city": "Berlin"}
        """
        parsed = m.parse_react(text)
        self.assertEqual(parsed["thought"], "I need to check the weather.")
        self.assertEqual(parsed["action"], "get_weather")
        self.assertEqual(parsed["action_input"], '{"city": "Berlin"}')
        self.assertIsNone(parsed["final_answer"])

    def test_final_answer_only(self):
        """Verify parsing when only Thought and Final Answer are present."""
        text = """
        Thought: I have found the answer.
        Final Answer: The capital of France is Paris.
        """
        parsed = m.parse_react(text)
        self.assertEqual(parsed["thought"], "I have found the answer.")
        self.assertIsNone(parsed["action"])
        self.assertIsNone(parsed["action_input"])
        self.assertEqual(parsed["final_answer"], "The capital of France is Paris.")

    def test_case_insensitivity(self):
        """Verify headers are matched case-insensitively."""
        text = """
        thought: thinking...
        action: search
        """
        parsed = m.parse_react(text)
        self.assertEqual(parsed["thought"], "thinking...")
        self.assertEqual(parsed["action"], "search")


if __name__ == "__main__":
    unittest.main(verbosity=2)
