"""Tests for 40 — Tools and their schemas."""

import importlib
import os
import unittest

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestToolClass(unittest.TestCase):
    def test_instantiation(self):
        def dummy_func(x):
            return x

        tool = m.Tool(
            name="dummy",
            description="A dummy tool",
            parameters={"type": "object", "properties": {}},
            func=dummy_func,
        )
        self.assertEqual(tool.name, "dummy")
        self.assertEqual(tool.description, "A dummy tool")
        self.assertEqual(tool.parameters["type"], "object")
        self.assertEqual(tool.func(42), 42)


class TestValidateArgs(unittest.TestCase):
    def setUp(self):
        self.schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "score": {"type": "number"},
                "active": {"type": "boolean"},
                "tags": {"type": "array"},
                "role": {"type": "string", "enum": ["admin", "user"]},
            },
            "required": ["name", "age"],
        }

    def test_valid_arguments(self):
        args = {
            "name": "Alice",
            "age": 30,
            "score": 95.5,
            "active": True,
            "tags": ["staff", "editor"],
            "role": "admin",
        }
        # Should pass without raising an exception
        m.validate_args(self.schema, args)

    def test_missing_required_field(self):
        args = {"name": "Alice"}  # Missing 'age'
        with self.assertRaises(ValueError) as ctx:
            m.validate_args(self.schema, args)
        self.assertIn("Missing required argument", str(ctx.exception))

    def test_unknown_argument(self):
        args = {"name": "Alice", "age": 30, "extra_field": "not_allowed"}
        with self.assertRaises(ValueError) as ctx:
            m.validate_args(self.schema, args)
        self.assertIn("Unknown argument", str(ctx.exception))

    def test_type_mismatch_boolean_instead_of_integer(self):
        # Python's bool is a subclass of int, so we must strictly catch this
        args = {"name": "Alice", "age": True}
        with self.assertRaises(ValueError) as ctx:
            m.validate_args(self.schema, args)
        self.assertIn("must be of type 'integer'", str(ctx.exception))

    def test_type_mismatch_string_instead_of_number(self):
        args = {"name": "Alice", "age": 30, "score": "95.5"}
        with self.assertRaises(ValueError) as ctx:
            m.validate_args(self.schema, args)
        self.assertIn("must be of type 'number'", str(ctx.exception))

    def test_enum_constraint_violation(self):
        args = {"name": "Alice", "age": 30, "role": "guest"}
        with self.assertRaises(ValueError) as ctx:
            m.validate_args(self.schema, args)
        self.assertIn("Must be one of", str(ctx.exception))

    def test_non_dict_arguments(self):
        with self.assertRaises(ValueError):
            m.validate_args(self.schema, "not a dict")


class TestCoerceArgs(unittest.TestCase):
    def setUp(self):
        self.schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
                "ratio": {"type": "number"},
                "flag": {"type": "boolean"},
                "name": {"type": "string"},
            },
        }

    def test_successful_coercion(self):
        args = {
            "count": "42",
            "ratio": "3.14",
            "flag": "true",
            "name": "Alice",
        }
        coerced = m.coerce_args(self.schema, args)
        self.assertEqual(coerced["count"], 42)
        self.assertEqual(coerced["ratio"], 3.14)
        self.assertEqual(coerced["flag"], True)
        self.assertEqual(coerced["name"], "Alice")

    def test_case_insensitive_boolean_coercion(self):
        args = {"flag": "FALSE"}
        coerced = m.coerce_args(self.schema, args)
        self.assertEqual(coerced["flag"], False)

    def test_failed_coercion_leaves_value_intact(self):
        args = {"count": "not_an_int", "ratio": "not_a_float"}
        coerced = m.coerce_args(self.schema, args)
        self.assertEqual(coerced["count"], "not_an_int")
        self.assertEqual(coerced["ratio"], "not_a_float")

    def test_does_not_mutate_original_dict(self):
        args = {"count": "42"}
        coerced = m.coerce_args(self.schema, args)
        self.assertEqual(args["count"], "42")
        self.assertEqual(coerced["count"], 42)


class TestToolRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = m.ToolRegistry()

        def add(a, b):
            return a + b

        self.add_tool = m.Tool(
            name="add",
            description="Adds two numbers",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
            func=add,
        )

    def test_register_and_get(self):
        self.registry.register(self.add_tool)
        self.assertEqual(self.registry.get("add"), self.add_tool)

    def test_register_duplicate_raises_error(self):
        self.registry.register(self.add_tool)
        with self.assertRaises(ValueError):
            self.registry.register(self.add_tool)

    def test_get_missing_raises_error(self):
        with self.assertRaises(KeyError):
            self.registry.get("nonexistent")

    def test_call_with_coercion_and_validation(self):
        self.registry.register(self.add_tool)
        # Arguments passed as strings should be coerced and executed successfully
        result = self.registry.call("add", {"a": "10.5", "b": "2.5"})
        self.assertEqual(result, 13.0)

    def test_call_validation_failure(self):
        self.registry.register(self.add_tool)
        with self.assertRaises(ValueError):
            # Missing required argument 'b'
            self.registry.call("add", {"a": 10})

    def test_describe_single_tool(self):
        self.registry.register(self.add_tool)
        expected = (
            "Tool: add\n"
            "Description: Adds two numbers\n"
            "Parameters:\n"
            "  - a (number) (required)\n"
            "  - b (number) (required)"
        )
        self.assertEqual(self.registry.describe(), expected)

    def test_describe_multiple_tools_sorted(self):
        def no_param_func():
            return "ok"

        no_param_tool = m.Tool(
            name="check_status",
            description="Checks system status",
            parameters={"type": "object", "properties": {}},
            func=no_param_func,
        )

        self.registry.register(self.add_tool)
        self.registry.register(no_param_tool)

        # 'add' should come before 'check_status' alphabetically
        expected = (
            "Tool: add\n"
            "Description: Adds two numbers\n"
            "Parameters:\n"
            "  - a (number) (required)\n"
            "  - b (number) (required)\n\n"
            "Tool: check_status\n"
            "Description: Checks system status\n"
            "Parameters:\n"
            "  None"
        )
        self.assertEqual(self.registry.describe(), expected)

    def test_describe_with_enums_and_optional_fields(self):
        def search(query, limit=10, category="all"):
            return []

        search_tool = m.Tool(
            name="search",
            description="Search items",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                    "category": {"type": "string", "enum": ["all", "books", "movies"]},
                },
                "required": ["query"],
            },
            func=search,
        )
        self.registry.register(search_tool)
        expected = (
            "Tool: search\n"
            "Description: Search items\n"
            "Parameters:\n"
            "  - category (string) (optional) [Enum: all, books, movies]\n"
            "  - limit (integer) (optional)\n"
            "  - query (string) (required)"
        )
        self.assertEqual(self.registry.describe(), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
