"""40 — Tools and their schemas

Implement the Tool class, validation, coercion, and registry logic so
that the agent can safely understand and execute tools.
"""

from typing import Any, Callable, Dict, List


class Tool:
    """Represents an agent tool.

    Attributes:
        name: The unique name of the tool (e.g., "calculate_tax").
        description: A description of what the tool does.
        parameters: A JSON Schema-like dictionary describing the arguments.
        func: The actual Python function to execute.
    """

    def __init__(self, name: str, description: str, parameters: dict, func: Callable):
        """Initialize the Tool with its metadata and executable function."""
        raise NotImplementedError("your code here")


def validate_args(schema: dict, args: dict) -> None:
    """Validate arguments against a JSON Schema-like dictionary.

    The schema is expected to be an object type, e.g.:
    {
        "type": "object",
        "properties": {
            "param1": {"type": "string", "enum": ["a", "b"]},
            "param2": {"type": "integer"}
        },
        "required": ["param1"]
    }

    Raises:
        ValueError: If args is not a dictionary, if required fields are missing,
                    if unknown keys are present, if types do not match,
                    or if enum constraints are violated.
    """
    raise NotImplementedError("your code here")


def coerce_args(schema: dict, args: dict) -> dict:
    """Coerce stringified arguments to their correct types based on the schema.

    Supports converting:
    - Strings to integers (if expected type is "integer")
    - Strings to floats (if expected type is "number")
    - Strings "true"/"false" (case-insensitive) to booleans (if expected type is "boolean")

    If conversion fails, the value should be left as-is so validation can catch it.
    Returns a new dictionary without mutating the original.
    """
    raise NotImplementedError("your code here")


class ToolRegistry:
    """Manages a collection of tools, handles execution, and generates prompts."""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool. Raises ValueError if a tool with the same name exists."""
        raise NotImplementedError("your code here")

    def get(self, name: str) -> Tool:
        """Retrieve a tool by name. Raises KeyError if not found."""
        raise NotImplementedError("your code here")

    def call(self, name: str, args: dict) -> Any:
        """Execute a tool by name with the given raw arguments.

        This method must:
        1. Retrieve the tool.
        2. Coerce the arguments.
        3. Validate the coerced arguments.
        4. Execute the tool's function with the validated arguments and return the result.
        """
        raise NotImplementedError("your code here")

    def describe(self) -> str:
        """Render a deterministic, human-readable description of all tools.

        Format:
        Tool: <name>
        Description: <description>
        Parameters:
          - <param_name> (<type>) (<required/optional>)[ [Enum: <val1>, <val2>]]

        If there are no parameters, print "  None" under "Parameters:".
        Tools must be sorted alphabetically by name.
        Parameters must be sorted alphabetically by parameter name.
        Multiple tools are separated by two newlines (\n\n).
        """
        raise NotImplementedError("your code here")
