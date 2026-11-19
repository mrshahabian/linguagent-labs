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
        self.name = name
        self.description = description
        self.parameters = parameters
        self.func = func


def validate_args(schema: dict, args: dict) -> None:
    """Validate arguments against a JSON Schema-like dictionary."""
    if not isinstance(args, dict):
        raise ValueError("Arguments must be a dictionary")

    properties = schema.get("properties", {})
    required = schema.get("required", [])

    # Check required fields
    for req in required:
        if req not in args:
            raise ValueError(f"Missing required argument: '{req}'")

    # Check for unknown keys
    for key in args:
        if key not in properties:
            raise ValueError(f"Unknown argument: '{key}'")

    # Validate types and enums
    for key, val in args.items():
        param_schema = properties[key]
        expected_type = param_schema.get("type")

        # Strict type checking (avoiding Python's bool-is-int trap)
        is_valid_type = True
        if expected_type == "boolean":
            is_valid_type = isinstance(val, bool)
        elif expected_type == "integer":
            is_valid_type = isinstance(val, int) and not isinstance(val, bool)
        elif expected_type == "number":
            is_valid_type = isinstance(val, (int, float)) and not isinstance(val, bool)
        elif expected_type == "string":
            is_valid_type = isinstance(val, str)
        elif expected_type == "array":
            is_valid_type = isinstance(val, list)

        if not is_valid_type:
            raise ValueError(
                f"Argument '{key}' must be of type '{expected_type}', got '{type(val).__name__}'"
            )

        # Check enum constraints
        if "enum" in param_schema:
            allowed = param_schema["enum"]
            if val not in allowed:
                raise ValueError(
                    f"Argument '{key}' has invalid value '{val}'. Must be one of: {allowed}"
                )


def coerce_args(schema: dict, args: dict) -> dict:
    """Coerce stringified arguments to their correct types based on the schema."""
    coerced = dict(args)
    properties = schema.get("properties", {})

    for key, val in args.items():
        if key in properties:
            param_schema = properties[key]
            expected_type = param_schema.get("type")

            if isinstance(val, str):
                if expected_type == "integer":
                    try:
                        coerced[key] = int(val)
                    except ValueError:
                        pass
                elif expected_type == "number":
                    try:
                        coerced[key] = float(val)
                    except ValueError:
                        pass
                elif expected_type == "boolean":
                    if val.lower() == "true":
                        coerced[key] = True
                    elif val.lower() == "false":
                        coerced[key] = False

    return coerced


class ToolRegistry:
    """Manages a collection of tools, handles execution, and generates prompts."""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool. Raises ValueError if a tool with the same name exists."""
        if tool.name in self.tools:
            raise ValueError(f"Tool with name '{tool.name}' is already registered")
        self.tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Retrieve a tool by name. Raises KeyError if not found."""
        if name not in self.tools:
            raise KeyError(f"Tool '{name}' not found")
        return self.tools[name]

    def call(self, name: str, args: dict) -> Any:
        """Execute a tool by name with the given raw arguments."""
        tool = self.get(name)
        coerced = coerce_args(tool.parameters, args)
        validate_args(tool.parameters, coerced)
        return tool.func(**coerced)

    def describe(self) -> str:
        """Render a deterministic, human-readable description of all tools."""
        tool_blocks = []
        for name in sorted(self.tools.keys()):
            tool = self.tools[name]
            lines = [
                f"Tool: {tool.name}",
                f"Description: {tool.description}",
                "Parameters:"
            ]
            properties = tool.parameters.get("properties", {})
            required = tool.parameters.get("required", [])

            if not properties:
                lines.append("  None")
            else:
                for param_name in sorted(properties.keys()):
                    param_schema = properties[param_name]
                    ptype = param_schema.get("type", "any")
                    req_status = "required" if param_name in required else "optional"

                    enum_suffix = ""
                    if "enum" in param_schema:
                        enum_vals = ", ".join(str(v) for v in param_schema["enum"])
                        enum_suffix = f" [Enum: {enum_vals}]"

                    lines.append(f"  - {param_name} ({ptype}) ({req_status}){enum_suffix}")

            tool_blocks.append("\n".join(lines))

        return "\n\n".join(tool_blocks)
