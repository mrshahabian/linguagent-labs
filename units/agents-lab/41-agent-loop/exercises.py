"""41 — The agent loop

Implement the message helpers, the ScriptedModel, and the core run_agent loop.
Read README.md first; every exercise is covered there.
"""

import dataclasses
import json
from typing import Any, Callable, Dict, List, Optional
import copy


@dataclasses.dataclass
class AgentResponse:
    """Represents a response from the LLM, which may contain text and/or tool calls."""
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ToolRegistry:
    """A registry that maps tool names to executable Python functions.
    
    This is provided for you to use in run_agent.
    """
    def __init__(self):
        self._tools: Dict[str, Callable] = {}

    def register(self, name: str, func: Callable) -> None:
        self._tools[name] = func

    def call(self, name: str, arguments: Dict[str, Any]) -> str:
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found")
        # Execute the tool and force the result to a string representation
        return str(self._tools[name](**arguments))


# --- 1. Message Helpers ---

def system_msg(content: str) -> Dict[str, Any]:
    """Return a system message dictionary."""
    raise NotImplementedError("your code here")


def user_msg(content: str) -> Dict[str, Any]:
    """Return a user message dictionary."""
    raise NotImplementedError("your code here")


def assistant_msg(content: Optional[str], tool_calls: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Return an assistant message dictionary.
    
    If tool_calls is provided, it must be included in the dictionary.
    If content is None, it should still be present in the dictionary as None.
    """
    raise NotImplementedError("your code here")


def tool_msg(tool_call_id: str, content: str) -> Dict[str, Any]:
    """Return a tool message dictionary linking back to the tool_call_id."""
    raise NotImplementedError("your code here")


# --- 2. ScriptedModel ---

class ScriptedModel:
    """A mock LLM client that returns queued responses and records call history."""

    def __init__(self, responses: List[AgentResponse]):
        """Initialize with a list of AgentResponse objects to yield in order."""
        raise NotImplementedError("your code here")

    def __call__(self, messages: List[Dict[str, Any]]) -> AgentResponse:
        """Record a copy of the messages sent, and return the next queued response.
        
        If no responses remain, return an AgentResponse with content="No more responses."
        """
        raise NotImplementedError("your code here")


# --- 3. The Agent Loop ---

def run_agent(
    model: ScriptedModel,
    registry: ToolRegistry,
    goal: str,
    max_steps: int = 5
) -> Dict[str, Any]:
    """Run the agent loop to achieve the given goal.
    
    The history must be initialized with:
      1. A system message: "You are a helpful assistant."
      2. A user message containing the goal.
      
    In each step of the loop (up to max_steps):
      1. Call the model with the current history.
      2. Append the model's response as an assistant message to the history.
      3. If there are no tool calls (or tool_calls is empty), stop and return
         with status "final_answer".
      4. If there are tool calls, execute each tool call in the list:
         - Check if this exact tool call (same name and arguments) has already been executed
           in this run. If so, stop immediately and return with status "repeated_tool_call".
         - Call the tool using the registry.
         - If the tool raises an exception, catch it and format the result as "Error: <exception_message>".
         - Append the tool result to the history as a tool message.
         
    If the loop finishes max_steps without returning, return with status "max_steps".
    
    Returns a dictionary:
    {
        "status": "final_answer" | "max_steps" | "repeated_tool_call",
        "history": List[Dict[str, Any]],
        "steps": int  # The number of model calls made
    }
    """
    raise NotImplementedError("your code here")
