"""44 — Planning: tasks, dependencies and order

Implement the Task dataclass, CycleError exception, and the planning/execution
functions so that the tests pass.
"""

from dataclasses import dataclass
from typing import Any, Callable


class CycleError(ValueError):
    """Raised when a circular dependency is detected in the task graph."""
    pass


@dataclass
class Task:
    """Represents a single unit of work in an execution plan.

    Attributes:
        id: A unique string identifier for the task.
        depends_on: A list of task IDs that must complete successfully before this task runs.
        action: A callable that takes a dict of {dependency_id: dependency_result} and returns any value.
        retries: The number of times to retry the action if it raises an exception (default 0).
    """
    id: str
    depends_on: list[str]
    action: Callable[[dict[str, Any]], Any]
    retries: int = 0


def validate_plan(tasks: list[Task]) -> None:
    """Validate that all declared dependencies actually exist in the task list.

    If any task depends on a task ID that is not present in the input list of tasks,
    raise a ValueError.
    """
    raise NotImplementedError("your code here")


def topological_order(tasks: list[Task]) -> list[str]:
    """Return a list of task IDs sorted in a valid topological execution order.

    If a task depends on a task ID that is not present in the input list of tasks,
    raise a ValueError (by calling validate_plan).

    If a circular dependency is detected, raise a CycleError containing the path
    of the cycle (e.g., "Cycle detected: A -> B -> A").
    """
    raise NotImplementedError("your code here")


def parallel_batches(tasks: list[Task]) -> list[list[str]]:
    """Group task IDs into batches that can be executed in parallel.

    Batch 0 contains tasks with no dependencies.
    Batch N contains tasks whose dependencies are fully satisfied by batches 0 to N-1.

    Within each batch, task IDs must be sorted alphabetically.
    If the graph contains a cycle, raise CycleError.
    """
    raise NotImplementedError("your code here")


def execute_plan(tasks: list[Task]) -> tuple[dict[str, Any], dict[str, str]]:
    """Execute the tasks in a valid topological order.

    For each task:
    1. If any of its dependencies failed or were skipped, mark this task as "skipped".
    2. Otherwise, inject the results of its direct dependencies as a dict of {dep_id: result}
       into its action callable.
    3. Execute the action. If it raises an exception, retry up to `task.retries` times.
    4. If it succeeds, record its result in the results dict and mark its status as "completed".
    5. If it fails all retries, mark its status as "failed".

    Returns:
        A tuple of (results, statuses) where:
        - results: A dict mapping successful task IDs to their return values.
        - statuses: A dict mapping ALL task IDs to their final status ("completed", "failed", or "skipped").
    """
    raise NotImplementedError("your code here")
