"""Reference solutions for 44 — Planning: tasks, dependencies and order"""

from dataclasses import dataclass
from typing import Any, Callable


class CycleError(ValueError):
    """Raised when a circular dependency is detected in the task graph."""
    pass


@dataclass
class Task:
    id: str
    depends_on: list[str]
    action: Callable[[dict[str, Any]], Any]
    retries: int = 0


def validate_plan(tasks: list[Task]) -> None:
    all_ids = {t.id for t in tasks}
    for t in tasks:
        for dep in t.depends_on:
            if dep not in all_ids:
                raise ValueError(f"Task '{t.id}' depends on non-existent task '{dep}'")


def topological_order(tasks: list[Task]) -> list[str]:
    validate_plan(tasks)
    task_map = {t.id: t for t in tasks}
    all_ids = set(task_map.keys())

    # 0 = unvisited, 1 = visiting, 2 = visited
    state = {t_id: 0 for t_id in all_ids}
    order = []
    path = []

    def dfs(node: str):
        state[node] = 1
        path.append(node)

        for dep in task_map[node].depends_on:
            if state[dep] == 1:
                # Reconstruct cycle path
                cycle_start = path.index(dep)
                cycle_path = path[cycle_start:] + [dep]
                cycle_str = " -> ".join(cycle_path)
                raise CycleError(f"Cycle detected: {cycle_str}")
            elif state[dep] == 0:
                dfs(dep)

        state[node] = 2
        path.pop()
        order.append(node)

    for t_id in sorted(all_ids):  # Sort for deterministic traversal start
        if state[t_id] == 0:
            dfs(t_id)

    return order


def parallel_batches(tasks: list[Task]) -> list[list[str]]:
    # Validate graph and detect cycles first
    topological_order(tasks)

    task_map = {t.id: t for t in tasks}
    completed = set()
    batches = []
    remaining = set(task_map.keys())

    while remaining:
        batch = []
        for t_id in sorted(remaining):
            t = task_map[t_id]
            if all(dep in completed for dep in t.depends_on):
                batch.append(t_id)
        
        if not batch:
            raise CycleError("Cycle detected during batching")
            
        batches.append(sorted(batch))
        completed.update(batch)
        remaining.difference_update(batch)

    return batches


def execute_plan(tasks: list[Task]) -> tuple[dict[str, Any], dict[str, str]]:
    # Ensure we have a valid topological order to execute sequentially
    order = topological_order(tasks)
    task_map = {t.id: t for t in tasks}

    results = {}
    statuses = {}

    for t_id in order:
        t = task_map[t_id]
        
        # Check if any dependency failed or was skipped
        dep_failed_or_skipped = False
        for dep in t.depends_on:
            if statuses.get(dep) in ("failed", "skipped"):
                dep_failed_or_skipped = True
                break

        if dep_failed_or_skipped:
            statuses[t_id] = "skipped"
            continue

        # Collect results of direct dependencies
        dep_results = {dep: results[dep] for dep in t.depends_on}

        # Execute with retries
        attempts = 0
        max_attempts = t.retries + 1
        success = False

        while attempts < max_attempts:
            try:
                res = t.action(dep_results)
                results[t_id] = res
                statuses[t_id] = "completed"
                success = True
                break
            except Exception:
                attempts += 1

        if not success:
            statuses[t_id] = "failed"

    return results, statuses
