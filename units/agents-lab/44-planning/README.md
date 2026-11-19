# 44 — Planning: tasks, dependencies and order

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/agents-lab/44-planning/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/agents-lab/planning?utm_source=github&utm_medium=labs)

*~45 minutes · 18 tests · Intermediate*

In simple LLM applications, an agent executes a single prompt or a linear sequence of steps. However, real-world agentic workflows—like compiling a research report, migrating a codebase, or running a multi-step data analysis—require breaking a complex goal down into multiple distinct tasks. These tasks form a **Directed Acyclic Graph (DAG)**, where some tasks cannot start until others finish.

If your agent cannot orchestrate these tasks correctly, it will run them in the wrong order (e.g., trying to summarize a document before downloading it), get stuck in infinite loops when circular dependencies occur, or waste expensive API credits running tasks whose prerequisites have already failed. Mastering dependency resolution and execution planning is what separates fragile toy agents from robust, production-grade systems.

In this unit, we run completely offline. The "model" is a deterministic fake (a scripted or rule-based object with the same interface a real LLM client has), so the tests pin the agent's engineering, not a model's mood. In a real system, you would plug in an OpenAI or Anthropic client inside the task actions.

## What you'll build

You will build a complete, offline-capable task planning and execution engine. This engine will validate task graphs, detect cycles, group tasks for parallel execution, and run them with automatic retries and cascading failure skips.

| Function / class | You'll implement | Shows up in real code as |
|---|---|---|
| `Task` | A dataclass representing a unit of work with dependencies and retry limits | The definition of an agent's step or tool execution plan |
| `CycleError` | A custom exception raised when circular dependencies are found | A safety guardrail preventing infinite loops in LLM-generated plans |
| `validate_plan` | A validator that ensures all declared dependencies exist | A pre-flight sanity check before executing any LLM-generated plan |
| `topological_order` | An algorithm to linearize tasks while respecting dependencies | The scheduler that determines the exact order of sequential execution |
| `parallel_batches` | An algorithm to group independent tasks into concurrent waves | The optimizer that runs independent LLM calls or API requests in parallel |
| `execute_plan` | An execution engine with retries, dependency injection, and cascading skips | The core agent loop executing a multi-step plan safely |

---

## 1. The Task Dataclass and Plan Validation

To represent a plan, we need a structured way to define a single unit of work. A `Task` contains an identifier, a list of other task identifiers it depends on, a callable action to execute, and a retry limit.

```python
@dataclass
class Task:
    id: str
    depends_on: list[str]
    action: Callable[[dict[str, Any]], Any]
    retries: int = 0
```

The `action` is a callable that takes a dictionary containing the results of its direct dependencies and returns any value. This is called **dependency injection**—a task shouldn't have to search for its inputs; the execution engine should deliver them.

Additionally, we need a custom exception, `CycleError`, which inherits from `ValueError`. If an LLM generates a plan where Task A depends on Task B, and Task B depends on Task A, we must raise this error and abort execution immediately rather than entering an infinite loop.

Before doing any complex graph operations, we must perform a basic sanity check: **validation**. LLMs often hallucinate dependency names. If Task A depends on Task B, but Task B is not present in the plan, the plan is invalid and we must raise a `ValueError` immediately.

**In `exercises.py`:** Implement the `Task` dataclass, the `CycleError` exception class, and the `validate_plan` function.

**Test it:** `python3 -m unittest test_exercises.TestTaskAndValidation -v`

---

## 2. Topological Sorting and Cycle Detection

Before running any tasks, we must find a valid linear order to execute them. If Task B depends on Task A, then Task A must run before Task B. This process is called **topological sorting**.

We can implement topological sorting using a Depth-First Search (DFS) algorithm with three-state coloring to detect cycles:
- **Unvisited (0)**: The node has not been touched yet.
- **Visiting (1)**: The node is in the current recursion stack. If we encounter a node in this state again, we have found a cycle!
- **Visited (2)**: The node and all its dependencies have been fully processed.

```python
# Conceptual DFS coloring structure (not the exact solution)
def dfs(node):
    state[node] = VISITING
    for neighbor in graph[node]:
        if state[neighbor] == VISITING:
            raise CycleError("Cycle detected!")
        elif state[neighbor] == UNVISITED:
            dfs(neighbor)
    state[node] = VISITED
    order.append(node)
```

**The Pitfall:** A common mistake is simply tracking a set of visited nodes. If you only track whether a node has been visited *at some point*, you will falsely report cycles in diamond-shaped dependency graphs (where A depends on B and C, and both B and C depend on D). You must specifically track nodes in the *current recursion path* to detect actual cycles.

Your `topological_order` function must also validate that all declared dependencies actually exist in the task list by calling `validate_plan`. If a cycle is detected, raise `CycleError` with a message showing the cycle path (e.g., `"Cycle detected: A -> B -> A"`).

**In `exercises.py`:** Implement `topological_order(tasks: list[Task]) -> list[str]` which returns a list of task IDs in a valid execution order.

**Test it:** `python3 -m unittest test_exercises.TestTopologicalOrder -v`

---

## 3. Parallel Execution Batches

Running tasks sequentially is safe but slow. If an agent has to fetch information from five different websites, and none of those tasks depend on each other, we should run them in parallel.

To do this, we can group tasks into **parallel batches**. 
- **Batch 0** contains all tasks that have zero dependencies.
- **Batch 1** contains all tasks whose dependencies are entirely satisfied by Batch 0.
- **Batch N** contains all tasks whose dependencies are satisfied by Batches 0 through N-1.

```python
# Example of batching:
# Tasks: A (no deps), B (no deps), C (depends on A), D (depends on B, C)
# Batch 0: ["A", "B"]
# Batch 1: ["C"]
# Batch 2: ["D"]
```

To ensure deterministic behavior in our agent, each batch should be sorted alphabetically by task ID.

**In `exercises.py`:** Implement `parallel_batches(tasks: list[Task]) -> list[list[str]]`. It should first validate the graph (raising `CycleError` if a cycle exists) and then return the list of batches.

**Test it:** `python3 -m unittest test_exercises.TestParallelBatches -v`

---

## 4. The Execution Loop with Retries and Cascading Skips

Now we build the execution engine. The engine must execute tasks in a valid topological order. For each task, it must:
1. Check if any of its direct dependencies failed or were skipped. If so, this task cannot run and must be marked as `"skipped"`.
2. Collect the results of its direct dependencies into a dictionary: `{dep_id: result}`.
3. Call the task's `action` with that dictionary.
4. If the action raises an exception, retry it up to `task.retries` times. 
5. If it still fails after all retries, mark it as `"failed"`.

```python
# Conceptual retry loop
attempts = 0
while attempts <= task.retries:
    try:
        result = task.action(dep_results)
        break
    except Exception as e:
        attempts += 1
        if attempts > task.retries:
            # Handle final failure
```

**Why this matters:** In real agent systems, network requests fail, LLM APIs rate-limit you, and tools timeout. If a critical task fails (e.g., "Download PDF"), any downstream task depending on it (e.g., "Extract PDF text") is guaranteed to fail or crash if run. Skipping them immediately saves time and money.

**In `exercises.py`:** Implement `execute_plan(tasks: list[Task]) -> tuple[dict[str, Any], dict[str, str]]`. It returns a tuple of two dictionaries:
- `results`: Maps successful task IDs to their return values.
- `statuses`: Maps *every* task ID to its final status: `"completed"`, `"failed"`, or `"skipped"`.

**Test it:** `python3 -m unittest test_exercises.TestExecutePlan -v`

---

## The exercises

| Function / Class | What it teaches |
|---|---|
| `Task` | Dataclass representation of structured execution steps |
| `CycleError` | Custom exceptions for graph validation |
| `validate_plan` | Pre-flight check for missing dependencies |
| `topological_order` | DFS graph traversal, cycle detection, and linear scheduling |
| `parallel_batches` | Layered dependency grouping for concurrent execution |
| `execute_plan` | Robust execution loops with retries, dependency injection, and cascading failure propagation |

## Check your understanding

1. **Why does `topological_order` need three-state coloring instead of a simple `visited` set?**
   <details><summary>Answer</summary>A simple <code>visited</code> set cannot distinguish between visiting a node via a different valid path (like a diamond dependency graph) and visiting a node that is currently active in the recursion stack (which indicates a circular dependency cycle).</details>

2. **What is dependency injection in the context of task execution?**
   <details><summary>Answer</summary>It is the practice of passing a task only the specific inputs it needs (the results of its direct dependencies) rather than giving it access to the entire global execution state. This makes tasks modular, easier to test, and less prone to side effects.</details>

3. **If Task C depends on Task B, and Task B depends on Task A, and Task A fails, what should happen to Task C?**
   <details><summary>Answer</summary>Task B must be marked as <code>"skipped"</code> because its dependency (Task A) failed. Consequently, Task C must also be marked as <code>"skipped"</code> because its dependency (Task B) was skipped. This is a cascading skip.</details>

4. **Why do we sort the task IDs alphabetically within each parallel batch?**
   <details><summary>Answer</summary>Sorting ensures deterministic execution. Without sorting, the order of tasks in a batch could depend on the arbitrary order of dictionary keys or set iteration, making debugging and testing difficult.</details>

5. **What happens if a task has `retries = 2`? How many total times can its action be called?**
   <details><summary>Answer</summary>It can be called up to 3 times: 1 initial attempt plus up to 2 retries if the previous attempts raised exceptions.</details>

## Interview questions this module answers

- *How do you detect a cycle in a directed graph?*
- *What is topological sorting, and what are its real-world applications?*
- *How would you design a fault-tolerant workflow engine that supports task retries and handles cascading failures?*
- *How can you group a set of dependent tasks so that independent tasks can run concurrently?*
