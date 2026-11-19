"""Tests for 44 — Planning: tasks, dependencies and order"""

import importlib
import os
import unittest

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestTaskAndValidation(unittest.TestCase):
    def test_task_creation_and_validation(self):
        def dummy_action(deps):
            return "ok"
        task = m.Task(id="A", depends_on=["B"], action=dummy_action, retries=3)
        self.assertEqual(task.id, "A")
        self.assertEqual(task.depends_on, ["B"])
        self.assertEqual(task.action({}), "ok")
        self.assertEqual(task.retries, 3)
        # This will raise NotImplementedError on the stub, ensuring the test fails
        m.validate_plan([task, m.Task("B", [], dummy_action)])

    def test_task_default_retries_and_validation(self):
        task = m.Task(id="A", depends_on=[], action=lambda x: None)
        self.assertEqual(task.retries, 0)
        # This will raise NotImplementedError on the stub, ensuring the test fails
        m.validate_plan([task])

    def test_cycle_error_subclass_and_detection(self):
        self.assertTrue(issubclass(m.CycleError, ValueError))
        # This will raise NotImplementedError on the stub, ensuring the test fails
        tasks = [m.Task("A", ["A"], lambda x: None)]
        with self.assertRaises(m.CycleError):
            m.topological_order(tasks)

    def test_validate_plan_missing_dependency(self):
        tasks = [
            m.Task("A", ["B"], lambda x: None),
        ]
        # This will raise NotImplementedError on the stub, ensuring the test fails
        with self.assertRaises(ValueError) as ctx:
            m.validate_plan(tasks)
        self.assertIn("non-existent task 'B'", str(ctx.exception))


class TestTopologicalOrder(unittest.TestCase):
    def test_no_dependencies(self):
        tasks = [
            m.Task("A", [], lambda x: None),
            m.Task("B", [], lambda x: None),
        ]
        order = m.topological_order(tasks)
        self.assertEqual(sorted(order), ["A", "B"])

    def test_linear_dependencies(self):
        # C depends on B, B depends on A
        tasks = [
            m.Task("C", ["B"], lambda x: None),
            m.Task("A", [], lambda x: None),
            m.Task("B", ["A"], lambda x: None),
        ]
        order = m.topological_order(tasks)
        self.assertEqual(order, ["A", "B", "C"])

    def test_branching_dependencies(self):
        # A has no deps, B depends on A, C depends on A, D depends on B and C
        tasks = [
            m.Task("D", ["B", "C"], lambda x: None),
            m.Task("B", ["A"], lambda x: None),
            m.Task("C", ["A"], lambda x: None),
            m.Task("A", [], lambda x: None),
        ]
        order = m.topological_order(tasks)
        self.assertEqual(order[0], "A")
        self.assertEqual(order[-1], "D")
        self.assertTrue(order.index("B") > order.index("A"))
        self.assertTrue(order.index("C") > order.index("A"))
        self.assertTrue(order.index("D") > order.index("B"))
        self.assertTrue(order.index("D") > order.index("C"))

    def test_simple_cycle_raises_cycle_error(self):
        # A depends on B, B depends on A
        tasks = [
            m.Task("A", ["B"], lambda x: None),
            m.Task("B", ["A"], lambda x: None),
        ]
        with self.assertRaises(m.CycleError) as ctx:
            m.topological_order(tasks)
        self.assertIn("Cycle detected", str(ctx.exception))
        self.assertTrue("A -> B -> A" in str(ctx.exception) or "B -> A -> B" in str(ctx.exception))

    def test_longer_cycle_raises_cycle_error(self):
        # A -> B -> C -> A
        tasks = [
            m.Task("A", ["B"], lambda x: None),
            m.Task("B", ["C"], lambda x: None),
            m.Task("C", ["A"], lambda x: None),
        ]
        with self.assertRaises(m.CycleError):
            m.topological_order(tasks)


class TestParallelBatches(unittest.TestCase):
    def test_all_independent(self):
        tasks = [
            m.Task("C", [], lambda x: None),
            m.Task("A", [], lambda x: None),
            m.Task("B", [], lambda x: None),
        ]
        batches = m.parallel_batches(tasks)
        self.assertEqual(batches, [["A", "B", "C"]])

    def test_layered_dependencies(self):
        # A, B have no deps, C depends on A, D depends on B, C
        tasks = [
            m.Task("D", ["B", "C"], lambda x: None),
            m.Task("C", ["A"], lambda x: None),
            m.Task("A", [], lambda x: None),
            m.Task("B", [], lambda x: None),
        ]
        batches = m.parallel_batches(tasks)
        self.assertEqual(batches, [
            ["A", "B"],
            ["C"],
            ["D"]
        ])

    def test_cycle_in_batching_raises_cycle_error(self):
        tasks = [
            m.Task("A", ["B"], lambda x: None),
            m.Task("B", ["A"], lambda x: None),
        ]
        with self.assertRaises(m.CycleError):
            m.parallel_batches(tasks)


class TestExecutePlan(unittest.TestCase):
    def test_successful_execution_flow(self):
        execution_log = []

        def make_action(name, val):
            def action(deps):
                execution_log.append(name)
                return val
            return action

        tasks = [
            m.Task("A", [], make_action("A", 10)),
            m.Task("B", ["A"], lambda deps: deps["A"] * 2),
        ]

        results, statuses = m.execute_plan(tasks)
        self.assertEqual(results, {"A": 10, "B": 20})
        self.assertEqual(statuses, {"A": "completed", "B": "completed"})
        self.assertEqual(execution_log, ["A"])

    def test_dependency_injection(self):
        tasks = [
            m.Task("fetch_data", [], lambda deps: {"users": [1, 2]}),
            m.Task("process_data", ["fetch_data"], lambda deps: f"Processed {len(deps['fetch_data']['users'])} users"),
        ]
        results, statuses = m.execute_plan(tasks)
        self.assertEqual(results["process_data"], "Processed 2 users")
        self.assertEqual(statuses["process_data"], "completed")

    def test_retry_success(self):
        attempts = 0

        def flaky_action(deps):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ConnectionError("Network timeout")
            return "success_val"

        tasks = [
            m.Task("flaky", [], flaky_action, retries=2)
        ]

        results, statuses = m.execute_plan(tasks)
        self.assertEqual(results, {"flaky": "success_val"})
        self.assertEqual(statuses, {"flaky": "completed"})
        self.assertEqual(attempts, 3)

    def test_retry_exhausted_failure(self):
        attempts = 0

        def failing_action(deps):
            nonlocal attempts
            attempts += 1
            raise ValueError("Permanent error")

        tasks = [
            m.Task("broken", [], failing_action, retries=1)
        ]

        results, statuses = m.execute_plan(tasks)
        self.assertEqual(results, {})
        self.assertEqual(statuses, {"broken": "failed"})
        self.assertEqual(attempts, 2)  # 1 initial + 1 retry

    def test_cascading_skips(self):
        def fail_action(deps):
            raise ValueError("Fail")

        tasks = [
            m.Task("A", [], fail_action),
            m.Task("B", ["A"], lambda deps: "B_val"),
            m.Task("C", ["B"], lambda deps: "C_val"),
            m.Task("D", [], lambda deps: "D_val"),
        ]

        results, statuses = m.execute_plan(tasks)
        self.assertEqual(results, {"D": "D_val"})
        self.assertEqual(statuses, {
            "A": "failed",
            "B": "skipped",
            "C": "skipped",
            "D": "completed"
        })

    def test_cycle_execution_aborts_immediately(self):
        tasks = [
            m.Task("A", ["B"], lambda deps: "A"),
            m.Task("B", ["A"], lambda deps: "B"),
        ]
        with self.assertRaises(m.CycleError):
            m.execute_plan(tasks)


if __name__ == "__main__":
    unittest.main(verbosity=2)
