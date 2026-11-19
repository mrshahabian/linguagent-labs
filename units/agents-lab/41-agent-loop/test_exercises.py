"""Tests for 41 — The agent loop."""

import importlib
import os
import unittest

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestMessageHelpers(unittest.TestCase):
    def test_system_msg(self):
        msg = m.system_msg("You are a coder.")
        self.assertEqual(msg, {"role": "system", "content": "You are a coder."})

    def test_user_msg(self):
        msg = m.user_msg("Solve this puzzle.")
        self.assertEqual(msg, {"role": "user", "content": "Solve this puzzle."})

    def test_assistant_msg_simple(self):
        msg = m.assistant_msg("I am thinking.")
        self.assertEqual(msg, {"role": "assistant", "content": "I am thinking."})
        self.assertNotIn("tool_calls", msg)

    def test_assistant_msg_with_tools(self):
        tool_calls = [{"id": "1", "name": "calc", "arguments": {"expr": "2+2"}}]
        msg = m.assistant_msg(None, tool_calls=tool_calls)
        self.assertEqual(msg, {
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls
        })

    def test_assistant_msg_none_content(self):
        msg = m.assistant_msg(None)
        self.assertEqual(msg, {"role": "assistant", "content": None})

    def test_tool_msg(self):
        msg = m.tool_msg("call_abc", "4")
        self.assertEqual(msg, {
            "role": "tool",
            "tool_call_id": "call_abc",
            "content": "4"
        })


class TestScriptedModel(unittest.TestCase):
    def test_returns_queued_responses(self):
        r1 = m.AgentResponse(content="First")
        r2 = m.AgentResponse(content="Second")
        model = m.ScriptedModel([r1, r2])
        
        self.assertEqual(model([]).content, "First")
        self.assertEqual(model([]).content, "Second")
        self.assertEqual(model([]).content, "No more responses.")

    def test_records_history_copies(self):
        r = m.AgentResponse(content="Rep")
        model = m.ScriptedModel([r])
        
        active_history = [{"role": "user", "content": "Hello"}]
        model(active_history)
        
        active_history.append({"role": "assistant", "content": "Rep"})
        
        self.assertEqual(len(model.history), 1)
        self.assertEqual(len(model.history[0]), 1)
        self.assertEqual(model.history[0][0]["content"], "Hello")

    def test_no_responses_returns_fallback(self):
        model = m.ScriptedModel([])
        resp = model([])
        self.assertEqual(resp.content, "No more responses.")
        self.assertIsNone(resp.tool_calls)

    def test_deep_copy_preserves_nested_structures(self):
        model = m.ScriptedModel([m.AgentResponse(content="Ok")])
        nested = [{"role": "user", "content": "Hi", "metadata": {"id": 123}}]
        model(nested)
        nested[0]["metadata"]["id"] = 999
        self.assertEqual(model.history[0][0]["metadata"]["id"], 123)


class TestRunAgent(unittest.TestCase):
    def setUp(self):
        self.registry = m.ToolRegistry()
        self.registry.register("add", lambda x, y: x + y)
        
        def fail_tool():
            raise ValueError("Tool failed!")
        self.registry.register("fail", fail_tool)

    def test_immediate_final_answer(self):
        model = m.ScriptedModel([
            m.AgentResponse(content="The answer is 42.")
        ])
        
        result = m.run_agent(model, self.registry, "What is the answer?", max_steps=3)
        
        self.assertEqual(result["status"], "final_answer")
        self.assertEqual(result["steps"], 1)
        self.assertEqual(len(result["history"]), 3)
        self.assertEqual(result["history"][0]["role"], "system")
        self.assertEqual(result["history"][1]["content"], "What is the answer?")
        self.assertEqual(result["history"][2]["content"], "The answer is 42.")

    def test_one_tool_call_loop(self):
        model = m.ScriptedModel([
            m.AgentResponse(content="Let me add.", tool_calls=[
                {"id": "c1", "name": "add", "arguments": {"x": 5, "y": 10}}
            ]),
            m.AgentResponse(content="The sum is 15.")
        ])
        
        result = m.run_agent(model, self.registry, "Add 5 and 10", max_steps=5)
        
        self.assertEqual(result["status"], "final_answer")
        self.assertEqual(result["steps"], 2)
        self.assertEqual(len(result["history"]), 5)
        self.assertEqual(result["history"][3]["role"], "tool")
        self.assertEqual(result["history"][3]["content"], "15")
        self.assertEqual(result["history"][4]["content"], "The sum is 15.")

    def test_tool_error_handling(self):
        model = m.ScriptedModel([
            m.AgentResponse(content="Let me call fail.", tool_calls=[
                {"id": "c2", "name": "fail", "arguments": {}}
            ]),
            m.AgentResponse(content="Ah, it failed. I will stop.")
        ])
        
        result = m.run_agent(model, self.registry, "Run fail", max_steps=5)
        
        self.assertEqual(result["status"], "final_answer")
        self.assertEqual(result["steps"], 2)
        self.assertEqual(result["history"][3]["role"], "tool")
        self.assertIn("Error: Tool failed!", result["history"][3]["content"])

    def test_max_steps_limit(self):
        model = m.ScriptedModel([
            m.AgentResponse(content="Step 1", tool_calls=[{"id": "1", "name": "add", "arguments": {"x": 1, "y": 1}}]),
            m.AgentResponse(content="Step 2", tool_calls=[{"id": "2", "name": "add", "arguments": {"x": 1, "y": 2}}]),
            m.AgentResponse(content="Step 3", tool_calls=[{"id": "3", "name": "add", "arguments": {"x": 1, "y": 3}}]),
        ])
        
        result = m.run_agent(model, self.registry, "Keep counting", max_steps=2)
        
        self.assertEqual(result["status"], "max_steps")
        self.assertEqual(result["steps"], 2)

    def test_repeated_tool_call_detection(self):
        model = m.ScriptedModel([
            m.AgentResponse(content="Call add", tool_calls=[
                {"id": "c1", "name": "add", "arguments": {"x": 2, "y": 2}}
            ]),
            m.AgentResponse(content="Call add again", tool_calls=[
                {"id": "c2", "name": "add", "arguments": {"x": 2, "y": 2}}
            ])
        ])
        
        result = m.run_agent(model, self.registry, "Loop forever", max_steps=5)
        
        self.assertEqual(result["status"], "repeated_tool_call")
        self.assertEqual(result["steps"], 2)
        self.assertEqual(result["history"][-1]["content"], "Call add again")

    def test_multiple_tools_in_one_step(self):
        model = m.ScriptedModel([
            m.AgentResponse(content="Parallel", tool_calls=[
                {"id": "p1", "name": "add", "arguments": {"x": 1, "y": 2}},
                {"id": "p2", "name": "add", "arguments": {"x": 10, "y": 20}}
            ]),
            m.AgentResponse(content="Done.")
        ])
        result = m.run_agent(model, self.registry, "Add both", max_steps=3)
        self.assertEqual(result["status"], "final_answer")
        self.assertEqual(result["steps"], 2)
        # 2 initial (system, user) + 1 assistant + 2 tools + 1 assistant = 6 messages
        self.assertEqual(len(result["history"]), 6)
        self.assertEqual(result["history"][3]["content"], "3")
        self.assertEqual(result["history"][4]["content"], "30")

    def test_tool_not_found_error_handling(self):
        model = m.ScriptedModel([
            m.AgentResponse(content="Call missing", tool_calls=[
                {"id": "m1", "name": "missing_tool", "arguments": {}}
            ]),
            m.AgentResponse(content="Oh well")
        ])
        result = m.run_agent(model, self.registry, "Run missing", max_steps=3)
        self.assertEqual(result["status"], "final_answer")
        self.assertIn("Error: Tool 'missing_tool' not found", result["history"][3]["content"])

    def test_repeated_tool_call_different_args_is_allowed(self):
        model = m.ScriptedModel([
            m.AgentResponse(content="Call 1", tool_calls=[
                {"id": "c1", "name": "add", "arguments": {"x": 1, "y": 1}}
            ]),
            m.AgentResponse(content="Call 2", tool_calls=[
                {"id": "c2", "name": "add", "arguments": {"x": 1, "y": 2}}
            ]),
            m.AgentResponse(content="Done")
        ])
        result = m.run_agent(model, self.registry, "Run both", max_steps=4)
        self.assertEqual(result["status"], "final_answer")
        self.assertEqual(result["steps"], 3)

    def test_non_string_tool_output_coerced_to_string(self):
        self.registry.register("get_list", lambda: [1, 2, 3])
        model = m.ScriptedModel([
            m.AgentResponse(content="Get list", tool_calls=[
                {"id": "l1", "name": "get_list", "arguments": {}}
            ]),
            m.AgentResponse(content="Got it")
        ])
        result = m.run_agent(model, self.registry, "Get list", max_steps=3)
        self.assertEqual(result["history"][3]["content"], "[1, 2, 3]")

    def test_zero_max_steps(self):
        model = m.ScriptedModel([
            m.AgentResponse(content="Should not run")
        ])
        result = m.run_agent(model, self.registry, "Goal", max_steps=0)
        self.assertEqual(result["status"], "max_steps")
        self.assertEqual(result["steps"], 0)
        self.assertEqual(len(result["history"]), 2)  # system, user

    def test_history_initialization_correctness(self):
        model = m.ScriptedModel([m.AgentResponse(content="Done")])
        result = m.run_agent(model, self.registry, "My Goal", max_steps=1)
        history = result["history"]
        self.assertEqual(history[0], {"role": "system", "content": "You are a helpful assistant."})
        self.assertEqual(history[1], {"role": "user", "content": "My Goal"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
