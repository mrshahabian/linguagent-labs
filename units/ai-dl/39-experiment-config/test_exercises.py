"""Tests for 39 - experiment configuration management."""

import datetime
import importlib
import os
import unittest

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestInstantiate(unittest.TestCase):
    def test_builds_the_target_class(self):
        result = m.instantiate({"_target_": "datetime.date", "year": 2024, "month": 1, "day": 1})
        self.assertIsInstance(result, datetime.date)

    def test_passes_remaining_keys_as_kwargs(self):
        result = m.instantiate({"_target_": "datetime.date", "year": 2024, "month": 1, "day": 1})
        self.assertEqual(result, datetime.date(2024, 1, 1))

    def test_works_with_a_deeper_module_path(self):
        result = m.instantiate({"_target_": "decimal.Decimal", "value": "1.5"})
        self.assertEqual(str(result), "1.5")

    def test_does_not_mutate_the_input_config(self):
        config = {"_target_": "datetime.date", "year": 2024, "month": 1, "day": 1}
        m.instantiate(config)
        self.assertIn("_target_", config)


class TestResolveInterpolations(unittest.TestCase):
    def test_substitutes_a_reference_to_a_sibling_field(self):
        config = {"pose": {"name": "yolo"}, "run_name": "demo_${pose.name}"}
        result = m.resolve_interpolations(config)
        self.assertEqual(result["run_name"], "demo_yolo")

    def test_non_string_values_pass_through(self):
        config = {"pose": {"confidence": 0.5}}
        result = m.resolve_interpolations(config)
        self.assertEqual(result["pose"]["confidence"], 0.5)

    def test_strings_without_interpolation_are_unchanged(self):
        config = {"device": "cpu"}
        result = m.resolve_interpolations(config)
        self.assertEqual(result["device"], "cpu")

    def test_resolves_nested_dicts_recursively(self):
        config = {"a": {"value": "42", "b": {"label": "x_${a.value}"}}}
        result = m.resolve_interpolations(config)
        self.assertEqual(result["a"]["b"]["label"], "x_42")


class TestMergeConfigGroups(unittest.TestCase):
    CONFIG_GROUPS = {
        "pose": {
            "yolo": {"_target_": "x.PoseExtractor", "name": "yolo", "confidence": 0.5},
            "mediapipe": {"_target_": "x.PoseExtractor", "name": "mediapipe", "confidence": 0.8},
        },
        "dataset": {
            "rhm_har": {"path": "/data/rhm_har"},
        },
    }

    def test_selects_the_named_option_per_group(self):
        merged = m.merge_config_groups({"pose": "yolo"}, self.CONFIG_GROUPS)
        self.assertEqual(merged["pose"]["name"], "yolo")

    def test_merges_multiple_groups(self):
        merged = m.merge_config_groups(
            {"pose": "mediapipe", "dataset": "rhm_har"}, self.CONFIG_GROUPS)
        self.assertEqual(merged["pose"]["name"], "mediapipe")
        self.assertEqual(merged["dataset"]["path"], "/data/rhm_har")

    def test_result_is_a_deep_copy_not_shared_state(self):
        merged = m.merge_config_groups({"pose": "yolo"}, self.CONFIG_GROUPS)
        merged["pose"]["confidence"] = 0.99
        self.assertEqual(self.CONFIG_GROUPS["pose"]["yolo"]["confidence"], 0.5)


class TestApplyOverrides(unittest.TestCase):
    CONFIG_GROUPS = {
        "pose": {
            "yolo": {"_target_": "x.PoseExtractor", "name": "yolo", "confidence": 0.5, "device": "cpu"},
            "mediapipe": {"_target_": "x.PoseExtractor", "name": "mediapipe", "confidence": 0.8, "device": "cpu"},
        },
    }

    def setUp(self):
        self.base = m.merge_config_groups({"pose": "yolo"}, self.CONFIG_GROUPS)

    def test_no_overrides_returns_the_defaults(self):
        result = m.apply_overrides(self.base, self.CONFIG_GROUPS, [])
        self.assertEqual(result["pose"]["name"], "yolo")
        self.assertEqual(result["pose"]["confidence"], 0.5)

    def test_group_override_swaps_the_whole_group(self):
        result = m.apply_overrides(self.base, self.CONFIG_GROUPS, ["pose=mediapipe"])
        self.assertEqual(result["pose"]["name"], "mediapipe")
        self.assertEqual(result["pose"]["confidence"], 0.8)

    def test_dotted_override_changes_only_that_key(self):
        result = m.apply_overrides(self.base, self.CONFIG_GROUPS, ["pose.confidence=0.9"])
        self.assertEqual(result["pose"]["name"], "yolo")
        self.assertEqual(result["pose"]["confidence"], 0.9)

    def test_group_swap_then_key_override_apply_in_order(self):
        result = m.apply_overrides(
            self.base, self.CONFIG_GROUPS, ["pose=mediapipe", "pose.device=cuda:0"])
        self.assertEqual(result["pose"]["name"], "mediapipe")
        self.assertEqual(result["pose"]["device"], "cuda:0")

    def test_override_values_are_type_coerced(self):
        result = m.apply_overrides(self.base, self.CONFIG_GROUPS, ["pose.confidence=0.9"])
        self.assertIsInstance(result["pose"]["confidence"], float)

    def test_string_values_that_are_not_numeric_stay_strings(self):
        result = m.apply_overrides(self.base, self.CONFIG_GROUPS, ["pose.device=cuda:0"])
        self.assertIsInstance(result["pose"]["device"], str)

    def test_does_not_mutate_the_input_config(self):
        m.apply_overrides(self.base, self.CONFIG_GROUPS, ["pose=mediapipe"])
        self.assertEqual(self.base["pose"]["name"], "yolo")


if __name__ == "__main__":
    unittest.main()
