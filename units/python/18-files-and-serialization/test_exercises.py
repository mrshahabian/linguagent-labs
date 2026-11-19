"""Tests for 18 - files, paths and serialization."""

import importlib
import os
import tempfile
import unittest
from pathlib import Path

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TempDirCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


class TestReadLines(TempDirCase):
    def test_splits_into_lines_without_newlines(self):
        f = self.dir / "a.txt"
        f.write_text("line1\nline2\nline3\n")
        self.assertEqual(m.read_lines(f), ["line1", "line2", "line3"])

    def test_empty_file(self):
        f = self.dir / "empty.txt"
        f.write_text("")
        self.assertEqual(m.read_lines(f), [])


class TestListPythonFiles(TempDirCase):
    def test_finds_files_recursively(self):
        (self.dir / "pkg").mkdir()
        (self.dir / "pkg" / "x.py").write_text("")
        (self.dir / "y.py").write_text("")
        (self.dir / "z.txt").write_text("")
        result = m.list_python_files(self.dir)
        names = sorted(Path(p).name for p in result)
        self.assertEqual(names, ["x.py", "y.py"])

    def test_result_is_sorted(self):
        (self.dir / "b.py").write_text("")
        (self.dir / "a.py").write_text("")
        result = m.list_python_files(self.dir)
        self.assertEqual(result, sorted(result))

    def test_no_python_files(self):
        (self.dir / "readme.md").write_text("")
        self.assertEqual(m.list_python_files(self.dir), [])


class TestEnsureDir(TempDirCase):
    def test_creates_nested_directories(self):
        nested = self.dir / "a" / "b" / "c"
        result = m.ensure_dir(nested)
        self.assertTrue(nested.is_dir())
        self.assertEqual(Path(result), nested)

    def test_idempotent_on_existing_dir(self):
        nested = self.dir / "a" / "b"
        m.ensure_dir(nested)
        try:
            m.ensure_dir(nested)
        except FileExistsError:
            self.fail("ensure_dir must not raise when the directory already exists")


class TestJsonRoundTrip(TempDirCase):
    def test_save_then_load(self):
        path = self.dir / "data.json"
        data = {"a": 1, "b": [1, 2, 3], "c": {"nested": True}}
        m.save_json(data, path)
        self.assertEqual(m.load_json(path), data)

    def test_save_writes_readable_json(self):
        path = self.dir / "data.json"
        m.save_json({"x": 1}, path)
        self.assertIn('"x"', path.read_text())


class TestMergeConfigs(unittest.TestCase):
    def test_override_wins_on_leaf_keys(self):
        result = m.merge_configs({"a": 1}, {"a": 2})
        self.assertEqual(result, {"a": 2})

    def test_nested_dicts_merge_instead_of_replace(self):
        base = {"nested": {"x": 1, "y": 2}}
        override = {"nested": {"y": 99, "z": 3}}
        result = m.merge_configs(base, override)
        self.assertEqual(result, {"nested": {"x": 1, "y": 99, "z": 3}})

    def test_new_top_level_keys_are_added(self):
        result = m.merge_configs({"a": 1}, {"b": 2})
        self.assertEqual(result, {"a": 1, "b": 2})

    def test_does_not_mutate_inputs(self):
        base = {"nested": {"x": 1}}
        override = {"nested": {"y": 2}}
        m.merge_configs(base, override)
        self.assertEqual(base, {"nested": {"x": 1}})
        self.assertEqual(override, {"nested": {"y": 2}})


class TestSetEncoder(unittest.TestCase):
    def test_serialises_sets_as_sorted_lists(self):
        result = m.dumps_with_sets({"tags": {"b", "a", "c"}})
        self.assertEqual(result, '{"tags": ["a", "b", "c"]}')

    def test_still_handles_plain_json_types(self):
        result = m.dumps_with_sets({"a": 1, "b": "x", "c": [1, 2]})
        self.assertEqual(result, '{"a": 1, "b": "x", "c": [1, 2]}')


class TestAtomicWriter(TempDirCase):
    def test_writes_file_on_success(self):
        target = self.dir / "out.txt"
        with m.AtomicWriter(target) as f:
            f.write("hello")
        self.assertEqual(target.read_text(), "hello")

    def test_no_leftover_tmp_file_on_success(self):
        target = self.dir / "out.txt"
        with m.AtomicWriter(target) as f:
            f.write("hello")
        self.assertFalse((self.dir / "out.txt.tmp").exists())

    def test_target_untouched_on_exception(self):
        target = self.dir / "out.txt"
        target.write_text("original")
        with self.assertRaises(ValueError):
            with m.AtomicWriter(target) as f:
                f.write("should not appear")
                raise ValueError("boom")
        self.assertEqual(target.read_text(), "original")

    def test_no_leftover_tmp_file_on_exception(self):
        target = self.dir / "out.txt"
        with self.assertRaises(ValueError):
            with m.AtomicWriter(target) as f:
                f.write("partial")
                raise ValueError("boom")
        self.assertFalse((self.dir / "out.txt.tmp").exists())


class TestTemporaryCwd(TempDirCase):
    def test_changes_and_restores_cwd(self):
        original = Path.cwd()
        with m.temporary_cwd(self.dir):
            self.assertEqual(Path.cwd().resolve(), self.dir.resolve())
        self.assertEqual(Path.cwd(), original)

    def test_restores_cwd_even_on_exception(self):
        original = Path.cwd()
        with self.assertRaises(ValueError):
            with m.temporary_cwd(self.dir):
                raise ValueError("boom")
        self.assertEqual(Path.cwd(), original)


class TestChunkedLines(TempDirCase):
    def test_splits_into_chunks_of_given_size(self):
        path = self.dir / "big.txt"
        path.write_text("\n".join(str(i) for i in range(7)) + "\n")
        chunks = list(m.chunked_lines(path, 3))
        self.assertEqual(chunks, [["0", "1", "2"], ["3", "4", "5"], ["6"]])

    def test_exact_multiple_of_chunk_size(self):
        path = self.dir / "exact.txt"
        path.write_text("\n".join(str(i) for i in range(6)) + "\n")
        chunks = list(m.chunked_lines(path, 2))
        self.assertEqual(chunks, [["0", "1"], ["2", "3"], ["4", "5"]])

    def test_empty_file_yields_nothing(self):
        path = self.dir / "empty.txt"
        path.write_text("")
        self.assertEqual(list(m.chunked_lines(path, 5)), [])

    def test_is_a_generator_not_a_list(self):
        path = self.dir / "big.txt"
        path.write_text("a\nb\n")
        result = m.chunked_lines(path, 1)
        self.assertTrue(hasattr(result, "__next__"))


if __name__ == "__main__":
    unittest.main()
