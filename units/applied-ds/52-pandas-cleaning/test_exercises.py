"""Tests for 52 — Loading and cleaning data with pandas
"""

import importlib
import os
import unittest
import pandas as pd
import numpy as np

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestLoadCsv(unittest.TestCase):
    def test_basic_load(self):
        csv_data = "id,name\n1,Alice\n2,Bob"
        df = m.load_csv(csv_data)
        self.assertEqual(len(df), 2)
        self.assertListEqual(list(df.columns), ["id", "name"])

    def test_explicit_dtypes(self):
        # Zip code must preserve leading zero
        csv_data = "id,zip\n1,02108\n2,90210"
        df = m.load_csv(csv_data, dtypes={"zip": str})
        self.assertEqual(df["zip"].iloc[0], "02108")
        # Text either way: object in pandas 2, the string dtype in pandas 3.
        self.assertTrue(pd.api.types.is_string_dtype(df["zip"]))

    def test_parse_dates(self):
        csv_data = "id,created_at\n1,2023-01-01 10:00:00\n2,2023-01-02 12:00:00"
        df = m.load_csv(csv_data, parse_dates=["created_at"])
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["created_at"]))
        self.assertEqual(df["created_at"].iloc[0].year, 2023)


class TestNormalizeColumns(unittest.TestCase):
    def test_basic_normalization(self):
        df = pd.DataFrame(columns=[" First Name ", "Last-Name!", "EMAIL ADDRESS"])
        cleaned_df = m.normalize_columns(df)
        self.assertListEqual(list(cleaned_df.columns), ["first_name", "last_name", "email_address"])

    def test_special_characters_and_numbers(self):
        df = pd.DataFrame(columns=["User-ID#", "col__2", "  multiple   spaces  "])
        cleaned_df = m.normalize_columns(df)
        self.assertListEqual(list(cleaned_df.columns), ["user_id", "col_2", "multiple_spaces"])

    def test_does_not_mutate_original(self):
        df = pd.DataFrame(columns=["A B"])
        cleaned_df = m.normalize_columns(df)
        self.assertListEqual(list(df.columns), ["A B"])
        self.assertListEqual(list(cleaned_df.columns), ["a_b"])


class TestCleanTextColumn(unittest.TestCase):
    def test_strip_and_lowercase(self):
        df = pd.DataFrame({"city": ["  New York  ", "CHICAGO", "  los angeles "]})
        cleaned_df = m.clean_text_column(df, "city")
        self.assertListEqual(list(cleaned_df["city"]), ["new york", "chicago", "los angeles"])

    def test_collapse_spaces(self):
        df = pd.DataFrame({"text": ["too   many    spaces", "normal space"]})
        cleaned_df = m.clean_text_column(df, "text")
        self.assertEqual(cleaned_df["text"].iloc[0], "too many spaces")

    def test_preserves_nan(self):
        df = pd.DataFrame({"name": [" Alice ", None, np.nan, "Bob"]})
        cleaned_df = m.clean_text_column(df, "name")
        self.assertEqual(cleaned_df["name"].iloc[0], "alice")
        self.assertTrue(pd.isna(cleaned_df["name"].iloc[1]))
        self.assertTrue(pd.isna(cleaned_df["name"].iloc[2]))
        self.assertEqual(cleaned_df["name"].iloc[3], "bob")

    def test_non_existent_column(self):
        df = pd.DataFrame({"a": [1, 2]})
        cleaned_df = m.clean_text_column(df, "b")
        pd.testing.assert_frame_equal(df, cleaned_df)


class TestFillMissing(unittest.TestCase):
    def test_fill_mean(self):
        df = pd.DataFrame({"val": [10.0, np.nan, 30.0]})
        filled = m.fill_missing(df, {"val": "mean"})
        self.assertEqual(filled["val"].iloc[1], 20.0)

    def test_fill_median(self):
        df = pd.DataFrame({"val": [10.0, np.nan, 100.0, 20.0]})
        # Median of 10, 20, 100 is 20
        filled = m.fill_missing(df, {"val": "median"})
        self.assertEqual(filled["val"].iloc[1], 20.0)

    def test_fill_mode(self):
        df = pd.DataFrame({"cat": ["A", "B", np.nan, "A"]})
        filled = m.fill_missing(df, {"cat": "mode"})
        self.assertEqual(filled["cat"].iloc[2], "A")

    def test_fill_constant(self):
        df = pd.DataFrame({"val": [1, np.nan, 3]})
        filled = m.fill_missing(df, {"val": -1})
        self.assertEqual(filled["val"].iloc[1], -1)

    def test_non_existent_column_ignored(self):
        df = pd.DataFrame({"val": [1.0, np.nan]})
        filled = m.fill_missing(df, {"missing_col": "mean"})
        self.assertTrue(pd.isna(filled["val"].iloc[1]))


class TestDropDuplicateRows(unittest.TestCase):
    def test_deduplicate_keeps_latest(self):
        df = pd.DataFrame({
            "user_id": [1, 1, 1],
            "status": ["bronze", "gold", "silver"],
            "updated_at": ["2023-01-01", "2023-01-03", "2023-01-02"]
        })
        deduped = m.drop_duplicate_rows(df, subset=["user_id"], timestamp_col="updated_at")
        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped["status"].iloc[0], "gold")
        self.assertEqual(deduped["updated_at"].iloc[0], pd.to_datetime("2023-01-03"))

    def test_subset_behavior(self):
        df = pd.DataFrame({
            "user_id": [1, 1, 2, 2],
            "device": ["mobile", "mobile", "desktop", "desktop"],
            "val": [10, 20, 30, 40],
            "ts": ["2023-01-01", "2023-01-02", "2023-01-01", "2023-01-02"]
        })
        deduped = m.drop_duplicate_rows(df, subset=["user_id", "device"], timestamp_col="ts")
        self.assertEqual(len(deduped), 2)
        self.assertEqual(deduped[deduped["user_id"] == 1]["val"].iloc[0], 20)
        self.assertEqual(deduped[deduped["user_id"] == 2]["val"].iloc[0], 40)

    def test_resets_index(self):
        df = pd.DataFrame({
            "id": [1, 2, 2],
            "ts": ["2023-01-01", "2023-01-02", "2023-01-03"]
        })
        deduped = m.drop_duplicate_rows(df, subset=["id"], timestamp_col="ts")
        self.assertListEqual(list(deduped.index), [0, 1])


class TestDataQualityReport(unittest.TestCase):
    def test_report_metrics(self):
        df = pd.DataFrame({
            "ints": [1, 2, np.nan, 4],
            "strings": ["A", "B", "A", np.nan]
        })
        report = m.data_quality_report(df)
        
        self.assertEqual(report.index.name, "column")
        self.assertIn("ints", report.index)
        self.assertIn("strings", report.index)
        
        # Check missing count and percentage
        self.assertEqual(report.loc["ints", "missing_count"], 1)
        self.assertEqual(report.loc["ints", "missing_pct"], 0.25)
        self.assertEqual(report.loc["strings", "missing_count"], 1)
        self.assertEqual(report.loc["strings", "missing_pct"], 0.25)
        
        # Check unique counts (excluding NaN)
        self.assertEqual(report.loc["ints", "unique_count"], 3)
        self.assertEqual(report.loc["strings", "unique_count"], 2)

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["a", "b"])
        report = m.data_quality_report(df)
        self.assertEqual(len(report), 2)
        self.assertEqual(report.loc["a", "missing_count"], 0)
        self.assertEqual(report.loc["a", "missing_pct"], 0.0)
        self.assertEqual(report.loc["a", "unique_count"], 0)

    def test_no_columns_dataframe(self):
        df = pd.DataFrame()
        report = m.data_quality_report(df)
        self.assertTrue(report.empty)
        self.assertEqual(report.index.name, "column")
        self.assertListEqual(list(report.columns), ["dtype", "missing_count", "missing_pct", "unique_count"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
