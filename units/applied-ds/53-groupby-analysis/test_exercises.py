"""Tests for 53 - Exploring data: groupby, pivots and joins."""

import importlib
import os
import unittest
import pandas as pd
import numpy as np

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestSummaryBy(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "category": ["A", "A", "B", "B", "A"],
            "sales": [100, 200, 150, 250, 300],
            "rating": [4.0, 4.5, 3.5, 5.0, 3.0]
        })

    def test_simple_aggregation(self):
        metrics = {
            "total_sales": ("sales", "sum"),
            "avg_rating": ("rating", "mean")
        }
        result = m.summary_by(self.df, "category", metrics)
        
        expected = pd.DataFrame({
            "total_sales": [600, 400],
            "avg_rating": [3.833333, 4.25]
        }, index=pd.Index(["A", "B"], name="category"))
        
        pd.testing.assert_frame_equal(result, expected, rtol=1e-5)

    def test_multiple_metrics_on_same_column(self):
        metrics = {
            "min_sales": ("sales", "min"),
            "max_sales": ("sales", "max")
        }
        result = m.summary_by(self.df, "category", metrics)
        expected = pd.DataFrame({
            "min_sales": [100, 150],
            "max_sales": [300, 250]
        }, index=pd.Index(["A", "B"], name="category"))
        pd.testing.assert_frame_equal(result, expected)

    def test_empty_dataframe(self):
        df_empty = pd.DataFrame(columns=["category", "sales", "rating"])
        metrics = {"total_sales": ("sales", "sum")}
        result = m.summary_by(df_empty, "category", metrics)
        self.assertTrue(result.empty)
        self.assertEqual(list(result.columns), ["total_sales"])

    def test_missing_values_in_group_key(self):
        df_with_nan = pd.DataFrame({
            "category": ["A", None, "B", "B", "A"],
            "sales": [100, 200, 150, 250, 300]
        })
        metrics = {"total_sales": ("sales", "sum")}
        result = m.summary_by(df_with_nan, "category", metrics)
        expected = pd.DataFrame({
            "total_sales": [400, 400]
        }, index=pd.Index(["A", "B"], name="category"))
        pd.testing.assert_frame_equal(result, expected)


class TestTopNPerGroup(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "group": ["X", "X", "X", "Y", "Y"],
            "score": [10, 50, 30, 20, 40]
        }, index=[100, 101, 102, 103, 104])

    def test_top_n_basic(self):
        result = m.top_n_per_group(self.df, "group", "score", 2)
        expected = self.df.loc[[101, 104, 102, 103]]
        pd.testing.assert_frame_equal(result, expected)

    def test_top_n_larger_than_group_size(self):
        result = m.top_n_per_group(self.df, "group", "score", 5)
        expected = self.df.loc[[101, 104, 102, 103, 100]]
        pd.testing.assert_frame_equal(result, expected)

    def test_top_n_with_n_equals_one(self):
        result = m.top_n_per_group(self.df, "group", "score", 1)
        expected = self.df.loc[[101, 104]]
        pd.testing.assert_frame_equal(result, expected)

    def test_empty_dataframe(self):
        df_empty = pd.DataFrame(columns=["group", "score"])
        result = m.top_n_per_group(df_empty, "group", "score", 2)
        self.assertTrue(result.empty)


class TestPivotMatrix(unittest.TestCase):
    def setUp(self): 
        self.df = pd.DataFrame({
            "user": ["Alice", "Alice", "Bob", "Charlie"],
            "item": ["item1", "item2", "item1", "item3"],
            "rating": [5.0, 4.0, 3.0, 2.0]
        })

    def test_pivot_basic(self):
        result = m.pivot_matrix(self.df, "user", "item", "rating", 0.0)
        expected = pd.DataFrame({
            "item1": [5.0, 3.0, 0.0],
            "item2": [4.0, 0.0, 0.0],
            "item3": [0.0, 0.0, 2.0]
        }, index=pd.Index(["Alice", "Bob", "Charlie"], name="user"))
        expected.columns.name = "item"
        pd.testing.assert_frame_equal(result, expected)

    def test_pivot_fills_nans(self):
        result = m.pivot_matrix(self.df, "user", "item", "rating", -1.0)
        expected = pd.DataFrame({
            "item1": [5.0, 3.0, -1.0],
            "item2": [4.0, -1.0, -1.0],
            "item3": [-1.0, -1.0, 2.0]
        }, index=pd.Index(["Alice", "Bob", "Charlie"], name="user"))
        expected.columns.name = "item"
        pd.testing.assert_frame_equal(result, expected)

    def test_pivot_empty(self):
        df_empty = pd.DataFrame(columns=["user", "item", "rating"])
        result = m.pivot_matrix(df_empty, "user", "item", "rating", 0.0)
        self.assertTrue(result.empty)

    def test_pivot_custom_fill_value(self):
        result = m.pivot_matrix(self.df, "user", "item", "rating", 99.0)
        expected = pd.DataFrame({
            "item1": [5.0, 3.0, 99.0],
            "item2": [4.0, 99.0, 99.0],
            "item3": [99.0, 99.0, 2.0]
        }, index=pd.Index(["Alice", "Bob", "Charlie"], name="user"))
        expected.columns.name = "item"
        pd.testing.assert_frame_equal(result, expected)


class TestJoinWithLookup(unittest.TestCase):
    def setUp(self):
        self.left = pd.DataFrame({
            "tx_id": [1, 2, 3, 4],
            "prod_id": ["A", "B", "C", "A"]
        })
        self.right = pd.DataFrame({
            "prod_id": ["A", "B"],
            "price": [10.0, 20.0]
        })

    def test_join_some_unmatched(self):
        joined, unmatched = m.join_with_lookup(self.left, self.right, "prod_id")
        self.assertEqual(unmatched, {"C"})
        expected_joined = pd.DataFrame({
            "tx_id": [1, 2, 3, 4],
            "prod_id": ["A", "B", "C", "A"],
            "price": [10.0, 20.0, np.nan, 10.0]
        })
        pd.testing.assert_frame_equal(joined, expected_joined)

    def test_join_all_matched(self):
        right_all = pd.DataFrame({
            "prod_id": ["A", "B", "C"],
            "price": [10.0, 20.0, 30.0]
        })
        joined, unmatched = m.join_with_lookup(self.left, right_all, "prod_id")
        self.assertEqual(unmatched, set())

    def test_join_with_nans_in_keys(self):
        left_nan = pd.DataFrame({
            "tx_id": [1, 2],
            "prod_id": ["A", None]
        })
        joined, unmatched = m.join_with_lookup(left_nan, self.right, "prod_id")
        self.assertEqual(unmatched, set())

    def test_join_empty_right(self):
        right_empty = pd.DataFrame(columns=["prod_id", "price"])
        joined, unmatched = m.join_with_lookup(self.left, right_empty, "prod_id")
        self.assertEqual(unmatched, {"A", "B", "C"})


class TestRollingAverage(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "ticker": ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT"],
            "date": ["2023-01-03", "2023-01-01", "2023-01-02", "2023-01-02", "2023-01-01"],
            "price": [150.0, 100.0, 120.0, 250.0, 200.0]
        }, index=[10, 20, 30, 40, 50])

    def test_rolling_basic(self):
        result = m.rolling_average(self.df, "ticker", "date", "price", 2)
        expected_index = [20, 30, 10, 50, 40]
        expected_values = [100.0, 110.0, 135.0, 200.0, 225.0]
        expected = pd.Series(expected_values, index=expected_index, name="price")
        pd.testing.assert_series_equal(result, expected)

    def test_rolling_min_periods(self):
        result = m.rolling_average(self.df, "ticker", "date", "price", 3)
        expected_index = [20, 30, 10, 50, 40]
        expected_values = [100.0, 110.0, 123.333333, 200.0, 225.0]
        expected = pd.Series(expected_values, index=expected_index, name="price")
        pd.testing.assert_series_equal(result, expected, rtol=1e-5)

    def test_rolling_unsorted_input(self):
        # Scramble index and verify chronological order is preserved
        scrambled = self.df.sample(frac=1, random_state=42)
        result = m.rolling_average(scrambled, "ticker", "date", "price", 2)
        expected_index = [20, 30, 10, 50, 40]
        expected_values = [100.0, 110.0, 135.0, 200.0, 225.0]
        expected = pd.Series(expected_values, index=expected_index, name="price")
        pd.testing.assert_series_equal(result, expected)

    def test_rolling_large_window(self):
        result = m.rolling_average(self.df, "ticker", "date", "price", 10)
        expected_index = [20, 30, 10, 50, 40]
        expected_values = [100.0, 110.0, 123.333333, 200.0, 225.0]
        expected = pd.Series(expected_values, index=expected_index, name="price")
        pd.testing.assert_series_equal(result, expected, rtol=1e-5)


class TestPercentChange(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "ticker": ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT"],
            "date": ["2023-01-03", "2023-01-01", "2023-01-02", "2023-01-02", "2023-01-01"],
            "price": [150.0, 100.0, 120.0, 250.0, 200.0]
        }, index=[10, 20, 30, 40, 50])

    def test_pct_change_basic(self):
        result = m.percent_change(self.df, "ticker", "date", "price")
        expected_index = [20, 30, 10, 50, 40]
        expected_values = [np.nan, 0.2, 0.25, np.nan, 0.25]
        expected = pd.Series(expected_values, index=expected_index, name="price")
        pd.testing.assert_series_equal(result, expected)

    def test_pct_change_first_is_nan(self):
        result = m.percent_change(self.df, "ticker", "date", "price")
        self.assertTrue(np.isnan(result.loc[20]))
        self.assertTrue(np.isnan(result.loc[50]))

    def test_pct_change_unsorted_input(self):
        scrambled = self.df.sample(frac=1, random_state=42)
        result = m.percent_change(scrambled, "ticker", "date", "price")
        expected_index = [20, 30, 10, 50, 40]
        expected_values = [np.nan, 0.2, 0.25, np.nan, 0.25]
        expected = pd.Series(expected_values, index=expected_index, name="price")
        pd.testing.assert_series_equal(result, expected)

    def test_pct_change_zero_division(self):
        df_zero = pd.DataFrame({
            "ticker": ["AAPL", "AAPL"],
            "date": ["2023-01-01", "2023-01-02"],
            "price": [0.0, 100.0]
        }, index=[1, 2])
        result = m.percent_change(df_zero, "ticker", "date", "price")
        self.assertEqual(result.loc[2], np.inf)


if __name__ == "__main__":
    unittest.main(verbosity=2)
