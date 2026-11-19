"""53 - Exploring data: groupby, pivots and joins

Implement each function so the tests in test_exercises.py pass. Read
README.md first; every exercise below is covered there.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Set


def summary_by(df: pd.DataFrame, key: str, metrics: Dict[str, Tuple[str, str]]) -> pd.DataFrame:
    """Group a DataFrame by a key and apply named aggregations.

    Args:
        df: The input pandas DataFrame.
        key: The column name to group by.
        metrics: A dictionary mapping output column names to tuples of (source_column, aggregation_function).
                 Example: {"total_sales": ("sales", "sum"), "avg_rating": ("rating", "mean")}

    Returns:
        A pandas DataFrame with the group key as the index and the aggregated metrics as columns.
    """
    raise NotImplementedError("your code here")


def top_n_per_group(df: pd.DataFrame, group_col: str, sort_col: str, n: int) -> pd.DataFrame:
    """Find the top N rows per group based on a sorting column in descending order.

    Args:
        df: The input pandas DataFrame.
        group_col: The column name to group by.
        sort_col: The column name to sort by (descending).
        n: The number of top rows to return per group.

    Returns:
        A pandas DataFrame containing the top N rows per group, preserving the original row indices.
    """
    raise NotImplementedError("your code here")


def pivot_matrix(df: pd.DataFrame, index_col: str, columns_col: str, values_col: str, fill_value: float) -> pd.DataFrame:
    """Pivot a DataFrame from long to wide format and fill missing values.

    Args:
        df: The input pandas DataFrame.
        index_col: The column name to use as the row index.
        columns_col: The column name to use as the columns.
        values_col: The column name to use as the values.
        fill_value: The value to fill NaNs with.

    Returns:
        A pivoted pandas DataFrame with missing values filled.
    """
    raise NotImplementedError("your code here")


def join_with_lookup(left_df: pd.DataFrame, right_df: pd.DataFrame, on_col: str) -> Tuple[pd.DataFrame, Set]:
    """Perform a left join and identify unmatched keys from the left DataFrame.

    Args:
        left_df: The left pandas DataFrame (e.g., transactions).
        right_df: The right pandas DataFrame (e.g., lookup catalog).
        on_col: The column name to join on.

    Returns:
        A tuple of:
            - The joined pandas DataFrame (left join).
            - A set of keys in left_df[on_col] that were not found in right_df[on_col].
    """
    raise NotImplementedError("your code here")


def rolling_average(df: pd.DataFrame, group_col: str, time_col: str, value_col: str, window_size: int) -> pd.Series:
    """Calculate a rolling average of a column per group, sorted chronologically.

    Args:
        df: The input pandas DataFrame.
        group_col: The column name to group by.
        time_col: The column name representing time (for sorting).
        value_col: The column name to calculate the rolling average on.
        window_size: The size of the rolling window.

    Returns:
        A pandas Series containing the rolling average, with an index matching the sorted DataFrame.
        Uses min_periods=1 to avoid initial NaNs.
    """
    raise NotImplementedError("your code here")


def percent_change(df: pd.DataFrame, group_col: str, time_col: str, value_col: str) -> pd.Series:
    """Calculate the percentage change of a column compared to the previous period within each group.

    Args:
        df: The input pandas DataFrame.
        group_col: The column name to group by.
        time_col: The column name representing time (for sorting).
        value_col: The column name to calculate the percentage change on.

    Returns:
        A pandas Series containing the percentage change, with an index matching the sorted DataFrame.
    """
    raise NotImplementedError("your code here")
