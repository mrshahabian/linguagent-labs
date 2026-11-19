"""53 - Exploring data: groupby, pivots and joins

Reference implementations for exercises.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Set


def summary_by(df: pd.DataFrame, key: str, metrics: Dict[str, Tuple[str, str]]) -> pd.DataFrame:
    agg_dict = {new_col: (src_col, agg_func) for new_col, (src_col, agg_func) in metrics.items()}
    return df.groupby(key).agg(**agg_dict)


def top_n_per_group(df: pd.DataFrame, group_col: str, sort_col: str, n: int) -> pd.DataFrame:
    sorted_df = df.sort_values(by=sort_col, ascending=False)
    return sorted_df.groupby(group_col).head(n)


def pivot_matrix(df: pd.DataFrame, index_col: str, columns_col: str, values_col: str, fill_value: float) -> pd.DataFrame:
    return df.pivot(index=index_col, columns=columns_col, values=values_col).fillna(fill_value)


def join_with_lookup(left_df: pd.DataFrame, right_df: pd.DataFrame, on_col: str) -> Tuple[pd.DataFrame, Set]:
    joined = pd.merge(left_df, right_df, on=on_col, how='left')
    left_keys = set(left_df[on_col].dropna().unique())
    right_keys = set(right_df[on_col].dropna().unique())
    unmatched = left_keys - right_keys
    return joined, unmatched


def rolling_average(df: pd.DataFrame, group_col: str, time_col: str, value_col: str, window_size: int) -> pd.Series:
    sorted_df = df.sort_values(by=[group_col, time_col])
    rolling_series = sorted_df.groupby(group_col)[value_col].rolling(window=window_size, min_periods=1).mean()
    return rolling_series.reset_index(level=0, drop=True)


def percent_change(df: pd.DataFrame, group_col: str, time_col: str, value_col: str) -> pd.Series:
    sorted_df = df.sort_values(by=[group_col, time_col])
    return sorted_df.groupby(group_col)[value_col].pct_change()
