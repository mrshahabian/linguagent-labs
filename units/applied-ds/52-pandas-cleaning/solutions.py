"""Reference solutions for 52 — Loading and cleaning data with pandas
"""

import io
import re
import typing
import numpy as np
import pandas as pd


def load_csv(
    csv_text: str,
    dtypes: typing.Dict[str, typing.Any] = None,
    parse_dates: typing.List[str] = None
) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(csv_text), dtype=dtypes, parse_dates=parse_dates)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    new_columns = []
    for col in df_copy.columns:
        col_str = str(col).strip().lower()
        # Replace non-alphanumeric characters with underscores
        normalized = re.sub(r"[^a-z0-9]+", "_", col_str)
        # Strip leading/trailing underscores
        normalized = normalized.strip("_")
        new_columns.append(normalized)
    df_copy.columns = new_columns
    return df_copy


def clean_text_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df_copy = df.copy()
    if column in df_copy.columns:
        series = df_copy[column]
        # Clean only non-null elements to preserve actual NaNs
        cleaned = (
            series.dropna()
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)
        )
        df_copy[column] = cleaned.reindex(df_copy.index)
    return df_copy


def fill_missing(df: pd.DataFrame, strategies: typing.Dict[str, typing.Any]) -> pd.DataFrame:
    df_copy = df.copy()
    for col, strategy in strategies.items():
        if col not in df_copy.columns:
            continue
        if strategy == "mean":
            val = df_copy[col].mean()
        elif strategy == "median":
            val = df_copy[col].median()
        elif strategy == "mode":
            modes = df_copy[col].mode()
            val = modes.iloc[0] if not modes.empty else np.nan
        else:
            val = strategy
        df_copy[col] = df_copy[col].fillna(val)
    return df_copy


def drop_duplicate_rows(df: pd.DataFrame, subset: typing.List[str], timestamp_col: str) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy[timestamp_col] = pd.to_datetime(df_copy[timestamp_col])
    df_copy = df_copy.sort_values(by=timestamp_col, ascending=True)
    df_copy = df_copy.drop_duplicates(subset=subset, keep="last")
    return df_copy.reset_index(drop=True)


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    report_data = []
    total_rows = len(df)
    for col in df.columns:
        dtype = str(df[col].dtype)
        missing_count = int(df[col].isna().sum())
        missing_pct = float(missing_count / total_rows) if total_rows > 0 else 0.0
        unique_count = int(df[col].nunique(dropna=True))
        report_data.append({
            "column": col,
            "dtype": dtype,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "unique_count": unique_count
        })
    
    if report_data:
        report_df = pd.DataFrame(report_data).set_index("column")
    else:
        report_df = pd.DataFrame(columns=["dtype", "missing_count", "missing_pct", "unique_count"])
        report_df.index.name = "column"
    return report_df
