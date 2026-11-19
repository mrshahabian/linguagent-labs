# 52 — Loading and cleaning data with pandas

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/applied-ds/52-pandas-cleaning/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/applied-ds/pandas-cleaning?utm_source=github&utm_medium=labs)

*~45 minutes · 21 tests · Intermediate*

In real-world data science, up to 80% of your time is spent cleaning and preparing data. Raw datasets are notoriously messy: they contain inconsistent casing, trailing whitespaces, incorrect data types, missing values, and duplicate records. If you feed this raw, uncleaned data directly into a machine learning model or an analytical pipeline, you will get unreliable predictions and incorrect business insights. 

In interviews, you will frequently be given a messy dataset and asked to clean it under a tight time limit. Knowing how to write robust, vectorized, and reproducible cleaning pipelines using pandas is a core skill that separates junior developers from experienced data professionals. This module teaches you how to build a robust data-cleaning pipeline that transforms a messy CSV string into a tidy, typed, and validated DataFrame.

## What you'll build

Six functions in `exercises.py` that form a complete data-cleaning pipeline. Keep this table open as you go:

| Function | You'll implement | Shows up in real code as |
|---|---|---|
| `load_csv` | Parsing CSV strings with explicit types and dates | Ingesting raw data pipelines safely without silent type coercion |
| `normalize_columns` | Standardizing column headers to snake_case | Standardizing raw schemas from external APIs or legacy databases |
| `clean_text_column` | Normalizing messy string data (spaces, casing) | Preparing categorical variables or text features for ML models |
| `fill_missing` | Handling missing values with custom strategies | Imputing missing features before training or evaluation |
| `drop_duplicate_rows` | Deduplicating records based on chronological order | Resolving duplicate events or logs by keeping the latest state |
| `data_quality_report` | Summarizing missingness and data types | Automated data validation reports in production pipelines |

---

## 1. Loading data with explicit types

```python
>>> import io
>>> import pandas as pd
>>> csv_data = "id,zip,date\n1,02108,2023-01-01\n2,90210,2023-01-02"
>>> df = pd.read_csv(io.StringIO(csv_data), dtype={"zip": str}, parse_dates=["date"])
>>> df["zip"].iloc[0]
'02108'
```

By default, pandas tries to infer data types automatically. While convenient, this inference is a common source of silent bugs. For example, postal codes like `'02108'` or numeric identifiers with leading zeros will be parsed as integers, stripping the leading zero and turning `'02108'` into the integer `2108`. Similarly, date columns will be loaded as generic string objects unless explicitly parsed. Always define your schema up front using the `dtype` and `parse_dates` arguments of `pd.read_csv`.

**In `exercises.py`:** `load_csv` is this section's target.

**Test it:** `python3 -m unittest test_exercises.TestLoadCsv -v`

## 2. Normalizing column names

```python
>>> df = pd.DataFrame(columns=[" First Name ", "Last-Name!", "EMAIL ADDRESS"])
>>> # Standardizing to lowercase, replacing non-alphanumeric with underscores
>>> df.columns = ["first_name", "last_name", "email_address"]
```

Raw datasets often have column names with inconsistent casing, trailing spaces, and special characters (e.g., `" First Name "`, `"Last-Name!"`). This makes querying columns tedious and error-prone. Standardizing column names to `snake_case` (lowercase, alphanumeric characters, words separated by a single underscore, and no leading/trailing underscores) ensures consistency across your entire codebase.

Avoid mutating `df.columns` in place with fragile manual loops. Instead, write a pure function that returns a copy of the DataFrame with normalized column names using regular expressions.

**In `exercises.py`:** `normalize_columns` is this section's target.

**Test it:** `python3 -m unittest test_exercises.TestNormalizeColumns -v`

## 3. Cleaning text columns

```python
>>> s = pd.Series(["  New   York ", None, " chicago "])
>>> # Strip, lowercase, and collapse multiple spaces
>>> s.dropna().astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
0      new york
2       chicago
dtype: object        (pandas 3 shows: str)
```

Categorical text columns often contain dirty strings with inconsistent casing and extra spaces. To group or match these values correctly, you must normalize them. 

A common pitfall is converting the entire column to string type first using `.astype(str)`. Doing this converts missing values (`NaN` or `None`) into the literal string `"nan"` or `"None"`, which corrupts your data. You must clean the text while carefully preserving actual missing values as `NaN`.

**In `exercises.py`:** `clean_text_column` is this section's target.

**Test it:** `python3 -m unittest test_exercises.TestCleanTextColumn -v`

## 4. Filling missing values

```python
>>> df = pd.DataFrame({"age": [25, None, 35]})
>>> median_age = df["age"].median()
>>> df["age"].fillna(median_age)
0    25.0
1    30.0
2    35.0
Name: age, dtype: float64
```

Missing data is inevitable. Depending on the column, you might want to fill missing values using different strategies:
- `mean`: Use the column's average (best for symmetric numerical data).
- `median`: Use the middle value (best for skewed numerical data with outliers).
- `mode`: Use the most frequent value (best for categorical data). If there are multiple modes, use the first one.
- `constant`: Use a specific fallback value (e.g., `0` or `"Unknown"`).

Make sure your implementation handles columns that do not exist in the DataFrame gracefully without crashing.

**In `exercises.py`:** `fill_missing` is this section's target.

**Test it:** `python3 -m unittest test_exercises.TestFillMissing -v`

## 5. Deduplicating rows chronologically

```python
>>> df = pd.DataFrame({
...     "user_id": [1, 1, 2],
...     "status": ["silver", "gold", "bronze"],
...     "updated_at": ["2023-01-01", "2023-01-02", "2023-01-01"]
... })
>>> df = df.sort_values("updated_at")
>>> df.drop_duplicates(subset=["user_id"], keep="last")
   user_id  status  updated_at
1        1    gold  2023-01-02
2        2  bronze  2023-01-01
```

When processing event logs or transactional data, you will often find duplicate records for the same entity. To get the current state of an entity, you must keep only the latest record based on a timestamp column.

A major pitfall is calling `.drop_duplicates()` without sorting first. If the raw file is not ordered chronologically, keeping the "last" record will yield arbitrary, incorrect results. Always convert your timestamp column to datetime, sort ascending, and then drop duplicates keeping the last occurrence.

**In `exercises.py`:** `drop_duplicate_rows` is this section's target.

**Test it:** `python3 -m unittest test_exercises.TestDropDuplicateRows -v`

## 6. Generating a data quality report

```python
>>> df = pd.DataFrame({"a": [1, None, 3], "b": ["x", "y", "x"]})
>>> df.isna().sum()
a    1
b    0
dtype: int64
```

Before and after cleaning, you need to verify the state of your dataset. A data quality report provides a quick, structured summary of your DataFrame's columns, including their data types, missing value counts, missing value percentages, and the count of unique values.

Avoid writing slow Python loops over the rows of your DataFrame to calculate these statistics. Instead, leverage pandas' highly optimized, vectorized methods like `.isna().sum()`, `.nunique()`, and `.dtypes` to build the report efficiently.

**In `exercises.py`:** `data_quality_report` is this section's target.

**Test it:** `python3 -m unittest test_exercises.TestDataQualityReport -v`

---

## The exercises

| Function | What it teaches |
|---|---|
| `load_csv` | Safe ingestion with explicit data typing and date parsing |
| `normalize_columns` | Regular expression-based column header standardization |
| `clean_text_column` | Vectorized string normalization while preserving missing values |
| `fill_missing` | Applying column-specific imputation strategies (mean, median, mode, constant) |
| `drop_duplicate_rows` | Chronological deduplication using sorting and subset constraints |
| `data_quality_report` | Vectorized aggregation of metadata for data validation |

## Check your understanding

1. **Why is it dangerous to let pandas automatically infer column types when loading a CSV?**
   <details><summary>Answer</summary>Pandas might strip critical leading zeros from numeric codes (like ZIP codes or ID numbers) by parsing them as integers, or fail to parse date strings as actual datetime objects, preventing chronological sorting and date arithmetic.</details>

2. **Why shouldn't we use `.astype(str)` directly on a text column containing missing values?**
   <details><summary>Answer</summary>Calling <code>.astype(str)</code> converts <code>NaN</code> or <code>None</code> values into the literal string <code>"nan"</code> or <code>"None"</code>. This makes it impossible to detect missing values later using <code>.isna()</code>.</details>

3. **How does `drop_duplicates(keep='last')` behave if the DataFrame is not sorted chronologically?**
   <details><summary>Answer</summary>It will keep whichever row happens to appear last in the physical file, which may not represent the most recent logical event. You must sort by a timestamp column first.</details>

4. **If a column has multiple modes, how should `fill_missing` handle it when using the `'mode'` strategy?**
   <details><summary>Answer</summary>The <code>.mode()</code> method returns a Series of all tied modes. To ensure a deterministic scalar value is used for filling, you should select the first mode (index 0) from that Series.</details>

5. **What is the benefit of returning a copy of the DataFrame in these functions instead of modifying them in place?**
   <details><summary>Answer</summary>Returning copies avoids side effects, keeps your functions pure, and allows you to chain operations together safely without accidentally corrupting the original raw dataset.</details>

## Interview questions this module answers

- *How do you prevent pandas from stripping leading zeros from ID columns when reading a CSV file?*
- *How do you clean a messy string column in pandas without converting missing values (NaN) to the string "nan"?*
- *What is the correct way to deduplicate a dataset to keep only the most recent record for each user?*
- *How would you write a function to generate a summary of missing values and unique counts for every column in a DataFrame?*
