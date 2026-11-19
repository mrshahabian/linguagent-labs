# 53 — Exploring data: groupby, pivots and joins

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/applied-ds/53-groupby-analysis/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/applied-ds/groupby-analysis?utm_source=github&utm_medium=labs)

*~45 minutes · 24 tests · Intermediate*

In real-world data science, raw data is rarely delivered in a clean, pre-aggregated format. It arrives as a stream of transactions, user events, or sensor readings. To extract business value, you must group these records, pivot them into matrices for machine learning models, and join them with metadata tables.

In interviews, pandas manipulation questions are the most common practical test. Interviewers don't just look for *a* working solution; they look for idiomatic pandas. They watch for common anti-patterns like using Python `for` loops over DataFrame rows, failing to handle missing values after a pivot, or ignoring chronological sorting before computing rolling windows.

This module covers the core operations required to reshape, aggregate, and analyze multi-group time-series data efficiently.

## What you'll build

Six data manipulation functions in `exercises.py`, in this order. Keep this table open as you go:

| Function | You'll implement | Shows up in real code as |
|---|---|---|
| `summary_by` | Dynamic named aggregations on a grouped DataFrame | Generating custom KPI dashboards with mixed metrics (e.g., sum of revenue, mean of rating) |
| `top_n_per_group` | Efficient retrieval of the top N records within each group | Finding the top 3 highest-value transactions for every customer |
| `pivot_matrix` | Reshaping long-format data into wide-format matrices with fill values | Preparing user-item interaction matrices for collaborative filtering recommendation systems |
| `join_with_lookup` | Left join that tracks and reports unmatched keys | Merging transaction logs with a product catalog while flagging missing product IDs |
| `rolling_average` | Group-aware chronological rolling window averages | Smoothing noisy daily stock prices or sensor readings per device |
| `percent_change` | Group-aware period-over-period percentage changes | Calculating month-over-month revenue growth rates per region |

---

## 1. Named aggregations with groupby

When you call `df.groupby('key')`, pandas does not immediately compute anything. It creates a `DataFrameGroupBy` object. This is the "Split" phase of the classic Split-Apply-Combine pattern. The actual computation happens during the "Apply" and "Combine" phases when you call an aggregation method like `.agg()`.

A common pitfall is passing a list of functions to `.agg()`, which results in a MultiIndex column structure:

```python
>>> df.groupby('category')['revenue'].agg(['sum', 'mean'])
          sum       mean
category
A        1500  500.00000
B        2400  800.00000
```

MultiIndex columns are notoriously difficult to query and flat-out break many downstream tools. Selecting a column requires a tuple like `df[("revenue", "sum")]`, and flattening them later requires ugly string manipulation.

To avoid this, pandas supports **Named Aggregation**. You pass keyword arguments to `.agg()`, where the keyword is the output column name, and the value is a tuple of `(source_column, aggregation_function)`:

```python
>>> df.groupby('category').agg(
...     total_revenue=('revenue', 'sum'),
...     average_rating=('rating', 'mean')
... )
          total_revenue  average_rating
category
A                  1500             4.2
B                  2400             4.7
```

If you need to build this dynamically (for example, if the metrics are passed as a dictionary to a function), you can unpack a dictionary of tuples using the `**` operator:

```python
>>> metrics = {
...     'total_revenue': ('revenue', 'sum'),
...     'average_rating': ('rating', 'mean')
... }
>>> agg_dict = {new_col: (src_col, agg_func) for new_col, (src_col, agg_func) in metrics.items()}
>>> df.groupby('category').agg(**agg_dict)
```

**In `exercises.py`:** Implement `summary_by` to group a DataFrame by a key and apply dynamic named aggregations.

**Test it:** `python3 -m unittest test_exercises.TestSummaryBy -v`

---

## 2. Top N rows per group

Often, you need to find the "best" or "worst" records within each category—for example, the top 3 highest-spending customers per region, or the 5 most recent transactions per user.

A common but incredibly slow approach is to use `.apply()` with a custom lambda function:

```python
# Slow anti-pattern!
>>> df.groupby('group').apply(lambda x: x.sort_values('score', ascending=False).head(3))
```

Why is this slow? The `.apply()` method forces pandas to drop out of its highly optimized vectorized C-engine and loop through each group in Python, constructing a new DataFrame, sorting it, slicing it, and then concatenating them all back together. On a dataset with millions of rows and thousands of groups, this will grind your pipeline to a halt.

A much faster, idiomatic approach is:
1. Sort the entire DataFrame globally by the sorting column in descending order.
2. Group by the grouping column.
3. Call `.head(n)` on the GroupBy object.

```python
>>> sorted_df = df.sort_values('score', ascending=False)
>>> sorted_df.groupby('group').head(3)
```

Because the DataFrame is already sorted, `.head(n)` simply takes the first `n` rows for each group. This operation is fully vectorized and runs orders of magnitude faster on large datasets.

**In `exercises.py`:** Implement `top_n_per_group` using this vectorized sorting and grouping pattern.

**Test it:** `python3 -m unittest test_exercises.TestTopNPerGroup -v`

---

## 3. Pivoting tables with fill values

Data often arrives in "long" format (one row per observation, e.g., user interactions, sensor logs). For machine learning algorithms or matrix factorization (like collaborative filtering), you need "wide" format (one row per entity, one column per feature).

Pandas provides `df.pivot(index, columns, values)` to reshape data from long to wide:

```python
>>> df.pivot(index='user_id', columns='movie_id', values='rating')
movie_id   101  102  103
user_id
1          5.0  NaN  4.0
2          NaN  3.0  NaN
```

However, if a user has not rated a specific movie, pandas inserts `NaN`. Most machine learning models cannot handle `NaN` values and will crash during training. You must fill these missing values with a sensible default (like `0.0` for ratings or interaction counts).

While you can chain `.fillna()` after pivoting, you must ensure that the fill value is applied correctly across the entire pivoted matrix without losing the index structure.

*Note:* If your long-format data has duplicate entries for the same index/column pair (e.g., a user rated the same movie twice), `df.pivot()` will raise a `ValueError`. In those cases, you must use `df.pivot_table()`, which aggregates duplicates. For this exercise, we assume the index/column pairs are unique and use `df.pivot()`.

**In `exercises.py`:** Implement `pivot_matrix` to pivot a DataFrame and fill missing values.

**Test it:** `python3 -m unittest test_exercises.TestPivotMatrix -v`

---

## 4. Joins and tracking unmatched keys

In relational databases, you join tables using SQL `LEFT JOIN`. In pandas, this is done with `pd.merge(left, right, on='key', how='left')`.

A common pitfall in data engineering is assuming that your lookup table (the right DataFrame) contains every key present in your transaction table (the left DataFrame). If keys are missing from the right DataFrame, pandas silently fills the right-side columns with `NaN`:

```python
>>> tx = pd.DataFrame({'tx_id': [1, 2], 'prod_id': ['A', 'X']})
>>> catalog = pd.DataFrame({'prod_id': ['A'], 'price': [10.0]})
>>> pd.merge(tx, catalog, on='prod_id', how='left')
   tx_id prod_id  price
0      1       A   10.0
1      2       X    NaN
```

If you do not audit these missing keys, you might introduce silent bugs. For example, if you join transactions with a product catalog to get prices, missing product IDs will result in `NaN` prices, which might be dropped or filled with 0, corrupting your financial metrics.

To prevent this, always audit your joins by identifying which keys in the left DataFrame had no match in the right DataFrame. You can find this by taking the set difference of the key columns:

```python
>>> unmatched = set(tx['prod_id']) - set(catalog['prod_id'])
>>> unmatched
{'X'}
```

**In `exercises.py`:** Implement `join_with_lookup` to perform a left join and return both the joined DataFrame and the set of unmatched keys.

**Test it:** `python3 -m unittest test_exercises.TestJoinWithLookup -v`

---

## 5. Rolling averages on grouped time series

A rolling average (or moving average) is used to smooth out short-term fluctuations and highlight longer-term trends. In pandas, you use `df[col].rolling(window).mean()`.

However, if your dataset contains multiple groups (e.g., stock prices for multiple tickers), calculating a rolling average across the entire DataFrame is a disaster. The rolling window at the transition point will mix prices from "AAPL" and "MSFT", producing mathematically meaningless results.

To calculate rolling metrics per group:
1. Sort the DataFrame by the group column and the time column to ensure chronological order within each group.
2. Group by the group column.
3. Select the value column and call `.rolling(window, min_periods=1).mean()`.

```python
>>> sorted_df = df.sort_values(by=['ticker', 'date'])
>>> sorted_df.groupby('ticker')['price'].rolling(window=3, min_periods=1).mean()
```

Why `min_periods=1`? By default, a rolling window of size `N` requires `N` non-null values to produce a result, meaning the first `N-1` rows of each group will be `NaN`. Setting `min_periods=1` allows pandas to compute the average of whatever values are available (e.g., 1 value for the first row, average of 2 for the second, etc.).

**Crucial detail:** `groupby().rolling()` returns a Series with a MultiIndex: `(group_col, original_index)`. To align this back to your sorted DataFrame, you must drop the group level from the index using `.reset_index(level=0, drop=True)`.

**In `exercises.py`:** Implement `rolling_average` to compute group-aware rolling averages on sorted data.

**Test it:** `python3 -m unittest test_exercises.TestRollingAverage -v`

---

## 6. Percent change within groups

Calculating period-over-period growth (e.g., daily or monthly percentage change) is a standard financial task. Pandas provides `.pct_change()` for this.

Just like rolling windows, calling `.pct_change()` on an ungrouped DataFrame will calculate the percentage change between the first record of one group and the last record of the previous group.

To do this correctly:
1. Sort the DataFrame by the group column and the time column.
2. Group by the group column.
3. Select the value column and call `.pct_change()`.

```python
>>> sorted_df = df.sort_values(by=['ticker', 'date'])
>>> sorted_df.groupby('ticker')['price'].pct_change()
```

**Crucial detail:** Unlike `.rolling()`, `groupby().pct_change()` does *not* return a MultiIndex. It returns a Series with the exact same index and order as the input DataFrame (`sorted_df`). This inconsistency in pandas' API is a frequent source of confusion, making it a favorite topic for advanced interviews!

**In `exercises.py`:** Implement `percent_change` to compute group-aware percentage changes.

**Test it:** `python3 -m unittest test_exercises.TestPercentChange -v`

---

## The exercises

| Function | What it teaches |
|---|---|
| `summary_by` | Dynamic named aggregations with `groupby` |
| `top_n_per_group` | Vectorized sorting and grouping instead of slow `.apply()` |
| `pivot_matrix` | Reshaping long data to wide and filling missing values |
| `join_with_lookup` | Left joins and auditing unmatched keys |
| `rolling_average` | Group-aware rolling windows and handling MultiIndex returns |
| `percent_change` | Group-aware percentage changes and API index consistency |

## Check your understanding

1. **Why is named aggregation preferred over passing a list of functions to `.agg()`?**
   <details><summary>Answer</summary>Passing a list of functions creates MultiIndex columns, which are difficult to select and manipulate. Named aggregation allows you to specify clean, flat output column names directly during the aggregation step.</details>

2. **Why is sorting globally and using `groupby().head(n)` faster than using `groupby().apply(lambda x: x.sort_values().head(n))`?**
   <details><summary>Answer</summary>The global sorting and <code>.head(n)</code> approach is fully vectorized and executed in pandas' optimized C-engine. The <code>.apply()</code> approach drops out of vectorization to run a Python loop for every single group, which is extremely slow on large datasets.</details>

3. **What is the difference between `df.pivot()` and `df.pivot_table()`?**
   <details><summary>Answer</summary><code>df.pivot()</code> reshapes data but raises a <code>ValueError</code> if there are duplicate entries for any index/column pair. <code>df.pivot_table()</code> is a generalization of pivot that can handle duplicates by aggregating them (using a function like mean, sum, etc.).</details>

4. **Why should you audit left joins for unmatched keys?**
   <details><summary>Answer</summary>A left join silently inserts <code>NaN</code> values for columns from the right DataFrame if a key in the left DataFrame has no match. If left unchecked, these missing values can corrupt downstream calculations or machine learning model training.</details>

5. **Why does `groupby().rolling()` require `.reset_index(level=0, drop=True)` while `groupby().pct_change()` does not?**
   <details><summary>Answer</summary>This is an inconsistency in the pandas API. <code>groupby().rolling()</code> returns a MultiIndex Series containing the group key as the outer index level, which must be dropped to align with the original DataFrame. <code>groupby().pct_change()</code> returns a flat Series already aligned with the input DataFrame's index.</details>

## Interview questions this module answers

- *How do you perform dynamic named aggregations in pandas?*
- *What is the most efficient way to find the top N records per group in a large DataFrame?*
- *How do you handle missing values when reshaping data from long to wide format?*
- *How do you calculate a rolling average within groups without leaking data between groups?*
- *Why does `groupby().rolling()` return a MultiIndex, and how do you align it back to your original DataFrame?*
