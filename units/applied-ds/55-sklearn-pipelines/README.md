# 55 — Features and pipelines with scikit-learn

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/applied-ds/55-sklearn-pipelines/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/applied-ds/sklearn-pipelines?utm_source=github&utm_medium=labs)

*~40 minutes · 26 tests*

Data leakage is one of the most common and embarrassing mistakes in machine learning. It occurs when information from the test set (or future data) accidentally leaks into the training process. When this happens, your model will perform beautifully during training and validation, but fail miserably when deployed to production.

The most frequent culprit is preprocessing. If you scale your features, impute missing values, or encode categorical variables using the entire dataset *before* splitting it into training and testing sets, you have leaked the global mean, standard deviation, and category frequencies into your training loop.

In this module, you will learn how to build robust, leak-proof machine learning workflows using scikit-learn's `Pipeline` and `ColumnTransformer`. These tools package your preprocessing steps and estimator together, ensuring that all transformations are fit *only* on the training split and safely applied to the test split.

## What you'll build

Five functions in `exercises.py` that build a complete, leak-proof classification pipeline and demonstrate the real-world impact of data leakage:

| Function / class | You'll implement | Shows up in real code as |
|---|---|---|
| `stratified_split` | Stratified train/test split | Ensuring class proportions are preserved in both splits |
| `build_preprocessor` | ColumnTransformer with imputation, scaling, and encoding | Preprocessing different columns differently without leaking data |
| `build_pipeline` | End-to-end Pipeline with preprocessor and classifier | Packaging preprocessing and modeling into a single estimator |
| `fit_and_score` | Fit and evaluate the pipeline | Training and scoring a pipeline safely |
| `leakage_demo` | Demonstration of data leakage | Visualizing and proving why scaling before splitting is a bug |

---

## 1. Stratified Splits

```python
>>> from sklearn.model_selection import train_test_split
>>> # Stratified split
>>> train, test = train_test_split(df, test_size=0.2, stratify=df['target'], random_state=42)
```

When splitting a dataset for classification, a simple random split can be dangerous—especially for imbalanced datasets. If only 1% of your rows represent the positive class, a random split might end up with all positive cases in the training set and none in the test set, making evaluation impossible.

Stratification solves this by ensuring that the training and testing splits contain approximately the same percentage of samples of each target class as the complete input dataset.

**In `exercises.py`:** Implement `stratified_split` to split a DataFrame into train and test sets while preserving target class proportions.

**Test it:** `python3 -m unittest test_exercises.TestStratifiedSplit -v`

---

## 2. Column-Specific Preprocessing with ColumnTransformer

```python
>>> from sklearn.compose import ColumnTransformer
>>> from sklearn.pipeline import Pipeline
>>> from sklearn.impute import SimpleImputer
>>> from sklearn.preprocessing import StandardScaler
>>> 
>>> num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
>>> preprocessor = ColumnTransformer([('num', num_pipe, ['age', 'fare'])])
```

Real-world datasets contain a mix of numeric and categorical columns, each requiring different preprocessing steps. Numeric columns need missing values imputed (often with the median) and values scaled. Categorical columns need missing values imputed (often with the most frequent value) and values converted to numbers using one-hot encoding.

`ColumnTransformer` allows you to apply separate transformers or pipelines to specific subsets of columns.

**The Pitfall:** If your test set contains a category that was never seen in the training set, a standard `OneHotEncoder` will raise a `ValueError` and crash your pipeline. To prevent this, always specify `handle_unknown='ignore'` when building your encoder. Additionally, set `sparse_output=False` to ensure the transformer returns a dense numpy array, which is easier to inspect and debug.

**In `exercises.py`:** Implement `build_preprocessor` to construct a `ColumnTransformer` that handles both numeric and categorical columns safely.

**Test it:** `python3 -m unittest test_exercises.TestBuildPreprocessor -v`

---

## 3. Assembling the End-to-End Pipeline

```python
>>> from sklearn.pipeline import Pipeline
>>> pipeline = Pipeline([
...     ('preprocessor', preprocessor),
...     ('classifier', LogisticRegression())
... ])
```

Once you have a preprocessor, you need to chain it with a machine learning model. A scikit-learn `Pipeline` bundles these steps into a single object that exposes the standard estimator interface (`fit`, `predict`, `score`).

By packaging your preprocessor and model together, you treat the entire workflow as a single unit. This makes cross-validation simple and completely eliminates the risk of accidentally applying a transformer to your test data before it has been fit on your training data.

**In `exercises.py`:** Implement `build_pipeline` to combine your preprocessor and a `LogisticRegression` classifier into a single `Pipeline` instance.

**Test it:** `python3 -m unittest test_exercises.TestBuildPipeline -v`

---

## 4. Fitting and Scoring Safely

```python
>>> pipeline.fit(X_train, y_train)
>>> train_score = pipeline.score(X_train, y_train)
>>> test_score = pipeline.score(X_test, y_test)
```

When you call `pipeline.fit(X_train, y_train)`:
1. The pipeline calls `fit_transform` on the preprocessor using `X_train` to learn the preprocessing parameters (like means, medians, and category mappings) and transform the training data.
2. It fits the classifier on the transformed training data.

When you call `pipeline.score(X_test, y_test)`:
1. The pipeline calls `transform` (not `fit_transform`!) on the preprocessor using `X_test`. It uses the *already learned* parameters from the training set.
2. It predicts and scores using the fitted classifier.

This guarantees that no information from `X_test` is used to scale or impute values.

**In `exercises.py`:** Implement `fit_and_score` to fit your pipeline on training data and return both training and testing accuracies.

**Test it:** `python3 -m unittest test_exercises.TestFitAndScore -v`

---

## 5. The Danger of Data Leakage (A Concrete Demo)

To understand why this matters, consider feature scaling. If you scale your numeric features *before* splitting your dataset, the `StandardScaler` calculates the mean and standard deviation of the *entire* dataset.

If your test set contains extreme values or outliers, those outliers will influence the calculated mean and standard deviation. Consequently, the scaled values in your training set will be different than they would have been in a real-world scenario where the test set is completely unseen. This is a subtle but severe form of data leakage.

**In `exercises.py`:** Implement `leakage_demo` to run two scenarios side-by-side: one where scaling is done before splitting (Leaky), and one where scaling is done after splitting (Clean). You will see how this affects the final model evaluation.

**Test it:** `python3 -m unittest test_exercises.TestLeakageDemo -v`

---

## The exercises

| Function | What it teaches |
|---|---|
| `stratified_split` | Splitting data while preserving target class proportions |
| `build_preprocessor` | Constructing a robust, multi-type preprocessing pipeline |
| `build_pipeline` | Packaging preprocessing and estimators together |
| `fit_and_score` | Training and evaluating a pipeline without leakage |
| `leakage_demo` | Demonstrating the mathematical difference between leaky and clean workflows |

## Check your understanding

1. **Why is stratification important when splitting a classification dataset?**
   <details><summary>Answer</summary>Stratification ensures that the proportion of classes in the target column is identical in both the training and testing splits. Without it, rare classes might be underrepresented or entirely absent in one of the splits, leading to highly unreliable evaluations.</details>

2. **What does `handle_unknown='ignore'` do in `OneHotEncoder`?**
   <details><summary>Answer</summary>It ensures that if the test set contains a category that was not present in the training set, the encoder will not raise an error. Instead, it will encode that unknown category as all zeros in the one-hot columns.</details>

3. **Why should we use `sparse_output=False` in `OneHotEncoder`?**
   <details><summary>Answer</summary>By default, <code>OneHotEncoder</code> returns a sparse matrix to save memory. However, passing sparse matrices through complex pipelines can sometimes cause compatibility issues or make debugging harder. Setting <code>sparse_output=False</code> forces it to return a standard dense numpy array.</details>

4. **How does a `Pipeline` prevent data leakage during evaluation?**
   <details><summary>Answer</summary>A <code>Pipeline</code> ensures that all preprocessing steps (like scaling and imputation) are fit <em>only</em> on the training split. When evaluating on the test split, the pipeline uses the parameters (like mean and variance) learned from the training split, ensuring no test information leaks into the training process.</details>

5. **What is the difference between `fit_transform` and `transform` on a scaler, and when should each be used?**
   <details><summary>Answer</summary><code>fit_transform</code> calculates the parameters (mean and standard deviation) and transforms the data; it should <em>only</em> be called on the training set. <code>transform</code> applies those pre-calculated parameters to new data; it should be used on the test or validation set to prevent leakage.</details>

## Interview questions this module answers

- *What is data leakage, and how does it happen during feature scaling?*
- *How do you handle categorical variables in a pipeline when the test set has categories not seen in training?*
- *What is the purpose of `ColumnTransformer` in scikit-learn?*
- *Why is it bad practice to impute missing values on the entire dataset before splitting?*
