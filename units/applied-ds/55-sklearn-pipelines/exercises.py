"""55 - Features and pipelines with scikit-learn

Implement each function so the tests in test_exercises.py pass. Read
README.md first; every exercise below is covered there.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


def stratified_split(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42):
    """Split the DataFrame into train and test sets, stratifying on target_col.

    Args:
        df: The input pandas DataFrame.
        target_col: The name of the column to stratify on.
        test_size: The proportion of the dataset to include in the test split.
        random_state: Controls the shuffling applied to the data before applying the split.

    Returns:
        A tuple of (df_train, df_test).
    """
    raise NotImplementedError("your code here")


def build_preprocessor(numeric_cols: list, categorical_cols: list) -> ColumnTransformer:
    """Build a ColumnTransformer that preprocesses numeric and categorical columns.

    For numeric columns:
    - Impute missing values with the median.
    - Scale values using StandardScaler.

    For categorical columns:
    - Impute missing values with the most frequent value.
    - One-hot encode values, ignoring unknown categories during transform,
      and returning dense arrays (sparse_output=False).

    Args:
        numeric_cols: List of numeric column names.
        categorical_cols: List of categorical column names.

    Returns:
        A ColumnTransformer instance.
    """
    raise NotImplementedError("your code here")


def build_pipeline(preprocessor: ColumnTransformer, C: float = 1.0, random_state: int = 42) -> Pipeline:
    """Build a Pipeline containing the preprocessor and a LogisticRegression classifier.

    Args:
        preprocessor: A ColumnTransformer instance.
        C: Inverse of regularization strength for LogisticRegression.
        random_state: Random state for LogisticRegression.

    Returns:
        A Pipeline instance with steps named 'preprocessor' and 'classifier'.
    """
    raise NotImplementedError("your code here")


def fit_and_score(pipeline: Pipeline, df_train: pd.DataFrame, df_test: pd.DataFrame, target_col: str) -> tuple:
    """Fit the pipeline on df_train and score it on both df_train and df_test.

    Args:
        pipeline: A scikit-learn Pipeline instance.
        df_train: Training DataFrame.
        df_test: Testing DataFrame.
        target_col: Name of the target column.

    Returns:
        A tuple of (train_accuracy, test_accuracy).
    """
    raise NotImplementedError("your code here")


def leakage_demo(df: pd.DataFrame, target_col: str, numeric_cols: list) -> tuple:
    """Demonstrate data leakage by comparing two scaling workflows.

    Scenario A (Leaky):
    1. Copy df.
    2. Fit and transform StandardScaler on df[numeric_cols], replacing those columns in df.
    3. Split the scaled df using stratified_split with test_size=0.2, random_state=42.
    4. Fit a LogisticRegression(random_state=42) on the train features (numeric_cols only)
       and score on the test features.

    Scenario B (Clean):
    1. Split the original df using stratified_split with test_size=0.2, random_state=42.
    2. Fit a StandardScaler on the train features (numeric_cols only) and transform both
       train and test features.
    3. Fit a LogisticRegression(random_state=42) on the scaled train features and score
       on the scaled test features.

    Args:
        df: The input pandas DataFrame.
        target_col: Name of the target column.
        numeric_cols: List of numeric column names to use as features.

    Returns:
        A tuple of (leaky_test_score, clean_test_score).
    """
    raise NotImplementedError("your code here")
