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
    """Split the DataFrame into train and test sets, stratifying on target_col."""
    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_col],
        random_state=random_state
    )
    return df_train, df_test


def build_preprocessor(numeric_cols: list, categorical_cols: list) -> ColumnTransformer:
    """Build a ColumnTransformer that preprocesses numeric and categorical columns."""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    return preprocessor


def build_pipeline(preprocessor: ColumnTransformer, C: float = 1.0, random_state: int = 42) -> Pipeline:
    """Build a Pipeline containing the preprocessor and a LogisticRegression classifier."""
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(C=C, random_state=random_state, max_iter=1000))
    ])
    return pipeline


def fit_and_score(pipeline: Pipeline, df_train: pd.DataFrame, df_test: pd.DataFrame, target_col: str) -> tuple:
    """Fit the pipeline on df_train and score it on both df_train and df_test."""
    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]

    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]

    pipeline.fit(X_train, y_train)

    train_acc = pipeline.score(X_train, y_train)
    test_acc = pipeline.score(X_test, y_test)

    return train_acc, test_acc


def leakage_demo(df: pd.DataFrame, target_col: str, numeric_cols: list) -> tuple:
    """Demonstrate data leakage by comparing two scaling workflows."""
    # Scenario A: Leaky
    df_leaky = df.copy()
    scaler_leaky = StandardScaler()
    df_leaky[numeric_cols] = scaler_leaky.fit_transform(df_leaky[numeric_cols])

    df_train_leaky, df_test_leaky = stratified_split(df_leaky, target_col, test_size=0.2, random_state=42)

    X_train_leaky = df_train_leaky[numeric_cols]
    y_train_leaky = df_train_leaky[target_col]
    X_test_leaky = df_test_leaky[numeric_cols]
    y_test_leaky = df_test_leaky[target_col]

    model_leaky = LogisticRegression(random_state=42)
    model_leaky.fit(X_train_leaky, y_train_leaky)
    leaky_score = model_leaky.score(X_test_leaky, y_test_leaky)

    # Scenario B: Clean
    df_train_clean, df_test_clean = stratified_split(df, target_col, test_size=0.2, random_state=42)

    X_train_clean = df_train_clean[numeric_cols].copy()
    y_train_clean = df_train_clean[target_col]
    X_test_clean = df_test_clean[numeric_cols].copy()
    y_test_clean = df_test_clean[target_col]

    scaler_clean = StandardScaler()
    X_train_scaled = scaler_clean.fit_transform(X_train_clean)
    X_test_scaled = scaler_clean.transform(X_test_clean)

    model_clean = LogisticRegression(random_state=42)
    model_clean.fit(X_train_scaled, y_train_clean)
    clean_score = model_clean.score(X_test_scaled, y_test_clean)

    return leaky_score, clean_score
