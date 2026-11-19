"""Tests for 55 - Features and pipelines with scikit-learn."""

import importlib
import os
import unittest
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestStratifiedSplit(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'feature': range(100),
            'target': [0] * 80 + [1] * 20
        })

    def test_split_shapes(self):
        df_train, df_test = m.stratified_split(self.df, 'target', test_size=0.2, random_state=42)
        self.assertEqual(len(df_train), 80)
        self.assertEqual(len(df_test), 20)

    def test_stratification_ratio(self):
        df_train, df_test = m.stratified_split(self.df, 'target', test_size=0.2, random_state=42)
        train_ratio = df_train['target'].mean()
        test_ratio = df_test['target'].mean()
        self.assertAlmostEqual(train_ratio, 0.2, places=2)
        self.assertAlmostEqual(test_ratio, 0.2, places=2)

    def test_determinism(self):
        df_train1, df_test1 = m.stratified_split(self.df, 'target', test_size=0.2, random_state=42)
        df_train2, df_test2 = m.stratified_split(self.df, 'target', test_size=0.2, random_state=42)
        pd.testing.assert_frame_equal(df_train1, df_train2)
        pd.testing.assert_frame_equal(df_test1, df_test2)

    def test_invalid_target_col(self):
        with self.assertRaises(KeyError):
            m.stratified_split(self.df, 'non_existent_column')

    def test_custom_test_size(self):
        df_train, df_test = m.stratified_split(self.df, 'target', test_size=0.3, random_state=42)
        self.assertEqual(len(df_train), 70)
        self.assertEqual(len(df_test), 30)

    def test_preserves_index(self):
        df_train, df_test = m.stratified_split(self.df, 'target', test_size=0.2, random_state=42)
        combined_indices = sorted(list(df_train.index) + list(df_test.index))
        self.assertEqual(combined_indices, list(range(100)))


class TestBuildPreprocessor(unittest.TestCase):
    def setUp(self):
        self.numeric_cols = ['age', 'income']
        self.categorical_cols = ['city', 'gender']
        self.preprocessor = m.build_preprocessor(self.numeric_cols, self.categorical_cols)

    def test_is_column_transformer(self):
        self.assertIsInstance(self.preprocessor, ColumnTransformer)

    def test_transformer_steps(self):
        transformers = {name: trans for name, trans, cols in self.preprocessor.transformers}
        self.assertIn('num', transformers)
        self.assertIn('cat', transformers)
        
        num_steps = [step[0] for step in transformers['num'].steps]
        self.assertEqual(num_steps, ['imputer', 'scaler'])
        
        cat_steps = [step[0] for step in transformers['cat'].steps]
        self.assertEqual(cat_steps, ['imputer', 'onehot'])

    def test_one_hot_encoder_settings(self):
        transformers = {name: trans for name, trans, cols in self.preprocessor.transformers}
        onehot = transformers['cat'].named_steps['onehot']
        self.assertEqual(onehot.handle_unknown, 'ignore')
        self.assertFalse(onehot.sparse_output)

    def test_imputer_strategies(self):
        transformers = {name: trans for name, trans, cols in self.preprocessor.transformers}
        num_imputer = transformers['num'].named_steps['imputer']
        cat_imputer = transformers['cat'].named_steps['imputer']
        self.assertEqual(num_imputer.strategy, 'median')
        self.assertEqual(cat_imputer.strategy, 'most_frequent')

    def test_numeric_only(self):
        preprocessor = m.build_preprocessor(self.numeric_cols, [])
        transformers = {name: trans for name, trans, cols in preprocessor.transformers}
        self.assertEqual(transformers['cat'].steps[1][1].sparse_output, False)

    def test_categorical_only(self):
        preprocessor = m.build_preprocessor([], self.categorical_cols)
        transformers = {name: trans for name, trans, cols in preprocessor.transformers}
        self.assertEqual(transformers['num'].steps[1][1].with_mean, True)

    def test_empty_columns(self):
        preprocessor = m.build_preprocessor([], [])
        self.assertIsInstance(preprocessor, ColumnTransformer)


class TestBuildPipeline(unittest.TestCase):
    def setUp(self):
        self.preprocessor = m.build_preprocessor(['num'], ['cat'])
        self.pipeline = m.build_pipeline(self.preprocessor, C=0.5, random_state=100)

    def test_is_pipeline(self):
        self.assertIsInstance(self.pipeline, Pipeline)

    def test_pipeline_steps(self):
        steps = [step[0] for step in self.pipeline.steps]
        self.assertEqual(steps, ['preprocessor', 'classifier'])

    def test_classifier_hyperparameters(self):
        classifier = self.pipeline.named_steps['classifier']
        self.assertEqual(classifier.C, 0.5)
        self.assertEqual(classifier.random_state, 100)

    def test_custom_C_value(self):
        pipeline = m.build_pipeline(self.preprocessor, C=10.0, random_state=42)
        classifier = pipeline.named_steps['classifier']
        self.assertEqual(classifier.C, 10.0)

    def test_pipeline_fit_predict(self):
        df = pd.DataFrame({
            'num': [1.0, 2.0, 3.0, 4.0],
            'cat': ['A', 'B', 'A', 'B'],
            'target': [0, 1, 0, 1]
        })
        X = df[['num', 'cat']]
        y = df['target']
        self.pipeline.fit(X, y)
        preds = self.pipeline.predict(X)
        self.assertEqual(len(preds), 4)


class TestFitAndScore(unittest.TestCase):
    def setUp(self):
        self.df_train = pd.DataFrame({
            'num': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            'cat': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
            'target': [0, 0, 0, 0, 1, 1, 1, 1]
        })
        self.df_test = pd.DataFrame({
            'num': [2.5, 6.5],
            'cat': ['A', 'C'],
            'target': [0, 1]
        })
        self.preprocessor = m.build_preprocessor(['num'], ['cat'])
        self.pipeline = m.build_pipeline(self.preprocessor, random_state=42)

    def test_fit_and_score_returns_floats(self):
        train_acc, test_acc = m.fit_and_score(self.pipeline, self.df_train, self.df_test, 'target')
        self.assertIsInstance(train_acc, float)
        self.assertIsInstance(test_acc, float)
        self.assertTrue(0.0 <= train_acc <= 1.0)
        self.assertTrue(0.0 <= test_acc <= 1.0)

    def test_handles_unknown_categories_safely(self):
        try:
            m.fit_and_score(self.pipeline, self.df_train, self.df_test, 'target')
        except ValueError as e:
            self.fail(f"fit_and_score raised ValueError: {e}. Did you forget handle_unknown='ignore'?")

    def test_accuracy_check(self):
        train_acc, test_acc = m.fit_and_score(self.pipeline, self.df_train, self.df_test, 'target')
        self.assertEqual(train_acc, 1.0)
        self.assertEqual(test_acc, 1.0)

    def test_mutates_pipeline(self):
        classifier = self.pipeline.named_steps['classifier']
        self.assertFalse(hasattr(classifier, 'classes_'))
        m.fit_and_score(self.pipeline, self.df_train, self.df_test, 'target')
        self.assertTrue(hasattr(classifier, 'classes_'))

    def test_missing_target_raises_error(self):
        df_test_missing = self.df_test.drop(columns=['target'])
        with self.assertRaises(KeyError):
            m.fit_and_score(self.pipeline, self.df_train, df_test_missing, 'target')


class TestLeakageDemo(unittest.TestCase):
    def test_leakage_demo_scores(self):
        dummy_df = pd.DataFrame({
            'num': np.linspace(1, 10, 20),
            'target': [0] * 10 + [1] * 10
        })
        train_df, test_df = m.stratified_split(dummy_df, 'target', test_size=0.2, random_state=42)
        test_indices = test_df.index.tolist()
        
        num_values = np.zeros(20)
        for i in range(20):
            if i < 10:
                num_values[i] = 1.0 + (i * 0.05)
            else:
                num_values[i] = 5.0 + ((i - 10) * 0.05)
        
        test_class_1_indices = [idx for idx in test_indices if idx >= 10]
        outlier_idx = test_class_1_indices[0]
        num_values[outlier_idx] = 100000.0
        
        df = pd.DataFrame({
            'num': num_values,
            'target': [0] * 10 + [1] * 10
        })
        
        leaky_score, clean_score = m.leakage_demo(df, 'target', ['num'])
        self.assertIsInstance(leaky_score, float)
        self.assertIsInstance(clean_score, float)
        self.assertNotEqual(leaky_score, clean_score)

    def test_leakage_demo_returns_tuple_of_floats(self):
        df = pd.DataFrame({
            'num': [1.0, 2.0, 3.0, 4.0, 5.0] * 2,
            'target': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        })
        res = m.leakage_demo(df, 'target', ['num'])
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        self.assertIsInstance(res[0], float)
        self.assertIsInstance(res[1], float)

    def test_leakage_demo_invalid_col(self):
        df = pd.DataFrame({
            'num': [1.0, 2.0, 3.0, 4.0, 5.0] * 2,
            'target': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        })
        with self.assertRaises(KeyError):
            m.leakage_demo(df, 'target', ['non_existent'])


if __name__ == "__main__":
    unittest.main(verbosity=2)
