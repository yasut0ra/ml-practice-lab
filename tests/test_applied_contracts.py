"""保存した前処理、文章の辞書、時間順の評価が教材の条件を守ることを確かめる。"""

from contextlib import redirect_stdout
from functools import wraps
from io import StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "answers"))
import ch17_model_persistence as persistence
import ch18_text_classification as text
import ch18_answer as text_answer
import ch19_time_series as series
import ch19_answer as series_answer
import predict_saved_rent


class AppliedLearningTests(unittest.TestCase):
    def test_saved_pipeline_keeps_preprocessing_without_refitting(self):
        model, _, _ = persistence.train_rent_model()
        rooms = persistence.example_rooms()
        expected = model.predict(rooms)
        with TemporaryDirectory() as folder:
            path = persistence.save_model(model, Path(folder))
            metadata = json.loads((Path(folder) / "rent_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["features"], list(rooms.columns))
            self.assertEqual(metadata["target_unit"], "万円/月")
            self.assertIn("scikit-learn", metadata["versions"])
            with patch.object(Pipeline, "fit", side_effect=AssertionError("予測時に学び直してはいけない")), \
                 redirect_stdout(StringIO()):
                restored = joblib.load(path)
                np.testing.assert_allclose(restored.predict(rooms), expected)
                predict_saved_rent.main(path)
            self.assertTrue(np.isfinite(expected).all())
            self.assertTrue(rooms["walk_min"].isna().any())
            self.assertNotIn("coastal", restored.named_steps["prepare"].named_transformers_["cat"].named_steps["onehot"].categories_[0])

    def test_text_cv_builds_dictionary_only_on_each_training_fold(self):
        X_train, X_test, y_train, _ = text.split_messages()
        real_transform, real_predict = TfidfVectorizer.fit_transform, Pipeline.predict
        sizes, final_predictions = [], []

        def record_dictionary(vectorizer, raw_documents, y=None):
            self.assertTrue(set(raw_documents.index) <= set(X_train.index))
            self.assertFalse(set(raw_documents.index) & set(X_test.index))
            sizes.append(len(raw_documents))
            transformed = real_transform(vectorizer, raw_documents, y)
            learned = set(vectorizer.vocabulary_)
            expected = {chunk for document in raw_documents for chunk in vectorizer.build_analyzer()(document)}
            self.assertEqual(learned, expected)
            return transformed

        @wraps(real_predict)
        def record_predict(model, X, **kwargs):
            if X.equals(X_test):
                final_predictions.append(1)
            return real_predict(model, X, **kwargs)

        with TemporaryDirectory() as folder, \
             patch.object(text_answer, "ROOT", Path(folder)), \
             patch.object(TfidfVectorizer, "fit_transform", new=record_dictionary), \
             patch.object(Pipeline, "predict", new=record_predict), \
             redirect_stdout(StringIO()):
            text_answer.main()
        self.assertEqual(sizes, [24] * 6 + [len(y_train)])
        self.assertEqual(len(final_predictions), 1)

    def test_lag_features_do_not_use_today_or_future_sales(self):
        sales = series.make_sales()
        original = series.make_features(sales)
        cutoff = sales.index[100]
        changed = sales.copy()
        changed.loc[cutoff:] += 10000
        recomputed = series.make_features(changed)
        pd.testing.assert_frame_equal(original.loc[:cutoff], recomputed.loc[:cutoff])
        self.assertNotEqual(original.loc[sales.index[101], "lag_1"], recomputed.loc[sales.index[101], "lag_1"])
        pd.testing.assert_series_equal(original["lag_1"], sales.shift(1).loc[original.index].astype(float), check_names=False)
        pd.testing.assert_series_equal(original["lag_7"], sales.shift(7).loc[original.index].astype(float), check_names=False)
        self.assertNotIn("sales", original.columns)

    def test_temporal_cv_preprocessing_and_final_prediction_keep_time_order(self):
        X_train, X_test, _, _ = series.split_sales()
        folds = series.chronological_folds(X_train)
        for train, valid in folds:
            self.assertLess(X_train.index[train].max(), X_train.index[valid].min())
            self.assertLess(X_train.index[valid].max(), X_test.index.min())
        self.assertTrue((X_train.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all())
        expected_fits = [list(X_train.iloc[train].index) for train, _ in folds] * 3 + [list(X_train.index)]
        real_fit, real_predict = StandardScaler.fit, Pipeline.predict
        fits, final_predictions = [], []

        def record_scale(scaler, X, y=None, **kwargs):
            self.assertLess(X.index.max(), X_test.index.min())
            fits.append(list(X.index))
            return real_fit(scaler, X, y, **kwargs)

        @wraps(real_predict)
        def record_predict(model, X, **kwargs):
            if X.equals(X_test):
                final_predictions.append(1)
            return real_predict(model, X, **kwargs)

        with TemporaryDirectory() as folder, \
             patch.object(series_answer, "ROOT", Path(folder)), \
             patch.object(StandardScaler, "fit", new=record_scale), \
             patch.object(Pipeline, "predict", new=record_predict), \
             redirect_stdout(StringIO()):
            series_answer.main()
        self.assertEqual(fits, expected_fits)
        self.assertEqual(len(final_predictions), 1)


if __name__ == "__main__":
    unittest.main()
