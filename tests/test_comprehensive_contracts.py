"""追加章の分割・校正・予測区間・監視が未使用データを守るか確認する。"""

from contextlib import redirect_stdout
from functools import wraps
from io import StringIO
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "answers"))
import ch20_feature_engineering as features
import ch20_answer as features_answer
import ch21_group_validation as groups
import ch22_gradient_boosting as boosting
import ch22_answer as boosting_answer
import ch23_neural_network as neural
import ch23_answer as neural_answer
import ch24_anomaly_detection as anomaly
import ch24_answer as anomaly_answer
import ch25_probability_calibration as probability
import ch25_answer as probability_answer
import ch26_prediction_intervals as intervals
import ch26_answer as intervals_answer
import ch27_monitoring as monitoring


class ComprehensiveLearningTests(unittest.TestCase):
    def run_cv_answer(self, module, transformer, method, X_train, X_test, folds, candidates):
        real_fit = getattr(transformer, method)
        real_predict = Pipeline.predict
        fits, final_predictions = [], []

        def record_fit(estimator, X, y=None, **kwargs):
            # 第20章の最初の3行は列の形を見るだけの例。
            if y is not None:
                self.assertTrue(set(X.index) <= set(X_train.index))
                self.assertFalse(set(X.index) & set(X_test.index))
                fits.append(list(X.index))
            return real_fit(estimator, X, y, **kwargs)

        @wraps(real_predict)
        def record_predict(model, X, **kwargs):
            if X.equals(X_test):
                final_predictions.append(1)
            return real_predict(model, X, **kwargs)

        expected = [list(X_train.iloc[a].index) for a, _ in folds] * candidates + [list(X_train.index)]
        with TemporaryDirectory() as folder, patch.object(module, "ROOT", Path(folder)), \
             patch.object(transformer, method, new=record_fit), \
             patch.object(Pipeline, "predict", new=record_predict), redirect_stdout(StringIO()):
            module.main()
        self.assertEqual(fits, expected)
        self.assertEqual(len(final_predictions), 1)

    def test_polynomial_candidates_learn_only_each_training_fold(self):
        X_train, X_test, _, _ = features.split_energy()
        folds = list(KFold(5, shuffle=True, random_state=42).split(X_train))
        self.run_cv_answer(features_answer, PolynomialFeatures, "fit", X_train, X_test, folds, 4)

    def test_boosting_preprocessing_is_refit_inside_cv(self):
        X_train, X_test, _, _ = boosting.split_rooms()
        folds = list(KFold(5, shuffle=True, random_state=42).split(X_train))
        self.run_cv_answer(boosting_answer, ColumnTransformer, "fit_transform", X_train, X_test, folds, 3)

    def test_neural_width_selection_keeps_final_digits_outside_scaling(self):
        X_train, X_test, y_train, _ = neural.split_digits()
        folds = list(StratifiedKFold(3, shuffle=True, random_state=42).split(X_train, y_train))
        self.run_cv_answer(neural_answer, StandardScaler, "fit", X_train, X_test, folds, 2)

    def test_group_cv_and_final_test_use_new_people_without_id_feature(self):
        train, final = groups.split_students()
        self.assertEqual((train.student_id.nunique(), final.student_id.nunique()), (48, 12))
        self.assertFalse(set(train.student_id) & set(final.student_id))
        folds = groups.group_folds(train)
        for a, b in folds:
            self.assertFalse(set(train.iloc[a].student_id) & set(train.iloc[b].student_id))
        expected = [list(train.iloc[a].index) for a, _ in folds]
        fits, final_predictions = [], []

        def wrap_fit(real_fit):
            @wraps(real_fit)
            def record(model, X, y, **kwargs):
                self.assertEqual(list(X.columns), groups.FEATURES)
                self.assertFalse(set(X.index) & set(final.index))
                fits.append(list(X.index))
                return real_fit(model, X, y, **kwargs)
            return record

        def wrap_predict(real_predict):
            @wraps(real_predict)
            def record(model, X, **kwargs):
                if X.equals(final[groups.FEATURES]):
                    final_predictions.append(1)
                return real_predict(model, X, **kwargs)
            return record

        with patch.object(RandomForestRegressor, "fit", new=wrap_fit(RandomForestRegressor.fit)), \
             patch.object(DummyRegressor, "fit", new=wrap_fit(DummyRegressor.fit)), \
             patch.object(RandomForestRegressor, "predict", new=wrap_predict(RandomForestRegressor.predict)), \
             patch.object(DummyRegressor, "predict", new=wrap_predict(DummyRegressor.predict)), \
             redirect_stdout(StringIO()):
            groups.main()
        self.assertEqual(fits, expected * 2 + [list(train.index)])
        self.assertEqual(len(final_predictions), 1)

    def test_anomaly_fit_uses_unlabeled_train_and_final_is_scored_once(self):
        X_train, X_valid, X_test, _, _ = anomaly.split_inspections()
        real_fit, real_predict = IsolationForest.fit, IsolationForest.predict
        fits, predictions = [], []

        @wraps(real_fit)
        def record_fit(model, X, y=None, **kwargs):
            pd.testing.assert_frame_equal(X, X_train)
            self.assertIsNone(y)
            fits.append(model.contamination)
            return real_fit(model, X, y, **kwargs)

        @wraps(real_predict)
        def record_predict(model, X, **kwargs):
            self.assertTrue(X.equals(X_valid) or X.equals(X_test))
            predictions.append((model.contamination, X.equals(X_test)))
            return real_predict(model, X, **kwargs)

        with TemporaryDirectory() as folder, patch.object(anomaly_answer, "ROOT", Path(folder)), \
             patch.object(IsolationForest, "fit", new=record_fit), \
             patch.object(IsolationForest, "predict", new=record_predict), redirect_stdout(StringIO()):
            anomaly_answer.main()
            table = pd.read_csv(Path(folder) / "outputs" / "anomaly_candidates.csv")
        self.assertEqual(fits, [0.02, 0.05, 0.1, 0.15])
        self.assertEqual([value for value, final in predictions if not final], fits)
        self.assertEqual([value for value, final in predictions if final], [anomaly.choose_candidate(table)])

    def test_probability_calibration_keeps_outer_validation_and_test_out_of_fit(self):
        X_train, X_valid, X_test, y_train, y_valid, y_test = probability.split_events()
        folds = list(StratifiedKFold(3, shuffle=True, random_state=42).split(X_train, y_train))
        real_fit, real_metrics = StandardScaler.fit, probability_answer.probability_metrics
        fits, scored = [], []

        def record_scale(scaler, X, y=None, **kwargs):
            self.assertTrue(set(X.index) <= set(X_train.index))
            self.assertFalse(set(X.index) & (set(X_valid.index) | set(X_test.index)))
            fits.append(list(X.index))
            return real_fit(scaler, X, y, **kwargs)

        def record_metrics(y, probabilities):
            self.assertTrue(y.equals(y_valid) or y.equals(y_test))
            scored.append(y.equals(y_test))
            return real_metrics(y, probabilities)

        with TemporaryDirectory() as folder, patch.object(probability_answer, "ROOT", Path(folder)), \
             patch.object(StandardScaler, "fit", new=record_scale), \
             patch.object(probability, "probability_metrics", new=record_metrics), \
             patch.object(probability_answer, "probability_metrics", new=record_metrics), redirect_stdout(StringIO()):
            probability_answer.main()
            bins = pd.read_csv(Path(folder) / "outputs" / "probability_bins.csv")
        inner = [list(X_train.iloc[a].index) for a, _ in folds] + [list(X_train.index)]
        self.assertEqual(fits, [list(X_train.index)] + inner * 2)
        self.assertEqual(scored, [False] * 3 + [True])
        self.assertEqual(bins.groupby("bins")["count"].sum().tolist(), [200, 200])
        empty_bins = probability.calibration_table([0, 1], [0.1, 0.9], bins=5)
        self.assertEqual(empty_bins["count"].tolist(), [1, 1])

    def test_conformal_rank_is_exact_and_small_samples_can_require_infinite_width(self):
        residuals = np.arange(1, 31)[::-1]
        self.assertEqual(intervals.conformal_radius(residuals, 0.8), 25)
        self.assertEqual(intervals.conformal_radius(residuals, 0.9), 28)
        self.assertTrue(np.isinf(intervals.conformal_radius(residuals, 0.99)))
        for invalid in [[], [np.nan], [np.inf], [-1], [[1, 2]]]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                intervals.conformal_radius(invalid)
        for target in [0, 1, -0.1, float("nan")]:
            with self.subTest(target=target), self.assertRaises(ValueError):
                intervals.conformal_radius([1, 2], target)
        table = intervals.interval_table(pd.Series([8, 12, 13]), [10, 10, 10], 2)
        self.assertEqual(table["covered"].tolist(), [True, True, False])

    def test_interval_training_and_width_ignore_changed_final_answers(self):
        X_train, X_cal, X_test, y_train, y_cal, y_test = intervals.split_rooms()
        self.assertFalse(set(X_train.index) & (set(X_cal.index) | set(X_test.index)))
        self.assertFalse(set(X_cal.index) & set(X_test.index))
        original_model = intervals.build_model().fit(X_train, y_train)
        expected_radius = intervals.conformal_radius(np.abs(y_cal - original_model.predict(X_cal)), 0.9)
        real_fit, real_predict = Pipeline.fit, Pipeline.predict
        fits, predictions = [], []

        def record_fit(model, X, y, **kwargs):
            pd.testing.assert_frame_equal(X, X_train)
            pd.testing.assert_series_equal(y, y_train)
            fits.append(1)
            return real_fit(model, X, y, **kwargs)

        @wraps(real_predict)
        def record_predict(model, X, **kwargs):
            self.assertTrue(X.equals(X_cal) or X.equals(X_test))
            predictions.append("cal" if X.equals(X_cal) else "test")
            return real_predict(model, X, **kwargs)

        changed = (X_train, X_cal, X_test, y_train, y_cal, y_test + 1000)
        with TemporaryDirectory() as folder, patch.object(intervals_answer, "ROOT", Path(folder)), \
             patch.object(intervals_answer, "split_rooms", return_value=changed), \
             patch.object(Pipeline, "fit", new=record_fit), \
             patch.object(Pipeline, "predict", new=record_predict), redirect_stdout(StringIO()):
            intervals_answer.main()
            result = pd.read_csv(Path(folder) / "outputs" / "rent_intervals.csv")
        self.assertEqual(fits, [1])
        self.assertEqual(predictions, ["cal", "test"])
        np.testing.assert_allclose(result["upper_man"] - result["lower_man"], 2 * expected_radius)
        self.assertFalse(result["covered"].any())

    def test_monitoring_handles_missing_and_reordered_partial_labels_without_refitting(self):
        X_train, y_train, batches = monitoring.monitoring_data()
        model = monitoring.build_model().fit(X_train, y_train)
        X, y = batches["relationship_shift"]
        pd.testing.assert_frame_equal(X, batches["input_shift"][0])
        reference_area = batches["reference"][0]["area_sqm"].mean()
        pred = pd.Series(model.predict(X), index=X.index)
        partial = y.iloc[:40].sample(frac=1, random_state=42)
        with patch.object(Pipeline, "fit", side_effect=AssertionError("監視ではモデルを学び直さない")):
            pending = monitoring.summarize_batch(model, X, None, reference_area, "pending")
            observed = monitoring.summarize_batch(model, X, partial, reference_area, "partial")
            complete = monitoring.summarize_batch(model, X, y, reference_area, "complete")
        self.assertTrue(pending["MAE_man"].isna().all())
        self.assertEqual(pending["n_labeled"].tolist(), [0, 0, 0])
        self.assertEqual(pending["error_alert"].tolist(), ["未評価"] * 3)
        self.assertTrue(pending.iloc[0]["input_alert"])
        self.assertEqual(observed.iloc[0]["n_labeled"], 40)
        self.assertEqual(observed.iloc[1:]["n_labeled"].sum(), 40)
        expected = np.abs(partial - pred.reindex(partial.index)).mean()
        self.assertAlmostEqual(observed.iloc[0]["MAE_man"], expected)
        self.assertEqual(complete.iloc[0]["n_labeled"], 120)
        self.assertEqual(complete.iloc[0]["error_alert"], "確認")


if __name__ == "__main__":
    unittest.main()
