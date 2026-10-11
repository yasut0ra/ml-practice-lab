"""発展編の改善・設定選びに、最終テストを混ぜないことを確認する。"""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import ch14_learning_curves as curves
import ch15_imbalanced_classification as imbalance
import ch16_feature_importance as importance
from sklearn.pipeline import Pipeline


class AdvancedEvaluationTests(unittest.TestCase):
    def test_learning_curve_fits_only_subsets_of_training_data(self):
        real_split = curves.train_test_split
        real_fit = curves.DecisionTreeClassifier.fit
        split_data, fitted_sizes = {}, []

        def record_split(*args, **kwargs):
            result = real_split(*args, **kwargs)
            split_data["train"] = set(result[0].index)
            split_data["final"] = set(result[1].index)
            return result

        def record_fit(model, X, y, **kwargs):
            self.assertTrue(set(X.index) <= split_data["train"])
            self.assertFalse(set(X.index) & split_data["final"])
            self.assertEqual(list(X.index), list(y.index))
            fitted_sizes.append(len(X))
            return real_fit(model, X, y, **kwargs)

        with patch.object(curves, "train_test_split", side_effect=record_split), \
             patch.object(curves.DecisionTreeClassifier, "fit", new=record_fit), \
             redirect_stdout(StringIO()):
            curves.main()
        self.assertGreater(len(set(fitted_sizes)), 1)
        self.assertLess(max(fitted_sizes), len(split_data["train"]))

    def test_threshold_selection_uses_validation_and_final_check_uses_one_threshold(self):
        real_split = imbalance.split_inspection_data
        real_fit = Pipeline.fit
        real_table = imbalance.threshold_table
        real_choose = imbalance.choose_threshold
        real_metrics = imbalance.metrics_at_threshold
        split_data, selected, final_checks = {}, {}, []

        def record_split():
            result = real_split()
            for name, frame in zip(["train", "valid", "final"], result[:3]):
                split_data[name] = set(frame.index)
            self.assertFalse(split_data["train"] & split_data["valid"])
            self.assertFalse(split_data["train"] & split_data["final"])
            self.assertFalse(split_data["valid"] & split_data["final"])
            return result

        def record_fit(model, X, y, **kwargs):
            self.assertEqual(set(X.index), split_data["train"])
            self.assertEqual(list(X.index), list(y.index))
            return real_fit(model, X, y, **kwargs)

        def record_table(y, probabilities, thresholds=None):
            self.assertEqual(set(y.index), split_data["valid"])
            return real_table(y, probabilities, thresholds)

        def record_choose(table):
            selected["threshold"] = real_choose(table)
            return selected["threshold"]

        def record_metrics(y, probabilities, threshold):
            if set(y.index) == split_data["final"]:
                self.assertEqual(threshold, selected["threshold"])
                final_checks.append(threshold)
            else:
                self.assertEqual(set(y.index), split_data["valid"])
            return real_metrics(y, probabilities, threshold)

        with patch.object(imbalance, "split_inspection_data", side_effect=record_split), \
             patch.object(Pipeline, "fit", new=record_fit), \
             patch.object(imbalance, "threshold_table", side_effect=record_table), \
             patch.object(imbalance, "choose_threshold", side_effect=record_choose), \
             patch.object(imbalance, "metrics_at_threshold", side_effect=record_metrics), \
             redirect_stdout(StringIO()):
            imbalance.main()
        self.assertEqual(len(final_checks), 1)

    def test_feature_inspection_excludes_training_and_reserved_final_test(self):
        real_split = importance.split_rooms
        real_fit = Pipeline.fit
        real_inspect = importance.permutation_importance
        split_data, inspections = {}, []

        def record_split():
            result = real_split()
            for name, frame in zip(["train", "inspect", "final"], result[:3]):
                split_data[name] = set(frame.index)
            self.assertFalse(split_data["train"] & split_data["inspect"])
            self.assertFalse(split_data["inspect"] & split_data["final"])
            return result

        def record_fit(model, X, y, **kwargs):
            self.assertEqual(set(X.index), split_data["train"])
            self.assertFalse(set(X.index) & split_data["final"])
            self.assertEqual(list(X.index), list(y.index))
            return real_fit(model, X, y, **kwargs)

        def record_inspect(model, X, y, **kwargs):
            self.assertEqual(set(X.index), split_data["inspect"])
            self.assertFalse(set(X.index) & split_data["final"])
            self.assertEqual(list(X.index), list(y.index))
            inspections.append(set(X.index))
            return real_inspect(model, X, y, **kwargs)

        with patch.object(importance, "split_rooms", side_effect=record_split), \
             patch.object(Pipeline, "fit", new=record_fit), \
             patch.object(importance, "permutation_importance", side_effect=record_inspect), \
             redirect_stdout(StringIO()):
            importance.main()
        self.assertEqual(len(inspections), 1)


if __name__ == "__main__":
    unittest.main()
