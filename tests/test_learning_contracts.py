"""教材の評価手順と、壊れた教材を見落とさない仕組みを確認する。"""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import ch10_model_comparison as comparison
import ch11_parameter_search as parameter_search
import ch13_csv_regression as csv_regression
from check_materials import check_document
import run_all


class EvaluationTests(unittest.TestCase):
    def test_comparison_excludes_final_test_and_reuses_folds(self):
        real_split = comparison.train_test_split
        real_cv = comparison.cross_validate
        split_indices = {}
        all_folds = []

        def record_split(*args, **kwargs):
            result = real_split(*args, **kwargs)
            split_indices["training"] = set(result[0].index)
            split_indices["test"] = set(result[1].index)
            return result

        def record_cv(model, X, y, **kwargs):
            self.assertEqual(set(X.index), split_indices["training"])
            self.assertFalse(set(X.index) & split_indices["test"])
            self.assertEqual(list(X.index), list(y.index))
            folds = [(tuple(train), tuple(valid)) for train, valid in kwargs["cv"].split(X, y)]
            all_folds.append(folds)
            return real_cv(model, X, y, **kwargs)

        with patch.object(comparison, "train_test_split", side_effect=record_split), \
             patch.object(comparison, "cross_validate", side_effect=record_cv), \
             redirect_stdout(StringIO()):
            comparison.main()
        self.assertIn("Baseline", comparison.build_models())
        self.assertEqual(len(all_folds), len(comparison.build_models()))
        self.assertTrue(all(folds == all_folds[0] for folds in all_folds))

    def test_parameter_search_fits_preprocessing_inside_each_fold(self):
        real_split = parameter_search.train_test_split
        real_scale_fit = parameter_search.StandardScaler.fit
        split_data = {}
        scale_fits = []

        def record_split(*args, **kwargs):
            result = real_split(*args, **kwargs)
            split_data["X_train"], split_data["X_test"] = result[:2]
            split_data["y_train"] = result[2]
            return result

        def record_scale_fit(scaler, X, y=None, **kwargs):
            indices = set(X.index)
            self.assertTrue(indices <= set(split_data["X_train"].index))
            self.assertFalse(indices & set(split_data["X_test"].index))
            scale_fits.append(frozenset(indices))
            return real_scale_fit(scaler, X, y, **kwargs)

        with patch.object(parameter_search, "train_test_split", side_effect=record_split), \
             patch.object(parameter_search.StandardScaler, "fit", new=record_scale_fit), \
             redirect_stdout(StringIO()):
            parameter_search.main()

        search = parameter_search.build_search()
        candidate_count = len(search.param_grid["clf__C"]) * len(search.param_grid["clf__kernel"])
        expected_folds = {
            frozenset(split_data["X_train"].iloc[train].index)
            for train, _ in search.cv.split(split_data["X_train"], split_data["y_train"])
        }
        self.assertEqual(set(scale_fits[:-1]), expected_folds)
        self.assertEqual(len(scale_fits), candidate_count * search.cv.n_splits + 1)
        self.assertEqual(scale_fits[-1], frozenset(split_data["X_train"].index))

    def test_csv_preprocessing_learns_from_training_rooms_only(self):
        real_split = csv_regression.train_test_split
        real_impute_fit = csv_regression.SimpleImputer.fit
        split_data = {}
        impute_fits = []

        def record_split(*args, **kwargs):
            result = real_split(*args, **kwargs)
            split_data["train"], split_data["test"] = set(result[0].index), set(result[1].index)
            self.assertNotIn(csv_regression.TARGET, result[0].columns)
            self.assertNotIn("room_id", result[0].columns)
            return result

        def record_impute_fit(imputer, X, y=None, **kwargs):
            self.assertEqual(set(X.index), split_data["train"])
            self.assertFalse(set(X.index) & split_data["test"])
            impute_fits.append(list(X.columns))
            return real_impute_fit(imputer, X, y, **kwargs)

        with TemporaryDirectory() as directory, \
             patch.object(csv_regression, "ROOT", Path(directory)), \
             patch.object(csv_regression, "train_test_split", side_effect=record_split), \
             patch.object(csv_regression.SimpleImputer, "fit", new=record_impute_fit), \
             redirect_stdout(StringIO()):
            csv_regression.main()
        self.assertEqual(impute_fits, [csv_regression.NUMERIC, ["zone"]])

    def test_csv_allows_missing_features_but_rejects_missing_answers(self):
        data = csv_regression.load_rent_data()
        self.assertTrue(data[csv_regression.FEATURES].isna().any().any())
        with TemporaryDirectory() as directory:
            path = Path(directory) / "missing_answer.csv"
            data.loc[0, csv_regression.TARGET] = float("nan")
            data.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "答えの家賃に空欄"):
                csv_regression.load_rent_data(path)


class MaterialChecksTests(unittest.TestCase):
    def test_missing_file_and_missing_heading_are_reported(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "target.md").write_text("# 見出し\n", encoding="utf-8")
            document = root / "lesson.md"
            document.write_text("[欠けたファイル](missing.md)\n[欠けた見出し](target.md#missing)\n", encoding="utf-8")
            self.assertEqual(len(check_document(document)), 2)

    def test_valid_unicode_and_duplicate_headings_and_code_examples(self):
        with TemporaryDirectory() as directory:
            document = Path(directory) / "lesson.md"
            document.write_text(
                "# 日本語の見出し\n# 日本語の見出し\n"
                "[最初](#日本語の見出し)\n[二番目](#日本語の見出し-1)\n"
                "```text\n[入力例](not-a-file.md)\n```\n", encoding="utf-8"
            )
            self.assertEqual(check_document(document), [])

    def test_answer_runner_stops_at_first_failure(self):
        failure = subprocess.CalledProcessError(1, ["python", "answer.py"])
        with patch.object(sys, "argv", ["run_all.py", "--answers"]), \
             patch.object(run_all.subprocess, "run", side_effect=failure) as execute, \
             redirect_stdout(StringIO()):
            with self.assertRaises(subprocess.CalledProcessError):
                run_all.main()
        self.assertEqual(execute.call_count, 1)


if __name__ == "__main__":
    unittest.main()
