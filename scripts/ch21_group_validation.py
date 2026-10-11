from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, cross_val_score

from common import print_header


FEATURES = ["practice_minutes", "quiz_score"]


def make_students() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    minutes = rng.uniform(10, 90, 60)
    quiz = rng.uniform(20, 80, 60)
    personal_offset = rng.normal(0, 12, 60)
    # 一人につき4回。固定のプロフィールと、その人固有のずれを持つ架空の記録。
    return pd.DataFrame({
        "student_id": np.repeat(np.arange(60), 4),
        "practice_minutes": np.repeat(minutes, 4), "quiz_score": np.repeat(quiz, 4),
        "next_score": np.repeat(20 + 0.6 * quiz + 0.03 * minutes + personal_offset, 4) + rng.normal(0, 1, 240),
    })


def split_students():
    data = make_students()
    train, final = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(
        data[FEATURES], data["next_score"], groups=data["student_id"]))
    return data.iloc[train], data.iloc[final]


def build_model():
    return RandomForestRegressor(n_estimators=60, random_state=42)


def group_folds(train):
    return list(GroupKFold(n_splits=3).split(train[FEATURES], train["next_score"], train["student_id"]))


def compare_models(train) -> pd.DataFrame:
    folds = group_folds(train)
    rows = []
    for name, model in [("mean", DummyRegressor(strategy="mean")), ("forest", build_model())]:
        scores = cross_val_score(model, train[FEATURES], train["next_score"], cv=folds,
                                 scoring="neg_mean_absolute_error", error_score="raise")
        rows.append({"model": name, "group_CV_MAE": -scores.mean()})
    return pd.DataFrame(rows)


def split_comparison(train) -> pd.DataFrame:
    rows = []
    folds_by_name = {
        "行を混ぜる": list(KFold(n_splits=3, shuffle=True, random_state=42).split(train)),
        "人ごとに分ける": group_folds(train),
    }
    for name, folds in folds_by_name.items():
        overlap = [len(set(train.iloc[a]["student_id"]) & set(train.iloc[b]["student_id"])) for a, b in folds]
        scores = cross_val_score(build_model(), train[FEATURES], train["next_score"], cv=folds,
                                 scoring="neg_mean_absolute_error", error_score="raise")
        rows.append({"split": name, "CV_MAE": -scores.mean(), "重なる人数の最大": max(overlap)})
    return pd.DataFrame(rows)


def main() -> None:
    print_header("Chapter 21: 同じ人の記録をまとめて分ける")
    train, final = split_students()
    print(f"学習: {train.student_id.nunique()}人・{len(train)}行 / 最終テスト: {final.student_id.nunique()}人・{len(final)}行")
    print("目的は、まだ記録のない新しい人の点数を予想することです。")
    table = compare_models(train)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    name = table.sort_values(["group_CV_MAE", "model"]).iloc[0]["model"]
    model = build_model() if name == "forest" else DummyRegressor(strategy="mean")
    model.fit(train[FEATURES], train["next_score"])
    pred = model.predict(final[FEATURES])
    print(f"選んだ{name}の、新しい12人での最終MAE: {mean_absolute_error(final['next_score'], pred):.2f}点")
    print("student_idは分割に使う番号です。予想の手がかりには入れません。")


if __name__ == "__main__":
    main()
