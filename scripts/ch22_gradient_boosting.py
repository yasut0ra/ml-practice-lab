from __future__ import annotations

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from ch13_csv_regression import FEATURES, TARGET, build_model as build_rent_model, load_rent_data
from common import print_header


def split_rooms():
    data = load_rent_data()
    return train_test_split(data[FEATURES], data[TARGET], test_size=0.25, random_state=42)


def build_boosting(learning_rate: float = 0.1, max_iter: int = 60) -> Pipeline:
    # 各モデルに新しい前処理を渡す。CVでもその回の学習側だけで学ぶ。
    return Pipeline([
        ("prepare", build_rent_model().named_steps["prepare"]),
        ("reg", HistGradientBoostingRegressor(learning_rate=learning_rate, max_iter=max_iter,
                 max_leaf_nodes=7, min_samples_leaf=10, early_stopping=False, random_state=42)),
    ])


def compare_models(X_train, y_train, models) -> pd.DataFrame:
    folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(X_train))
    rows = []
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=folds,
                                 scoring="neg_mean_absolute_error", error_score="raise")
        rows.append({"model": name, "CV_MAE_man": -scores.mean()})
    return pd.DataFrame(rows)


def main() -> None:
    print_header("Chapter 22: 小さな木で予想を少しずつ直す")
    X_train, X_test, y_train, y_test = split_rooms()
    models = {"mean": DummyRegressor(strategy="mean"), "Ridge": build_rent_model(), "boosting": build_boosting()}
    table = compare_models(X_train, y_train, models)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    name = table.sort_values(["CV_MAE_man", "model"]).iloc[0]["model"]
    model = models[name].fit(X_train, y_train)
    print(f"学習90件 / 最終テスト30件: 選んだ{name}のMAE={mean_absolute_error(y_test, model.predict(X_test)):.2f}万円")
    print("ブースティングが必ず勝つわけではありません。同じCVで、単純な方法と比べます。")


if __name__ == "__main__":
    main()
