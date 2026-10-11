from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from common import print_header


def split_energy():
    rng = np.random.default_rng(42)
    temperature = rng.uniform(-10, 35, 320)
    # 暖房・冷房を使うというイメージの架空データ。実際の電力の式ではない。
    energy = 3 + 0.015 * (temperature - 20) ** 2 + rng.normal(0, 0.25, 320)
    X = pd.DataFrame({"temperature_c": temperature})
    y = pd.Series(energy, name="energy_kwh")
    return train_test_split(X, y, test_size=0.25, random_state=42)


def build_model(degree: int = 2) -> Pipeline:
    return Pipeline([
        ("features", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scale", StandardScaler()), ("reg", Ridge(alpha=0.1)),
    ])


def compare_degrees(X_train, y_train, degrees=(1, 2, 3)) -> pd.DataFrame:
    folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(X_train))
    rows = []
    for degree in degrees:
        scores = cross_val_score(build_model(degree), X_train, y_train, cv=folds,
                                 scoring="neg_mean_absolute_error", error_score="raise")
        rows.append({"degree": degree, "CV_MAE_kwh": -scores.mean()})
    return pd.DataFrame(rows)


def choose_degree(table) -> int:
    return int(table.sort_values(["CV_MAE_kwh", "degree"]).iloc[0]["degree"])


def main() -> None:
    print_header("Chapter 20: 手がかりの作り方を変える")
    X_train, X_test, y_train, y_test = split_energy()
    print("架空の気温と1日の使用電力量 / 学習240件・最終テスト80件")
    print(X_train.describe().to_string(float_format=lambda x: f"{x:.2f}"))
    table = compare_degrees(X_train, y_train)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    degree = choose_degree(table)
    model = build_model(degree).fit(X_train, y_train)
    baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
    print(f"選んだ次数: {degree}")
    print(f"平均の基準の最終MAE: {mean_absolute_error(y_test, baseline.predict(X_test)):.3f}kWh")
    print(f"選んだモデルの最終MAE: {mean_absolute_error(y_test, model.predict(X_test)):.3f}kWh")
    example = pd.DataFrame({"temperature_c": [0, 10, 20]})
    print("\n同じ入力から作る列の例（2次）:")
    print(build_model(2).named_steps["features"].fit_transform(example))
    print("新しい測定値を集めなくても、気温の2乗などを手がかりにできます。答えの電力量は入力に入れません。")


if __name__ == "__main__":
    main()
