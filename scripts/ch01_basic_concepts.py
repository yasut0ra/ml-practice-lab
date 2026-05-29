from __future__ import annotations

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression

from common import print_header


def main() -> None:
    print_header("Chapter 1: 機械学習の基本概念")
    print(
        "機械学習では、特徴量 X から目的変数 y を予測する規則をデータから学習します。"
    )

    data = load_diabetes(as_frame=True)
    X = data.data[["bmi"]]
    y = data.target

    model = LinearRegression()
    model.fit(X, y)

    example = X.iloc[[0]]
    prediction = model.predict(example)[0]

    print(f"特徴量の形: {X.shape}")
    print(f"目的変数の形: {y.shape}")
    print(f"使う特徴量: {list(X.columns)}")
    print(f"最初のデータの実測値: {y.iloc[0]:.1f}")
    print(f"最初のデータの予測値: {prediction:.1f}")
    print(f"学習された係数: {model.coef_[0]:.2f}")


if __name__ == "__main__":
    main()
