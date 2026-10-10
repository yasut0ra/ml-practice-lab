from __future__ import annotations

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression

from common import print_header


def main() -> None:
    print_header("Chapter 1: 機械学習の基本概念")
    print(
        "機械学習は、過去の例から規則を見つけ、新しい例の答えを予想する方法です。"
    )
    print("この章では BMI を手がかり X にして、病気の進行度 y を予想します。")

    # 1. 手がかり X と、実際の答え y を用意する。
    data = load_diabetes(as_frame=True)
    X = data.data[["bmi"]]
    y = data.target

    # 2. モデルを準備し、手がかりと答えから学ぶ。
    model = LinearRegression()
    model.fit(X, y)

    # 3. 手がかりだけを渡して予想する（今回は学習に使った1件）。
    example = X.iloc[[0]]
    prediction = model.predict(example)[0]

    print(f"特徴量の形: {X.shape}")
    print(f"目的変数の形: {y.shape}")
    print(f"使う特徴量: {list(X.columns)}")
    print(f"最初のデータの実測値: {y.iloc[0]:.1f}")
    print(f"最初のデータの予測値: {prediction:.1f}")
    print(f"学習された係数: {model.coef_[0]:.2f}")
    print(f"読み方: {X.shape[0]}件のデータに、{X.shape[1]}種類の手がかりがあります。")
    print("実測値は本当の答え、予測値はモデルの予想です。違いが予測のずれです。")
    print("この1件は学習に使っています。初めて見るデータでの評価は第2章で行います。")


if __name__ == "__main__":
    main()
