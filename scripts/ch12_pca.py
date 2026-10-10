from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import print_header


def build_projection(n_components: int = 2) -> Pipeline:
    return Pipeline(steps=[
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=n_components, svd_solver="full")),
    ])


def main() -> None:
    print_header("Chapter 12: たくさんの特徴を2軸で見る")
    print("花の4つの測定値を組み合わせ、新しい2つの軸にまとめます。")
    data = load_iris(as_frame=True)
    X_train, X_check, _, y_check = train_test_split(
        data.data, data.target, test_size=0.25, random_state=42, stratify=data.target
    )
    projection = build_projection()
    # PCAの軸を決めるとき、品種の答えは渡さない。
    train_coordinates = projection.fit_transform(X_train)
    check_coordinates = projection.transform(X_check)
    ratios = projection.named_steps["pca"].explained_variance_ratio_

    print(f"学習用: {X_train.shape} → {train_coordinates.shape}（行数, 列数）")
    print(f"確認用: {X_check.shape} → {check_coordinates.shape}")
    for axis, ratio in enumerate(ratios, start=1):
        print(f"第{axis}軸の寄与率: {ratio:.1%}")
    print(f"2軸の累積寄与率: {ratios.sum():.1%}")
    print("寄与率は標準化した学習用のばらつきを残せた割合です。分類の正解率ではありません。")

    result = pd.DataFrame(check_coordinates[:5], columns=["第1軸", "第2軸"])
    result["本当の品種"] = data.target_names[y_check.iloc[:5].to_numpy()]
    print("\n確認用の最初の5件（品種は変換後に表示するだけ）:")
    print(result.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("各軸は4つの測定値の組み合わせです。cm単位の長さとしては読みません。")


if __name__ == "__main__":
    main()
