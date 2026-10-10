from pathlib import Path
import sys

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ch12_pca import build_projection
from common import print_header


def main() -> None:
    print_header("Chapter 12 Answer: 軸の数と、残せるばらつき")
    data = load_iris(as_frame=True)
    X_train, X_check, _, y_check = train_test_split(
        data.data, data.target, test_size=0.25, random_state=42, stratify=data.target
    )
    # 問1: 同じ学習用で、軸の数だけを変える。
    rows = []
    for count in [1, 2, 3]:
        projection = build_projection(count).fit(X_train)
        ratios = projection.named_steps["pca"].explained_variance_ratio_
        rows.append({"軸の数": count, "累積寄与率": ratios.sum()})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("軸を増やすと残せるばらつきは減りませんが、2軸の図には描けなくなります。")

    # 問2: 2軸の寄与率をそれぞれ見て、足した値を確認する。
    projection = build_projection(2).fit(X_train)
    ratios = projection.named_steps["pca"].explained_variance_ratio_
    print(f"第1軸 {ratios[0]:.1%} + 第2軸 {ratios[1]:.1%} = {ratios.sum():.1%}")
    print("寄与率が高くても、品種を当てる正解率が高いとは限りません。")

    # 問3: 新しい例にはtransformを使い、決めた軸を動かさない。
    coordinates = projection.transform(X_check.iloc[:5])
    result = pd.DataFrame(coordinates, columns=["第1軸", "第2軸"])
    result["本当の品種"] = data.target_names[y_check.iloc[:5].to_numpy()]
    print(result.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
