from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ch14_learning_curves import curve_table, make_practice_data
from common import print_header


def main() -> None:
    print_header("Chapter 14 Answer: 曲線を比べ、最後に確かめる")
    X, y = make_practice_data()
    X_train, X_final, y_train, y_final = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    models = {"深さ3": DecisionTreeClassifier(max_depth=3, random_state=42),
              "上限なし": DecisionTreeClassifier(max_depth=None, random_state=42)}
    curves, candidates = [], []
    for name, model in models.items():
        # 問1・2: 深さを2通り、件数の割合を0.1刻みの10通りにする。
        result = curve_table(model, X_train, y_train, train_sizes=[count / 10 for count in range(1, 11)])
        result.insert(0, "モデル", name)
        curves.append(result)
        candidates.append({"モデル": name, "最大件数のCV平均": result.iloc[-1]["検証用の平均"]})
    combined = pd.concat(curves, ignore_index=True)
    print(combined.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("練習した例の成績と、別の例の成績の差を比べます。件数が増えても毎回伸びるとは限りません。")

    # 問3: CVの最大件数での平均を基準に選び、最終テストは選んだ1つだけ。
    comparison = pd.DataFrame(candidates).sort_values(["最大件数のCV平均", "モデル"], ascending=[False, True])
    best_name = comparison.iloc[0]["モデル"]
    model = models[best_name].fit(X_train, y_train)
    print(f"選んだモデル: {best_name} / 最後の正解率: {accuracy_score(y_final, model.predict(X_final)):.4f}")
    output = Path(__file__).resolve().parents[1] / "outputs" / "learning_curves.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    print(f"曲線の比較表を保存しました: {output}")


if __name__ == "__main__":
    main()
