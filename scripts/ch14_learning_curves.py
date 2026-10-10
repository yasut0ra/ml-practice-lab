from __future__ import annotations

import pandas as pd
from sklearn.datasets import make_moons
from sklearn.model_selection import StratifiedKFold, learning_curve, train_test_split
from sklearn.tree import DecisionTreeClassifier

from common import print_header


def make_practice_data() -> tuple[pd.DataFrame, pd.Series]:
    X, y = make_moons(n_samples=400, noise=0.3, random_state=42)
    return pd.DataFrame(X, columns=["x1", "x2"]), pd.Series(y, name="group")


def curve_table(model, X_train, y_train, train_sizes=None) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sizes, training, validation = learning_curve(
        model, X_train, y_train, cv=cv, scoring="accuracy",
        train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0] if train_sizes is None else train_sizes,
        shuffle=True, random_state=42, error_score="raise",
    )
    return pd.DataFrame({
        "学習件数": sizes,
        "学習用の平均": training.mean(axis=1),
        "検証用の平均": validation.mean(axis=1),
        "検証用のばらつき": validation.std(axis=1),
    })


def main() -> None:
    print_header("Chapter 14: データを増やすと成績は変わる？")
    X, y = make_practice_data()
    X_train, X_final, y_train, _ = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = DecisionTreeClassifier(max_depth=None, random_state=42)
    result = curve_table(model, X_train, y_train)
    print(result.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n最終テスト{len(X_final)}件は、曲線を作るときに使っていません。")
    print("学習件数はCVの各回でモデルが学ぶ例の数です。300件全体の件数ではありません。")
    print("練習した例と別の例の成績を比べ、次に何を試すか考えます。")
    print("検証用が毎回良くなるとは限りません。この曲線だけで今後の伸びは保証できません。")


if __name__ == "__main__":
    main()
