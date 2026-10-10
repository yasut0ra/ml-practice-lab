from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from common import print_header, print_table


def main() -> None:
    print_header("Chapter 7: 決定木・ランダムフォレスト")
    print("決定木は『この数値は大きい？』と質問をたどって判断する仕組みです。")
    print("ランダムフォレストは、違いのある複数の木の予測を組み合わせます。")

    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = (data.target == 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 同じ確認用データで、1本の木と100本の木を比べる。
    models = {
        "DecisionTree": DecisionTreeClassifier(max_depth=4, random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42
        ),
    }

    rows = []
    fitted_forest = None
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append(
            {
                "model": name,
                "accuracy": f"{accuracy_score(y_test, pred):.4f}",
                "f1": f"{f1_score(y_test, pred):.4f}",
            }
        )
        if name == "RandomForest":
            fitted_forest = model

    print_table(rows, ["model", "accuracy", "f1"])

    importances = pd.Series(
        fitted_forest.feature_importances_, index=X.columns
    ).sort_values(ascending=False)
    print("\nRandomForest の重要特徴量 top 5:")
    for feature, value in importances.head(5).items():
        print(f"  {feature}: {value:.4f}")
    print("\n重要度は、木の分岐でどの手がかりが使われたかの目安です。原因を示す値ではありません。")


if __name__ == "__main__":
    main()
