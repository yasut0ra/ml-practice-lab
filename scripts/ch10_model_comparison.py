from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import print_header


def main() -> None:
    print_header("Chapter 10: 簡単なモデル比較")
    print("同じデータ、同じ分割方法、同じ評価指標で複数モデルを比較します。")

    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = (data.target == 0).astype(int)

    models = {
        "LogisticRegression": Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=3000)),
            ]
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=150, max_depth=5, random_state=42
        ),
        "SVM": Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=3, gamma="scale")),
            ]
        ),
        "k-NN": Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=7)),
            ]
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for name, model in models.items():
        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring={"accuracy": "accuracy", "f1": "f1"},
        )
        rows.append(
            {
                "model": name,
                "accuracy_mean": scores["test_accuracy"].mean(),
                "accuracy_std": scores["test_accuracy"].std(),
                "f1_mean": scores["test_f1"].mean(),
                "f1_std": scores["test_f1"].std(),
            }
        )

    result = pd.DataFrame(rows).sort_values("f1_mean", ascending=False)
    print(result.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
