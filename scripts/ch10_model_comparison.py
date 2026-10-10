from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import print_header


def build_models() -> dict:
    """同じ候補をサンプルと解答例で使う。呼ぶたびに新しいモデルを作る。"""
    return {
        "Baseline": DummyClassifier(strategy="most_frequent"),
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


def main() -> None:
    print_header("Chapter 10: 簡単なモデル比較")
    print("同じデータ、同じ分割方法、同じ評価指標で複数モデルを比較します。")
    print("同じ問題を、同じ採点方法で比べると考えてください。")

    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = (data.target == 0).astype(int)

    # 1. 最終テストは、モデル選びの前に取っておく。
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    models = build_models()

    # 2. 学習用の中で、単純な基準を含む5つの方法を比較する。
    # どのモデルでも同じ5回の分割を使い、平均とばらつきを比べる。
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for name, model in models.items():
        scores = cross_validate(
            model,
            X_train,
            y_train,
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

    result = pd.DataFrame(rows).sort_values(
        ["f1_mean", "model"], ascending=[False, True]
    )
    print("CVの比較表（モデルを選ぶための成績）:")
    print(result.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nmean は5回の平均、std はばらつきです。F1 の平均が高い順に並べています。")
    print("一番上は、今回のデータと条件で良かったモデルです。すべての用途で一番とは限りません。")

    # 3. F1の平均で選んでから学び直す。テストの成績は選び方に使わない。
    best_name = result.iloc[0]["model"]
    best_model = models[best_name]
    best_model.fit(X_train, y_train)
    final_pred = best_model.predict(X_test)
    print(f"\nCVで選んだモデル: {best_name}")
    print(f"最終テスト accuracy: {accuracy_score(y_test, final_pred):.4f}")
    print(f"最終テスト F1: {f1_score(y_test, final_pred, zero_division=0):.4f}")
    print("最終テストを見て設定を選び直すと、最後の確認として使えなくなります。")


if __name__ == "__main__":
    main()
