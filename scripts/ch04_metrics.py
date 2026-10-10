from __future__ import annotations

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import print_header, print_metrics, print_table


def main() -> None:
    print_header("Chapter 4: 評価指標")
    print("予想が当たった割合だけでなく、誤検知と見逃しの件数を確認します。")
    print("ここでの陽性（1）は悪性、陰性（0）は良性です。")

    # 1. 見つけたい対象である悪性を1にする。
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = (data.target == 0).astype(int)  # malignant を陽性クラス 1 として扱う

    # 2. 学習用と確認用に分け、学習用だけで前処理とモデルを学ぶ。
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000)),
        ]
    )
    model.fit(X_train, y_train)

    # 3. 予想した種類と、陽性である確率を求める。
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    print_metrics(
        {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, prob),
        }
    )
    print("confusion matrix [[TN, FP], [FN, TP]]:")
    matrix = confusion_matrix(y_test, pred, labels=[0, 1])
    print(matrix)
    print("\n行は実際の答え、列は予想です。右上が誤検知、左下が見逃しです。")
    print_table(
        [
            {"実際の答え": "良性（0）", "予想:良性（0）": matrix[0, 0], "予想:悪性（1）": matrix[0, 1]},
            {"実際の答え": "悪性（1）", "予想:良性（0）": matrix[1, 0], "予想:悪性（1）": matrix[1, 1]},
        ],
        ["実際の答え", "予想:良性（0）", "予想:悪性（1）"],
    )
    print("precision: 陽性と予想した中で本当に陽性の割合。recall: 本当の陽性を拾えた割合。")


if __name__ == "__main__":
    main()
