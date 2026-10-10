from __future__ import annotations

from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import print_header


def main() -> None:
    print_header("Chapter 3: 分類")
    print("分類は『どの種類？』を当てる方法です。花の測定値から、3種類の品種を当てます。")

    # 1. 花の測定値 X と、品種の番号 y を取り出す。
    data = load_iris(as_frame=True)
    X = data.data
    y = data.target

    # 2. 品種の割合を保ちながら、学習用と確認用に分ける。
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. 数値の尺度をそろえる処理と、分類モデルをまとめて学ぶ。
    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    # 手がかりを見ず、学習用で一番多い品種をいつも答える方法と比べる。
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_accuracy = accuracy_score(y_test, baseline.predict(X_test))
    print(f"Baseline（いつも最多の品種）: {baseline_accuracy:.1%}")
    print(f"accuracy: {accuracy_score(y_test, pred):.4f}")
    print(f"正解率: {accuracy_score(y_test, pred):.1%}（確認用の{len(y_test)}件全体での割合）")
    print("正解率の数値だけでなく、単純な基準より良いかを見ます。")
    print("予測例:")
    for i in range(5):
        actual = data.target_names[y_test.iloc[i]]
        predicted = data.target_names[pred[i]]
        print(f"  actual（本当）={actual}, predicted（予想）={predicted}")


if __name__ == "__main__":
    main()
