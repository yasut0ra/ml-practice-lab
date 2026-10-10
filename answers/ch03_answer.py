from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from common import print_header, print_table
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    print_header("Answer 3: 分類")

    data = load_iris(as_frame=True)
    y = data.target
    sepal_columns = ["sepal length (cm)", "sepal width (cm)"]
    conditions = [
        ("4特徴量・割合を保つ", data.data, y),
        ("4特徴量・割合指定なし", data.data, None),
        ("がく片の2特徴量", data.data[sepal_columns], y),
    ]
    rows = []
    for name, X, stratify in conditions:
        # 問2ではstratifyだけ、問3では特徴量だけを変えて比べる。
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )
        model = Pipeline(
            [("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({"条件": name, "accuracy": f"{accuracy_score(y_test, pred):.4f}",
                     "クラス0・1・2の件数": [int((y_test == i).sum()) for i in range(3)]})
        # 問1: どの種類を間違えたかも、同じ順序の行と列で確認する。
        print(f"\n{name} の混同行列（行=実際、列=予想）:")
        print("品種の順序:", " → ".join(data.target_names))
        print(confusion_matrix(y_test, pred, labels=[0, 1, 2]))
    print_table(rows, ["条件", "accuracy", "クラス0・1・2の件数"])


if __name__ == "__main__":
    main()
