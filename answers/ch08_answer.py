from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from common import print_header, print_table
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def main() -> None:
    print_header("Answer 8: SVM と k-NN")

    data = load_digits()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    rows = []
    for c in [0.5, 3, 10]:
        model = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("clf", SVC(C=c, kernel="rbf", gamma="scale")),
            ]
        )
        model.fit(X_train, y_train)
        rows.append(
            {
                "model": f"SVM C={c}",
                "accuracy": f"{accuracy_score(y_test, model.predict(X_test)):.4f}",
            }
        )

    for k in [3, 5, 11]:
        model = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=k)),
            ]
        )
        model.fit(X_train, y_train)
        rows.append(
            {
                "model": f"k-NN k={k}",
                "accuracy": f"{accuracy_score(y_test, model.predict(X_test)):.4f}",
            }
        )

    no_scale = SVC(C=3, kernel="rbf", gamma="scale")
    no_scale.fit(X_train, y_train)
    rows.append(
        {
            "model": "SVM no scale",
            "accuracy": f"{accuracy_score(y_test, no_scale.predict(X_test)):.4f}",
        }
    )

    # 問3: k-NNでも、同じk=5のまま標準化の有無だけを比べる。
    no_scale_knn = KNeighborsClassifier(n_neighbors=5)
    no_scale_knn.fit(X_train, y_train)
    rows.append({"model": "k-NN no scale", "accuracy":
                 f"{accuracy_score(y_test, no_scale_knn.predict(X_test)):.4f}"})

    print_table(rows, ["model", "accuracy"])
    print("SVM C=3とSVM no scale、k-NN k=5とk-NN no scaleを対応させて比べます。")
    print("画素は同じ単位です。標準化したほうが必ず正解率が上がるわけではありません。")


if __name__ == "__main__":
    main()
