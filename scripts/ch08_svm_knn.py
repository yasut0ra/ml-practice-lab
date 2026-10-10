from __future__ import annotations

from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import print_header, print_table


def main() -> None:
    print_header("Chapter 8: SVM と k-NN")
    print("SVM は境界をうまく引くモデル、k-NN は近いデータの多数決で分類するモデルです。")
    print("ここでは、手書き数字の画像から0〜9のどの数字かを当てます。")

    # 画像1枚を、8×8 = 64個の明るさの数値として使う。
    data = load_digits()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    models = {
        "SVM": Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=10, gamma="scale")),
            ]
        ),
        "k-NN": Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
    }

    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({"model": name, "accuracy": f"{accuracy_score(y_test, pred):.4f}"})

    print_table(rows, ["model", "accuracy"])
    print(f"画像データの形: {data.images.shape} (枚数, 高さ, 幅)")
    print(f"モデルに渡す表の形: {X.shape}（枚数, ピクセルの数）")
    print("accuracy は確認用の画像での正解率です。0.98なら約98%が正解です。")


if __name__ == "__main__":
    main()
