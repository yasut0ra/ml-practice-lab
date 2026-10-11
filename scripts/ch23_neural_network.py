from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_digits
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import print_header


def split_digits():
    data = load_digits(as_frame=True)
    return train_test_split(data.data, data.target, test_size=0.2, stratify=data.target, random_state=42)


def build_model(width: int = 32) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("network", MLPClassifier(hidden_layer_sizes=(width,), activation="relu", solver="adam",
                    max_iter=250, early_stopping=True, validation_fraction=0.15,
                    n_iter_no_change=12, random_state=42)),
    ])


def compare_widths(X_train, y_train, widths=(16, 32)) -> pd.DataFrame:
    folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=42).split(X_train, y_train))
    rows = []
    for width in widths:
        scores = cross_val_score(build_model(width), X_train, y_train, cv=folds, scoring="accuracy", error_score="raise")
        rows.append({"width": width, "CV_accuracy": scores.mean()})
    return pd.DataFrame(rows)


def main() -> None:
    print_header("Chapter 23: ニューラルネットワークで数字を読む")
    X_train, X_test, y_train, y_test = split_digits()
    model = build_model().fit(X_train, y_train)
    pred = model.predict(X_test)
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    network = model.named_steps["network"]
    print(f"学習用{len(y_train)}件 / 最終テスト{len(y_test)}件 / 入力{X_train.shape[1]}列")
    print(f"最多の数字の基準: {accuracy_score(y_test, baseline.predict(X_test)):.3f}")
    print(f"32個の中間ユニットのモデル: {accuracy_score(y_test, pred):.3f}")
    print(f"重みの表の形: {[weights.shape for weights in network.coefs_]}")
    print(f"学習の繰り返し: {network.n_iter_} / 最後に記録した学習loss: {network.loss_curve_[-1]:.3f}")
    print(pd.DataFrame({"actual": y_test, "predicted": pred}).head(8).to_string(index=False))
    print("内部の検証用で伸びが止まると学習を止めます。最終テストはその判断に使いません。")
    print("lossは学習のずれをまとめた量です。正解率とは単位も意味も違います。")


if __name__ == "__main__":
    main()
