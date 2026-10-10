from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import print_header


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000)),
        ]
    )


def main() -> None:
    print_header("Chapter 5: train/test split と cross validation")
    print("練習した問題を覚えただけかもしれないので、使っていない問題で確かめます。")
    print("交差検証（CV）では確認する部分を交代し、毎回学び直して評価します。")

    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = (data.target == 0).astype(int)

    # 1. 最後の確認用（テスト）を先に取っておく。
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 2. 学習用の中だけで、5回の学習と検証を行う。
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(build_model(), X_train, y_train, cv=cv, scoring="accuracy")

    # 3. 学習用すべてで学び直し、取っておいたテストで最後に確認する。
    holdout_model = build_model()
    holdout_model.fit(X_train, y_train)
    holdout_pred = holdout_model.predict(X_test)
    holdout_accuracy = accuracy_score(y_test, holdout_pred)

    print(f"cv scores: {np.round(cv_scores, 4)}")
    print(f"cv mean +/- std: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"holdout accuracy: {holdout_accuracy:.4f}")
    print("cv scores は毎回の正解率、mean は平均、std はばらつきです。")
    print("holdout は最後のテストの成績です。CV にこのテストのデータは使っていません。")


if __name__ == "__main__":
    main()
