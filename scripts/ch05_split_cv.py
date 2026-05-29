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
    print("一度だけの分割では評価がぶれるため、cross validation で複数回評価します。")

    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = (data.target == 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    holdout_model = build_model()
    holdout_model.fit(X_train, y_train)
    holdout_pred = holdout_model.predict(X_test)
    holdout_accuracy = accuracy_score(y_test, holdout_pred)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(build_model(), X, y, cv=cv, scoring="accuracy")

    print(f"holdout accuracy: {holdout_accuracy:.4f}")
    print(f"cv scores: {np.round(cv_scores, 4)}")
    print(f"cv mean +/- std: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")


if __name__ == "__main__":
    main()
