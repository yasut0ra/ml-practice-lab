from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from common import print_header, print_table
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000)),
        ]
    )


def main() -> None:
    print_header("Answer 5: split と cross validation")

    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = (data.target == 0).astype(int)

    # 第5章と同じく、最終テストはCVの比較には使わない。
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    rows = []
    for n_splits in [3, 5, 10]:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for scoring in ["accuracy", "f1"]:
            scores = cross_val_score(build_model(), X_train, y_train, cv=cv, scoring=scoring)
            rows.append(
                {
                    "cv": n_splits,
                    "metric": scoring,
                    "mean": f"{scores.mean():.4f}",
                    "std": f"{scores.std():.4f}",
                }
            )

    print_table(rows, ["cv", "metric", "mean", "std"])


if __name__ == "__main__":
    main()
