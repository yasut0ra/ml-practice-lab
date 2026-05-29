from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from common import print_header, print_table
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    print_header("Answer 4: 評価指標")

    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = (data.target == 0).astype(int)
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
    prob = model.predict_proba(X_test)[:, 1]

    rows = []
    for threshold in [0.3, 0.5, 0.7]:
        pred = (prob >= threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "precision": f"{precision_score(y_test, pred):.4f}",
                "recall": f"{recall_score(y_test, pred):.4f}",
                "f1": f"{f1_score(y_test, pred):.4f}",
            }
        )

    print_table(rows, ["threshold", "precision", "recall", "f1"])


if __name__ == "__main__":
    main()
