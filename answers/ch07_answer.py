from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

import pandas as pd
from common import print_header, print_table
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def main() -> None:
    print_header("Answer 7: 木系モデル")

    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = (data.target == 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    rows = []
    for depth in [2, 4, None]:
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append(
            {
                "model": f"Tree depth={depth}",
                "accuracy": f"{accuracy_score(y_test, pred):.4f}",
                "f1": f"{f1_score(y_test, pred):.4f}",
            }
        )

    forest = None
    for n_estimators in [50, 150]:
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=5, random_state=42
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append(
            {
                "model": f"Forest n={n_estimators}",
                "accuracy": f"{accuracy_score(y_test, pred):.4f}",
                "f1": f"{f1_score(y_test, pred):.4f}",
            }
        )
        forest = model

    print_table(rows, ["model", "accuracy", "f1"])

    importances = pd.Series(forest.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )
    print("\nfeature importance top 10:")
    print(importances.head(10).to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
