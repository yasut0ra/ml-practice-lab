from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from common import print_header
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    print_header("Answer 3: 分類")

    data = load_iris(as_frame=True)
    X = data.data[["sepal length (cm)", "sepal width (cm)"]]
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print(f"accuracy with sepal features: {accuracy_score(y_test, pred):.4f}")
    print("confusion matrix:")
    print(confusion_matrix(y_test, pred))
    print("test class counts:")
    print(y_test.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
