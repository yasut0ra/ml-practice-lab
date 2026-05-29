from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

import pandas as pd
from ch06_preprocessing import make_customer_data
from common import print_header
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def main() -> None:
    print_header("Answer 6: 前処理")

    X, y = make_customer_data()
    numeric_features = ["age", "income"]
    categorical_features = ["city", "plan"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="mean")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)

    new_customer = pd.DataFrame(
        [{"age": 46, "income": 640, "city": "Tokyo", "plan": "standard"}]
    )
    probability = model.predict_proba(new_customer)[0, 1]
    print(f"upgrade probability: {probability:.4f}")
    print("encoded feature names:")
    for name in model.named_steps["preprocess"].get_feature_names_out():
        print(f"  {name}")


if __name__ == "__main__":
    main()
