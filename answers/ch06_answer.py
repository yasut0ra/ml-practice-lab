from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

import pandas as pd
from ch06_preprocessing import make_customer_data
from common import print_header, print_table
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_model(strategy: str) -> Pipeline:
    numeric_features = ["age", "income"]
    categorical_features = ["city", "plan"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy=strategy)),
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
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


def main() -> None:
    print_header("Answer 6: 前処理")
    X, y = make_customer_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    # 問1: 1行の表で、学習用と同じ名前の列を用意する。
    new_customer = pd.DataFrame(
        [{"age": 46, "income": 640, "city": "Tokyo", "plan": "standard"}]
    )
    rows = []
    # 問2: 同じ分割のまま、空欄を埋める方法だけを変えて比べる。
    for strategy in ["median", "mean"]:
        model = build_model(strategy)
        model.fit(X_train, y_train)
        rows.append({"補完方法": strategy,
                     "accuracy": f"{accuracy_score(y_test, model.predict(X_test)):.4f}",
                     "新顧客の予測確率": f"{model.predict_proba(new_customer)[0, 1]:.4f}"})
    print_table(rows, ["補完方法", "accuracy", "新顧客の予測確率"])
    print("予測確率はモデルが出した見込みです。この小さなデータだけで実際の割合を保証する値ではありません。")
    # 問3: 種類ごとにどの列ができたかを確かめる。
    print("encoded feature names:")
    for name in model.named_steps["preprocess"].get_feature_names_out():
        print(f"  {name}")


if __name__ == "__main__":
    main()
