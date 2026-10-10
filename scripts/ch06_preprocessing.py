from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import print_header


def make_customer_data() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(
        {
            "age": [22, 35, 58, 44, np.nan, 27, 31, 62, 48, 39, 29, 53, 41, np.nan, 25, 57],
            "income": [320, 540, 720, 610, 450, 380, 410, 800, 690, np.nan, 360, 750, 620, 470, 330, 780],
            "city": [
                "Tokyo",
                "Osaka",
                "Tokyo",
                "Fukuoka",
                "Osaka",
                "Tokyo",
                "Fukuoka",
                "Tokyo",
                "Osaka",
                "Fukuoka",
                "Tokyo",
                "Osaka",
                "Tokyo",
                "Fukuoka",
                "Osaka",
                "Tokyo",
            ],
            "plan": [
                "free",
                "standard",
                "premium",
                "standard",
                "free",
                "free",
                "standard",
                "premium",
                "premium",
                "standard",
                "free",
                "premium",
                "standard",
                "free",
                "free",
                "premium",
            ],
        }
    )
    y = pd.Series([0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1], name="upgrade")
    return X, y


def main() -> None:
    print_header("Chapter 6: 前処理")
    print("空欄を埋め、数値の尺度をそろえ、文字の種類を0と1の列に変換します。")

    # 1. 数値の列と、文字の列で別々の準備をする。
    X, y = make_customer_data()
    numeric_features = ["age", "income"]
    categorical_features = ["city", "plan"]

    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    # 2. データを整える手順と、予想するモデルをひとまとめにする。
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    # 3. 分割してから学ぶ。空欄を埋める値なども学習用だけで決まる。
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print(f"accuracy: {accuracy_score(y_test, pred):.4f}")
    print("変換前の列:", list(X.columns))
    transformed = model.named_steps["preprocess"].transform(X.head(3))
    print("変換後の特徴量数:", transformed.shape[1])
    print("都市やプランが種類ごとの列に分かれたため、列数が増えました。")
    print(f"確認用は{len(y_test)}件だけです。正解率より、変換の流れを見てみましょう。")


if __name__ == "__main__":
    main()
