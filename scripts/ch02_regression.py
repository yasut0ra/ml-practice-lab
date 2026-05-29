from __future__ import annotations

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from common import print_header, print_metrics


def evaluate_regression(name: str, y_true, y_pred) -> None:
    print(f"\n{name}")
    print_metrics(
        {
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": r2_score(y_true, y_pred),
        }
    )


def main() -> None:
    print_header("Chapter 2: 回帰")
    print("回帰は、連続値を予測する教師あり学習です。ここでは糖尿病データの進行度を予測します。")

    data = load_diabetes(as_frame=True)
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    linear = LinearRegression()
    linear.fit(X_train, y_train)
    evaluate_regression("LinearRegression", y_test, linear.predict(X_test))

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    evaluate_regression("Ridge", y_test, ridge.predict(X_test))


if __name__ == "__main__":
    main()
