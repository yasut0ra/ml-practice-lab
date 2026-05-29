from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from common import print_header
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression


def main() -> None:
    print_header("Answer 1: 基本概念")

    data = load_diabetes(as_frame=True)
    X = data.data[["bmi", "bp"]]
    y = data.target

    model = LinearRegression()
    model.fit(X, y)
    pred = model.predict(X.iloc[:5])

    for i, (actual, predicted) in enumerate(zip(y.iloc[:5], pred), start=1):
        print(f"{i}: actual={actual:.1f}, predicted={predicted:.1f}")

    for feature, coef in zip(X.columns, model.coef_):
        print(f"coef {feature}: {coef:.2f}")


if __name__ == "__main__":
    main()
