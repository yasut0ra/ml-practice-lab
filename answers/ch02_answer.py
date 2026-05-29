from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
from common import print_header, print_table
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def main() -> None:
    print_header("Answer 2: 回帰")

    data = load_diabetes(as_frame=True)
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    rows = []
    for alpha in [0.1, 1.0, 10.0]:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append(
            {
                "model": f"Ridge(alpha={alpha})",
                "MAE": f"{mean_absolute_error(y_test, pred):.3f}",
                "RMSE": f"{np.sqrt(mean_squared_error(y_test, pred)):.3f}",
                "R2": f"{r2_score(y_test, pred):.3f}",
            }
        )

    print_table(rows, ["model", "MAE", "RMSE", "R2"])


if __name__ == "__main__":
    main()
