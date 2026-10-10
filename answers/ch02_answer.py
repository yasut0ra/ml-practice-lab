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
    rows = []
    # 問1・2: 同じ分割の中でalphaを比べる。問3: 分割の割合も変える。
    for test_size in [0.2, 0.3]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        for alpha in [0.1, 1.0, 10.0]:
            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            rows.append(
                {
                    "test_size": test_size,
                    "model": f"Ridge(alpha={alpha})",
                    "MAE": f"{mean_absolute_error(y_test, pred):.3f}",
                    "RMSE": f"{np.sqrt(mean_squared_error(y_test, pred)):.3f}",
                    "R2": f"{r2_score(y_test, pred):.3f}",
                }
            )

    print_table(rows, ["test_size", "model", "MAE", "RMSE", "R2"])
    print("test_sizeを変えると学習する例と確認する例も変わるため、モデルの差だけではありません。")


if __name__ == "__main__":
    main()
