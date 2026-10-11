from __future__ import annotations

from math import ceil

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from ch13_csv_regression import FEATURES, TARGET, build_model, load_rent_data
from common import print_header


def split_rooms():
    data = load_rent_data()
    X_rest, X_test, y_rest, y_test = train_test_split(data[FEATURES], data[TARGET], test_size=0.25, random_state=42)
    X_train, X_cal, y_train, y_cal = train_test_split(X_rest, y_rest, test_size=1 / 3, random_state=42)
    return X_train, X_cal, X_test, y_train, y_cal, y_test


def conformal_radius(residuals, coverage: float = 0.9) -> float:
    scores = np.asarray(residuals, dtype=float)
    if scores.ndim != 1 or len(scores) == 0 or not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("幅の調整用のずれは、空でない一次元の、有限な0以上の値にしてください。")
    if not 0 < coverage < 1:
        raise ValueError("目標の割合は0より大きく、1より小さくしてください。")
    rank = ceil((len(scores) + 1) * coverage)
    # n+1番目は無限大として扱う。小さなデータで高すぎる目標を有限の幅にしない。
    return float(np.sort(scores)[rank - 1]) if rank <= len(scores) else float("inf")


def interval_table(y, pred, radius: float) -> pd.DataFrame:
    prediction = np.asarray(pred)
    result = pd.DataFrame({"actual_man": y, "predicted_man": prediction,
                          "lower_man": prediction - radius, "upper_man": prediction + radius})
    result["covered"] = (result["actual_man"] >= result["lower_man"]) & (result["actual_man"] <= result["upper_man"])
    return result


def main() -> None:
    print_header("Chapter 26: 予想を幅で伝える")
    X_train, X_cal, X_test, y_train, y_cal, y_test = split_rooms()
    model = build_model().fit(X_train, y_train)
    residuals = np.abs(y_cal - model.predict(X_cal))
    radius = conformal_radius(residuals, coverage=0.9)
    pred = model.predict(X_test)
    result = interval_table(y_test, pred, radius)
    print("架空の家賃 / 学習60件・幅の調整30件・最終テスト30件")
    print(f"目標90%: ceil((30+1)×0.9)=28番目のずれ / 半幅={radius:.2f}万円 / 全幅={2 * radius:.2f}万円")
    print(f"最終テストのMAE: {mean_absolute_error(y_test, pred):.2f}万円")
    print(f"実際の家賃が幅に入った件数: {int(result.covered.sum())}/{len(result)}（{result.covered.mean():.1%}）")
    print(result.head(5).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("目標90%でも、今回の30件で必ず90%以上になるという意味ではありません。")
    print("同じ分布から来る調整用と新しい例、固定したモデルという前提があります。地域ごとや各部屋ごとの90%保証ではありません。")


if __name__ == "__main__":
    main()
