from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ch13_csv_regression import ROOT, build_model
from ch26_prediction_intervals import conformal_radius, interval_table, split_rooms
from common import print_header


def main() -> None:
    print_header("Chapter 26 Answer: 幅と入る割合の目標を区別する")
    X_train, X_cal, X_test, y_train, y_cal, y_test = split_rooms()
    model = build_model().fit(X_train, y_train)
    residuals = np.abs(y_cal - model.predict(X_cal))
    # 問1: 同じ固定モデル・調整用で80%と90%の幅を作る。テストからは幅を選ばない。
    for target in [0.8, 0.9]:
        radius = conformal_radius(residuals, target)
        print(f"目標{target:.0%}: 半幅{radius:.2f}万円 / 全幅{2 * radius:.2f}万円")
    # 問2: 先に決めた90%だけを、最後のテストで一度確認。
    radius = conformal_radius(residuals, 0.9)
    result = interval_table(y_test, model.predict(X_test), radius)
    print(f"最終テストの実測: {int(result.covered.sum())}/{len(result)}が幅の中（{result.covered.mean():.1%}）")
    # 問3: 予想・上下限・答えが入ったかを保存し、保証の範囲を書く。
    output = ROOT / "outputs" / "rent_intervals.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index_label="source_row")
    print("同じ分布から来る調整用と新しい例を通した全体の目標です。各部屋・各地域や今回の30件すべてでの保証ではありません。")
    print(f"30件で99%を目標にすると半幅は{conformal_radius(residuals, 0.99)}。31番目のずれが必要なので有限の幅を作れません。")


if __name__ == "__main__":
    main()
