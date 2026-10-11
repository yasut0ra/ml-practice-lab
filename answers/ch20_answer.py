from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ch13_csv_regression import ROOT
from ch20_feature_engineering import build_model, choose_degree, compare_degrees, split_energy
from common import print_header


def main() -> None:
    print_header("Chapter 20 Answer: 列を増やせば必ず良い？")
    X_train, X_test, y_train, y_test = split_energy()
    # 問1: 気温だけから作った列を見比べる。答えを加工して入力にしない。
    example = pd.DataFrame({"temperature_c": [0, 10, 20]})
    poly = build_model(3).named_steps["features"]
    print(pd.DataFrame(poly.fit_transform(example), columns=poly.get_feature_names_out()).to_string(index=False))
    # 問2: 4次を加え、同じ学習用CVで選ぶ。
    table = compare_degrees(X_train, y_train, degrees=(1, 2, 3, 4))
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    degree = choose_degree(table)
    model = build_model(degree).fit(X_train, y_train)
    print(f"選んだ{degree}次の最終MAE: {mean_absolute_error(y_test, model.predict(X_test)):.3f}kWh")
    # 問3: 比較表を残し、予測できる入力と答えを区別する。
    output = ROOT / "outputs" / "feature_degrees.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print("気温の2乗は予測時に計算できます。当日の実際の使用電力量を手がかりにすると、答えの漏洩です。")


if __name__ == "__main__":
    main()
