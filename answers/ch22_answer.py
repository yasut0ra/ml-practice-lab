from pathlib import Path
import sys

from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ch13_csv_regression import ROOT
from ch22_gradient_boosting import build_boosting, compare_models, split_rooms
from common import print_header


def main() -> None:
    print_header("Chapter 22 Answer: 一度の修正を小さくすると？")
    X_train, X_test, y_train, y_test = split_rooms()
    # 問1: 回数と木の大きさを固定し、修正の割合だけ変える。
    models = {f"rate={rate}": build_boosting(learning_rate=rate) for rate in [0.05, 0.1, 0.2]}
    table = compare_models(X_train, y_train, models)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    # 問2: 同じCVのMAEで選び、最終テストは選んだ一つだけ。
    name = table.sort_values(["CV_MAE_man", "model"]).iloc[0]["model"]
    model = models[name].fit(X_train, y_train)
    print(f"{name}の最終MAE: {mean_absolute_error(y_test, model.predict(X_test)):.2f}万円")
    # 問3: 記録し、森との違いを説明する。
    output = ROOT / "outputs" / "boosting_rates.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print("森は複数の木の予想を平均します。ブースティングは、今の予想を直す小さな木を順番に足します。")


if __name__ == "__main__":
    main()
