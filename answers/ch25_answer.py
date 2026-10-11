from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ch13_csv_regression import ROOT
from ch25_probability_calibration import calibration_table, choose_model, compare_models, probability_metrics, split_events
from common import print_header


def main() -> None:
    print_header("Chapter 25 Answer: 件数と確率の意味を確認する")
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_events()
    # 問1: 校正法を追加。どちらも学習用の中のCVで校正し、別の検証用で選ぶ。
    models, table = compare_models(X_train, y_train, X_valid, y_valid, methods=("sigmoid", "isotonic"))
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    name = choose_model(table)
    model = models[name]
    # 問2: 5箱と10箱。箱を細かくすると件数も減り、割合が揺れやすい。
    tables = []
    for bins in [5, 10]:
        result = calibration_table(y_valid, model.predict_proba(X_valid)[:, 1], bins=bins)
        result.insert(0, "bins", bins)
        tables.append(result)
        print(result.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"選んだ{name}の最終テスト: {probability_metrics(y_test, model.predict_proba(X_test)[:, 1])}")
    # 問3: 検証用の校正表を保存し、個々の正しさとは区別する。
    output = ROOT / "outputs" / "probability_bins.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(tables, ignore_index=True).to_csv(output, index=False)
    print("80%の多数の例で約80%が陽性、という読み方です。ある一件が必ず陽性・陰性とわかる値ではありません。")


if __name__ == "__main__":
    main()
