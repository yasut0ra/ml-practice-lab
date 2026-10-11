from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ch13_csv_regression import ROOT
from ch24_anomaly_detection import anomaly_metrics, choose_candidate, compare_candidates, split_inspections
from common import print_header


def main() -> None:
    print_header("Chapter 24 Answer: 異常候補と故障を区別する")
    X_train, X_valid, X_test, y_valid, y_test = split_inspections()
    # 問1: 候補0.15を追加。学習は入力だけ、設定選びはラベル付き検証用。
    models, table = compare_candidates(X_train, X_valid, y_valid, candidates=(0.02, 0.05, 0.1, 0.15))
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    value = choose_candidate(table)
    model = models[value]
    print(f"選んだ{value}の最終テスト: {anomaly_metrics(y_test, model.predict(X_test))}")
    # 問2: 異常っぽい順に並べる。decision_functionは小さいほど異常側。
    ranked = X_valid.copy()
    ranked["decision_score"] = model.decision_function(X_valid)
    ranked["actual"] = y_valid
    ranked = ranked.sort_values("decision_score")
    print(ranked.head(5).to_string(float_format=lambda x: f"{x:.3f}"))
    # 問3: 検証用の候補表と、確認したい理由を残す。
    output = ROOT / "outputs" / "anomaly_candidates.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print("センサーの入力ミス、普段と違う運転、故障などを人が確認します。珍しさの数値は、故障確率ではありません。")


if __name__ == "__main__":
    main()
