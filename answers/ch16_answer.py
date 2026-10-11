from pathlib import Path
import sys

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ch13_csv_regression import ROOT, build_model
from ch16_feature_importance import importance_table, split_rooms
from common import print_header


def main() -> None:
    print_header("Chapter 16 Answer: 比較するモデルでも重要度は変わる")
    X_train, X_inspect, _, y_train, y_inspect, _ = split_rooms()
    models = {"Ridge": build_model(), "平均を答える基準": DummyRegressor(strategy="mean")}
    tables = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"\n{name} / 入れ替え前MAE: {mean_absolute_error(y_inspect, model.predict(X_inspect)):.2f}万円")
        # 問1: 10回と30回を比べる。モデルを学び直す回数ではない。
        repeats = [10, 30] if name == "Ridge" else [30]
        for count in repeats:
            result = importance_table(model, X_inspect, y_inspect, n_repeats=count)
            result.insert(0, "model", name)
            result.insert(1, "repeats", count)
            tables.append(result)
            print(result.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    # 問2: 基準は入力を使わず同じ平均を答えるので、どの列を入れ替えても成績は同じ。
    print("基準の重要度がすべて0なのは、手がかりを使っていないためです。手がかり自体が無意味とは言えません。")
    # 問3: 比較表を保存し、重要度から言える範囲を説明する。
    output = ROOT / "outputs" / "feature_importance.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(tables, ignore_index=True).to_csv(output, index=False)
    print(f"保存しました: {output}")
    print("このモデルと確認用での予測への寄与の目安です。重要な原因や、全データに共通する順位とは言えません。")


if __name__ == "__main__":
    main()
