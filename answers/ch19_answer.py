from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ch19_time_series import ROOT, build_model, chronological_folds, prediction_table, split_sales
from common import print_header


def main() -> None:
    print_header("Chapter 19 Answer: 過去から先へ設定を比べる")
    X_train, X_test, y_train, y_test = split_sales()
    # 問1: 同じ時間順の3分割で、設定を比べる。
    folds = chronological_folds(X_train)
    for train, valid in folds:
        print(f"学習の最後{X_train.index[train[-1]]:%Y-%m-%d} < 検証の最初{X_train.index[valid[0]]:%Y-%m-%d}")
    candidates = [0.1, 1, 10]
    rows = []
    for alpha in candidates:
        scores = cross_val_score(build_model(alpha), X_train, y_train, cv=folds, scoring="neg_mean_absolute_error", error_score="raise")
        rows.append({"alpha": alpha, "CV_MAE": -scores.mean()})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    chosen = candidates[table["CV_MAE"].idxmin()]
    model = build_model(chosen).fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"選んだalpha={chosen} / 最終テストのMAE: {mean_absolute_error(y_test, pred):.2f}個")
    # 問2: モデルと二つの単純な基準を、同じ30日で比べる。
    print(f"昨日と同じ: {mean_absolute_error(y_test, X_test['lag_1']):.2f}個")
    print(f"先週と同じ: {mean_absolute_error(y_test, X_test['lag_7']):.2f}個")
    # 問3: 日付・答え・予想を保存し、何日先の予想なのかを書く。
    result = prediction_table(X_test, y_test, pred)
    result["last_week_baseline"] = X_test["lag_7"]
    output = ROOT / "outputs" / "sales_predictions.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output)
    print(f"保存しました: {output}")
    print("毎日、昨日までの実測値で次の1日を予想しました。30日先まで一度に予測する場合、途中の実測値は使えません。")


if __name__ == "__main__":
    main()
