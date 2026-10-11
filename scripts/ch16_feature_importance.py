from __future__ import annotations

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from ch13_csv_regression import FEATURES, TARGET, build_model, load_rent_data
from common import print_header


def split_rooms():
    data = load_rent_data()
    X_rest, X_final, y_rest, y_final = train_test_split(
        data[FEATURES], data[TARGET], test_size=0.25, random_state=42
    )
    X_train, X_inspect, y_train, y_inspect = train_test_split(
        X_rest, y_rest, test_size=0.25, random_state=42
    )
    return X_train, X_inspect, X_final, y_train, y_inspect, y_final


def importance_table(model, X_inspect, y_inspect, n_repeats: int = 10) -> pd.DataFrame:
    # 元の4列を1列ずつ入れ替える。前処理も含む学習済みPipelineを使う。
    result = permutation_importance(
        model, X_inspect, y_inspect, scoring="neg_mean_absolute_error",
        n_repeats=n_repeats, random_state=42,
    )
    # 採点値が -MAE なので、importanceは「入れ替え後MAE - 元のMAE」。
    return pd.DataFrame({
        "feature": X_inspect.columns,
        "MAE増加の平均": result.importances_mean,
        "ばらつき": result.importances_std,
    }).sort_values(["MAE増加の平均", "feature"], ascending=[False, True])


def main() -> None:
    print_header("Chapter 16: モデルが使う手がかりを調べる")
    X_train, X_inspect, X_final, y_train, y_inspect, _ = split_rooms()
    model = build_model().fit(X_train, y_train)
    original_mae = mean_absolute_error(y_inspect, model.predict(X_inspect))
    baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
    print(f"学習用{len(y_train)}件 / 調べるための確認用{len(y_inspect)}件 / 最終テスト{len(X_final)}件")
    print(f"入れ替える前の確認用MAE: {original_mae:.2f}万円")
    print(f"平均を答える基準の確認用MAE: {mean_absolute_error(y_inspect, baseline.predict(X_inspect)):.2f}万円")
    print(importance_table(model, X_inspect, y_inspect).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("1列の値を別の部屋と入れ替え、予測のずれがどれだけ増えるかを測りました。")
    print("大きい値は、このモデルがこのデータで使っている手がかりの目安です。原因の証明ではありません。")
    print("調べた結果を使って改良する可能性があるので、最終テストはまだ使いません。")


if __name__ == "__main__":
    main()
