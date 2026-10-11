from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import print_header


def split_events():
    values, labels = make_classification(n_samples=1000, n_features=6, n_informative=2,
                    n_redundant=4, class_sep=0.8, flip_y=0.03, random_state=42)
    X, y = pd.DataFrame(values, columns=[f"measure_{i}" for i in range(6)]), pd.Series(labels, name="event")
    X_rest, X_test, y_rest, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_rest, y_rest, test_size=0.25, stratify=y_rest, random_state=42)
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def build_raw() -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("classifier", GaussianNB())])


def build_calibrated(method: str = "sigmoid"):
    # 学習用の中で、各行を学習しなかったモデルの確率を作って校正に使う。
    return CalibratedClassifierCV(estimator=build_raw(), method=method,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), ensemble=False)


def probability_metrics(y, probabilities) -> dict:
    return {"Brier": brier_score_loss(y, probabilities), "log_loss": log_loss(y, probabilities, labels=[0, 1]),
            "accuracy": accuracy_score(y, probabilities >= 0.5)}


def calibration_table(y, probabilities, bins: int = 5) -> pd.DataFrame:
    data = pd.DataFrame({"probability": np.asarray(probabilities), "actual": np.asarray(y)})
    data["bin"] = pd.cut(data["probability"], np.linspace(0, 1, bins + 1), include_lowest=True)
    # 空の箱には観測した割合がない。0%だと置き換えず、その行を省く。
    return data.groupby("bin", observed=True).agg(count=("actual", "size"),
            mean_probability=("probability", "mean"), observed_rate=("actual", "mean")).reset_index()


def compare_models(X_train, y_train, X_valid, y_valid, methods=("sigmoid",)):
    models = {"raw": build_raw(), **{method: build_calibrated(method) for method in methods}}
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        rows.append({"model": name, **probability_metrics(y_valid, model.predict_proba(X_valid)[:, 1])})
    return models, pd.DataFrame(rows)


def choose_model(table) -> str:
    return str(table.sort_values(["Brier", "model"]).iloc[0]["model"])


def main() -> None:
    print_header("Chapter 25: 80%という予想を確かめる")
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_events()
    print("架空の出来事0/1 / 学習600件・設定選び200件・最終テスト200件")
    models, table = compare_models(X_train, y_train, X_valid, y_valid)
    print("設定選び用の成績（Brierとlog_lossは小さいほど良い）:")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    name = choose_model(table)
    model = models[name]
    print("\n選んだモデルの、設定選び用の確率ごとの表:")
    print(calibration_table(y_valid, model.predict_proba(X_valid)[:, 1]).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n選んだ{name}の最終テスト: {probability_metrics(y_test, model.predict_proba(X_test)[:, 1])}")
    print("80%と予想した多数の例で、実際の出来事が約80%かを確かめます。一件の正しさの保証ではありません。")
    print("Brierには確率の校正以外の要因も関係します。表の件数と校正曲線も合わせて読みます。")


if __name__ == "__main__":
    main()
