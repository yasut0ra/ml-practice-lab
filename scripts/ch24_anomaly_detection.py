from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from common import print_header


FEATURES = ["sensor_a", "sensor_b"]


def make_inspections(seed: int, normals: int, anomalies: int):
    rng = np.random.default_rng(seed)
    usual = rng.normal(0, 1, (normals, 2))
    unusual = rng.uniform(2.5, 5, (anomalies, 2))
    index = [f"{seed}-{i}" for i in range(normals + anomalies)]
    X = pd.DataFrame(np.vstack([usual, unusual]), columns=FEATURES, index=index)
    y = pd.Series([0] * normals + [1] * anomalies, index=index, name="is_anomaly")
    return X, y


def split_inspections():
    X_train, _ = make_inspections(41, 240, 0)
    X_valid, y_valid = make_inspections(42, 80, 20)
    X_test, y_test = make_inspections(43, 80, 20)
    return X_train, X_valid, X_test, y_valid, y_test


def build_model(contamination: float = 0.05):
    return IsolationForest(n_estimators=80, contamination=contamination, random_state=42)


def anomaly_metrics(y, predictions) -> dict:
    flagged = (predictions == -1).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, flagged, labels=[0, 1]).ravel()
    return {"precision": precision_score(y, flagged, zero_division=0),
            "recall": recall_score(y, flagged, zero_division=0), "F1": f1_score(y, flagged, zero_division=0),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}


def compare_candidates(X_train, X_valid, y_valid, candidates=(0.02, 0.05, 0.1)):
    models, rows = {}, []
    for value in candidates:
        # 正常期間の入力だけで学ぶ。異常の正解ラベルは渡さない。
        model = build_model(value).fit(X_train)
        models[value] = model
        rows.append({"contamination": value, **anomaly_metrics(y_valid, model.predict(X_valid))})
    return models, pd.DataFrame(rows)


def choose_candidate(table) -> float:
    return float(table.sort_values(["F1", "contamination"], ascending=[False, True]).iloc[0]["contamination"])


def main() -> None:
    print_header("Chapter 24: いつもと違う記録を見つける")
    X_train, X_valid, X_test, y_valid, y_test = split_inspections()
    print("正常期間の学習240件 / 設定選び100件 / 最終テスト100件（架空の測定値）")
    models, table = compare_candidates(X_train, X_valid, y_valid)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    value = choose_candidate(table)
    print(f"選んだcontamination={value} / 最終テスト: {anomaly_metrics(y_test, models[value].predict(X_test))}")
    print("IsolationForestの予想は1=通常側、-1=異常候補です。正解の0/1とは別の番号です。")
    print("珍しいことと故障していることは別です。異常候補は、追加で確認する入口として使います。")


if __name__ == "__main__":
    main()
