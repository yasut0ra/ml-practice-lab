from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import print_header


def make_inspection_data() -> tuple[pd.DataFrame, pd.Series]:
    X, y = make_classification(
        n_samples=1000, n_features=4, n_informative=3, n_redundant=0,
        weights=[0.95, 0.05], flip_y=0.01, class_sep=0.6, random_state=42,
    )
    # 測定値は人工的な数値。不良品の検査を想定した練習用の題材。
    features = ["measure_a", "measure_b", "measure_c", "measure_d"]
    return pd.DataFrame(X, columns=features), pd.Series(y, name="defective")


def split_inspection_data():
    X, y = make_inspection_data()
    X_rest, X_final, y_rest, y_final = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_rest, y_rest, test_size=0.25, random_state=42, stratify=y_rest
    )
    return X_train, X_valid, X_final, y_train, y_valid, y_final


def build_model() -> Pipeline:
    return Pipeline(steps=[("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])


def metrics_at_threshold(y, probabilities, threshold: float) -> dict:
    pred = np.asarray(probabilities) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "しきい値": threshold,
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def threshold_table(y_valid, probabilities, thresholds=None) -> pd.DataFrame:
    thresholds = [0.2, 0.5, 0.8] if thresholds is None else thresholds
    return pd.DataFrame([metrics_at_threshold(y_valid, probabilities, value) for value in thresholds])


def choose_threshold(result: pd.DataFrame) -> float:
    # 比較前に決めたルール：F1が最大。同率なら、しきい値が大きい方。
    return float(result.sort_values(["F1", "しきい値"], ascending=[False, False]).iloc[0]["しきい値"])


def main() -> None:
    print_header("Chapter 15: 少ない不良品を見つける")
    X_train, X_valid, X_final, y_train, y_valid, y_final = split_inspection_data()
    print(f"学習用{len(y_train)}件 / 設定選び用{len(y_valid)}件 / 最終テスト{len(y_final)}件")
    print(f"設定選び用の本当の不良品: {int(y_valid.sum())}件 / 割合: {y_valid.mean():.1%}")
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    baseline_pred = baseline.predict(X_valid)
    print(f"いつも正常と答える基準のaccuracy: {accuracy_score(y_valid, baseline_pred):.4f}")
    print(f"その基準のrecall: {recall_score(y_valid, baseline_pred, zero_division=0):.4f}")

    model = build_model().fit(X_train, y_train)
    probabilities = model.predict_proba(X_valid)[:, 1]
    result = threshold_table(y_valid, probabilities)
    print("\n設定選び用で、同じ予測確率のしきい値だけを変える:")
    print(result.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    threshold = choose_threshold(result)
    print(f"\n設定選び用のF1で選んだしきい値: {threshold:g}")
    # 学び直さず、同じモデルと選んだしきい値で最後に1回確認する。
    final = metrics_at_threshold(y_final, model.predict_proba(X_final)[:, 1], threshold)
    print("最終テスト:")
    print(pd.DataFrame([final]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("不良品は少ないので、正解率だけでなく、見逃しFNとrecallも読みます。")


if __name__ == "__main__":
    main()
