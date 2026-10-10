from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

import pandas as pd
from ch10_model_comparison import build_models
from common import print_header, print_metrics
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split


def main() -> None:
    print_header("Answer 10: モデル比較")

    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = (data.target == 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    models = build_models()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for name, model in models.items():
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring={"accuracy": "accuracy", "f1": "f1", "recall": "recall"},
        )
        rows.append(
            {
                "model": name,
                "accuracy_mean": scores["test_accuracy"].mean(),
                "f1_mean": scores["test_f1"].mean(),
                "recall_mean": scores["test_recall"].mean(),
            }
        )

    # 問2: 選ぶ基準はF1の平均とし、同率なら名前順で決める。
    result = pd.DataFrame(rows).sort_values(["f1_mean", "model"], ascending=[False, True])
    best_name = result.iloc[0]["model"]
    best_model = models[best_name]
    best_model.fit(X_train, y_train)
    final_pred = best_model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, final_pred, labels=[0, 1]).ravel()
    final_metrics = {
        "model": best_name,
        "accuracy": accuracy_score(y_test, final_pred),
        "precision": precision_score(y_test, final_pred, zero_division=0),
        "recall": recall_score(y_test, final_pred, zero_division=0),
        "f1": f1_score(y_test, final_pred, zero_division=0),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }

    # 問3と総合課題: 比較表と、選んだ1モデルの最終テストを別の表にする。
    output_dir = Path(__file__).resolve().parents[1] / "outputs"
    output_dir.mkdir(exist_ok=True)
    result.to_csv(output_dir / "model_comparison.csv", index=False)
    pd.DataFrame([final_metrics]).to_csv(output_dir / "final_test_metrics.csv", index=False)

    print(result.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"CVのF1で選んだモデル: {best_name}")
    print("最終テスト:")
    print_metrics({name: final_metrics[name] for name in ["accuracy", "precision", "recall", "f1"]})
    print(f"件数: TN={tn} / 誤検知FP={fp} / 見逃しFN={fn} / TP={tp}")
    print("saved: outputs/model_comparison.csv")
    print("saved: outputs/final_test_metrics.csv")


if __name__ == "__main__":
    main()
