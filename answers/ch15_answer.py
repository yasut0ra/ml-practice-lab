from pathlib import Path
import sys

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ch15_imbalanced_classification import build_model, choose_threshold, metrics_at_threshold, split_inspection_data, threshold_table
from common import print_header


def main() -> None:
    print_header("Chapter 15 Answer: 正解率と見逃しを分けて読む")
    X_train, X_valid, X_final, y_train, y_valid, y_final = split_inspection_data()
    # 問1: 正常とだけ答えれば、正解率は高くても不良品は1件も拾えない。
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    pred = baseline.predict(X_valid)
    print(f"基準のaccuracy: {accuracy_score(y_valid, pred):.4f} / recall: {recall_score(y_valid, pred, zero_division=0):.4f}")
    print(f"不良品{int(y_valid.sum())}件をすべて見逃しても、正常な品が多いので正解率は高くなります。")

    # 問2: 確率はそのまま、しきい値に0.1・0.3を追加する。
    model = build_model().fit(X_train, y_train)
    result = threshold_table(y_valid, model.predict_proba(X_valid)[:, 1], thresholds=[0.1, 0.2, 0.3, 0.5, 0.8])
    print(result.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    threshold = choose_threshold(result)
    final = metrics_at_threshold(y_final, model.predict_proba(X_final)[:, 1], threshold)
    print("選んだしきい値の最終テスト:")
    print(pd.DataFrame([final]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("しきい値を下げると見逃しは増えませんが、誤検知は減りません。F1の最大値は用途の正解とは限りません。")

    # 問3: 候補の検証結果を保存し、最終テストでの候補比較は作らない。
    output = Path(__file__).resolve().parents[1] / "outputs" / "threshold_comparison.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"設定選び用の比較表を保存しました: {output}")


if __name__ == "__main__":
    main()
