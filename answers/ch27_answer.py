from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ch13_csv_regression import ROOT, build_model
from ch27_monitoring import monitoring_data, summarize_batch
from common import print_header


def main() -> None:
    print_header("Chapter 27 Answer: 答えがまだ来ていなかったら？")
    X_train, y_train, batches = monitoring_data()
    model = build_model().fit(X_train, y_train)
    reference_area = batches["reference"][0]["area_sqm"].mean()
    X, y = batches["relationship_shift"]
    # 問1: 答えがない時点では入力の変化だけを見る。MAEを0にしない。
    early = summarize_batch(model, X, None, reference_area, "labels_pending")
    # 問2: 一部・全部の答えが来たとき、採点した件数も一緒に確認する。
    partial = y.copy()
    partial.iloc[40:] = float("nan")
    later = summarize_batch(model, X, partial, reference_area, "40_labels")
    complete = summarize_batch(model, X, y, reference_area, "120_labels")
    table = pd.concat([early, later, complete], ignore_index=True)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    # 問3: モデルカードに、入力・単位・限界・監視後の対応を記録する。
    output = ROOT / "outputs" / "monitoring_label_delay.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print("答えが来た40件のMAEは、残り80件も同じという保証ではありません。欠け方と地域ごとの件数を確認します。")
    print("モデルカードには『架空の家賃の練習用』『相場判断には使わない』『悪化時は入力・正解・地域別のずれを確認』と記録できます。")


if __name__ == "__main__":
    main()
