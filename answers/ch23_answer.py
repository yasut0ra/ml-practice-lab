from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ch13_csv_regression import ROOT
from ch23_neural_network import build_model, compare_widths, split_digits
from common import print_header


def main() -> None:
    print_header("Chapter 23 Answer: 中間のユニット数を比べる")
    X_train, X_test, y_train, y_test = split_digits()
    # 問1: 同じ外側のCVで16個と32個を比較。内部の早期終了も学習側だけで行う。
    table = compare_widths(X_train, y_train)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    width = int(table.sort_values(["CV_accuracy", "width"], ascending=[False, True]).iloc[0]["width"])
    model = build_model(width).fit(X_train, y_train)
    print(f"選んだ{width}個の最終正解率: {accuracy_score(y_test, model.predict(X_test)):.3f}")
    # 問2: 入力64→中間width→出力10の重みの数を数える。バイアスは別。
    network = model.named_steps["network"]
    print(f"重みの数: {sum(w.size for w in network.coefs_)} / バイアスの数: {sum(b.size for b in network.intercepts_)}")
    # 問3: 学習の記録と、内部検証の正解率を保存する。
    history = pd.DataFrame({"epoch": range(1, network.n_iter_ + 1), "training_loss": network.loss_curve_,
                            "internal_validation_accuracy": network.validation_scores_})
    output = ROOT / "outputs" / "neural_training_history.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(output, index=False)
    print("lossが下がることと、初めての例の正解率が伸びることは別です。内部検証の履歴は最終テストの履歴ではありません。")


if __name__ == "__main__":
    main()
