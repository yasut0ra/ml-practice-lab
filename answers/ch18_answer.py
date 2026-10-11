from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ch13_csv_regression import ROOT
from ch18_text_classification import build_model, prediction_table, split_messages
from common import print_header


def main() -> None:
    print_header("Chapter 18 Answer: 文字の切り方を比べる")
    X_train, X_test, y_train, y_test = split_messages()
    folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=42).split(X_train, y_train))
    # 問1: 文字の辞書と重みも、各分割の学習用だけから作る。
    candidates = [(1, 1), (2, 3)]
    rows = []
    for lengths in candidates:
        scores = cross_val_score(build_model(lengths), X_train, y_train, cv=folds, scoring="f1_macro", error_score="raise")
        rows.append({"min_chars": lengths[0], "max_chars": lengths[1], "CV_macro_F1": scores.mean()})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    # 同点なら先の候補。最終テストの成績では選ばない。
    chosen = candidates[table["CV_macro_F1"].idxmax()]
    model = build_model(chosen).fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"選んだ文字数{chosen}: 最終テストの正解率={accuracy_score(y_test, pred):.3f}, macro F1={f1_score(y_test, pred, average='macro'):.3f}")
    # 問2: 答え・予想・文章を並べて保存し、間違えた行を見る。
    result = prediction_table(X_test, y_test, pred)
    print("\n間違えた文章:")
    print(result[result["actual"] != result["predicted"]].to_string(index=False))
    output = ROOT / "outputs" / "text_predictions.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    # 問3: 二つの話題を含む文でも、一つのラベルを強制的に答える。
    mixed = pd.Series(["送料を返金してほしいです", "届いた商品の請求金額が違います"])
    print(pd.DataFrame({"text": mixed, "predicted": model.predict(mixed)}).to_string(index=False))
    print("一つの分類結果だけでは判断が難しい文です。両方の担当で確認する運用や、複数ラベルの課題を考えます。")


if __name__ == "__main__":
    main()
