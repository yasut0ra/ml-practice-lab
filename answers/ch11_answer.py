from pathlib import Path
import sys

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ch11_parameter_search import build_search, comparison_table
from common import print_header


def main() -> None:
    print_header("Chapter 11 Answer: 候補を増やして、比較表を残す")
    data = load_iris(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.25, random_state=42, stratify=data.target
    )
    # 問1: Cに100を追加。4種類 × 2種類 = 8候補、8 × 5 = 40回のCV学習。
    search = build_search([0.1, 1, 10, 100]).fit(X_train, y_train)
    result = comparison_table(search)
    print(result.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"候補数: {len(result)} / CVでの学習回数: {len(result) * search.n_splits_}")
    print("これに加えて、選んだ設定で学習用全体を1回学び直しています。")

    # 問2: 最終テストは選ばれたモデルだけで評価する。
    print(f"選んだ設定: {search.best_params_}")
    print(f"最後の正解率: {accuracy_score(y_test, search.predict(X_test)):.4f}")
    print("Cが大きいほど良いとは限りません。候補を増やしても成績が同じになることがあります。")

    # 問3: 全候補のCV結果を保存。最終テストの候補別比較は作らない。
    output = Path(__file__).resolve().parents[1] / "outputs" / "parameter_search.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"保存しました: {output}")


if __name__ == "__main__":
    main()
