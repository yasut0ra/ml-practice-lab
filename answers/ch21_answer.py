from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ch13_csv_regression import ROOT
from ch21_group_validation import group_folds, split_comparison, split_students
from common import print_header


def main() -> None:
    print_header("Chapter 21 Answer: 行の件数と人数を区別する")
    train, final = split_students()
    # 問1: 最終テストと、CVの各回で人が重ならないことを調べる。
    print(f"学習用と最終テストの共通の人: {set(train.student_id) & set(final.student_id)}")
    for a, b in group_folds(train):
        shared = set(train.iloc[a].student_id) & set(train.iloc[b].student_id)
        print(f"CV: 学習{len(a)}行 / 検証{len(b)}行 / 共通の人数{len(shared)}")
    # 問2: 行を混ぜる評価と、人ごとの評価を、学習用の中で診断する。
    table = split_comparison(train)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    # 問3: 誤った分割の良い成績でモデルを選ばず、比較の目的を説明する。
    output = ROOT / "outputs" / "group_split_comparison.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print("行を混ぜると同じ人のプロフィールをすでに見た成績になります。新しい人の予想には、人ごとの評価を使います。")
    print("この解答は分割の診断だけです。最後の人の点数で設定を選び直しません。")


if __name__ == "__main__":
    main()
