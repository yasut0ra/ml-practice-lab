from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from common import print_header
from sklearn.linear_model import LinearRegression


def main() -> None:
    print_header("Answer 0: 家賃の予想")
    model = LinearRegression().fit([[20], [40], [60]], [6, 10, 14])
    # 問1・2: 入力だけを変え、同じモデルで予想する。
    rooms = [[25], [45], [100]]
    for room, prediction in zip(rooms, model.predict(rooms)):
        print(f"{room[0]}㎡ → {prediction:.1f}万円")
    # 問3: 学習した範囲から大きく外れた予想は、特に確かめる必要がある。
    print("100㎡は学習した20〜60㎡の範囲外です。範囲外でも同じ関係が続くかは、このデータでは確かめられません。")


if __name__ == "__main__":
    main()
