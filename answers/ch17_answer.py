from pathlib import Path
import sys

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ch13_csv_regression import ROOT
from ch17_model_persistence import example_rooms, save_model, train_rent_model
from common import print_header


def main() -> None:
    print_header("Chapter 17 Answer: 前処理も保存されている？")
    model, _, _ = train_rent_model()
    # 単独で解答を実行しても動くよう、自分の練習用モデルをここで作る。
    output_dir = ROOT / "outputs" / "answer17"
    model_path = save_model(model, output_dir)
    restored = joblib.load(model_path)
    # 問1: 元のPipelineと、読み戻したPipelineを同じ入力で比べる。
    rooms = example_rooms()
    before, after = model.predict(rooms), restored.predict(rooms)
    print(f"3件とも保存前後で一致: {np.allclose(before, after)}")
    # 問2: 2行目は数値の空欄、3行目は未知の地域。保存済みの前処理が扱う。
    result = rooms.copy()
    result["before_man"], result["after_man"] = before, after
    print(result.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("未知の地域は学習済みの地域の列がすべて0になります。その地域の相場は学んでいません。")
    # 問3: 単位・列名・環境の記録と、実際の家賃がないことを確認する。
    print((output_dir / "rent_metadata.json").read_text(encoding="utf-8"))
    result.to_csv(output_dir / "rent_roundtrip.csv", index=False)
    print("保存前後の一致は、保存ができた確認です。新しい部屋のMAEは正解の家賃がないため測れません。")


if __name__ == "__main__":
    main()
