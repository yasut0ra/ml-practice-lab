"""第17章で自分が保存したモデルを、別の実行で使う。ここにはfitがない。"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from ch13_csv_regression import ROOT
from ch17_model_persistence import example_rooms
from common import print_header


def main(model_path: Path) -> None:
    if not model_path.is_file():
        raise SystemExit("モデルが見つかりません。先に第17章を実行して保存してください。")
    print_header("保存したモデルで家賃を予想する")
    # この教材で自分が保存したファイルを、同じPython環境で読み込む。
    model = joblib.load(model_path)
    rooms = example_rooms()
    result = rooms.copy()
    result["predicted_man"] = model.predict(rooms)
    print(result.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("学び直さず予想しました。新しい部屋の実際の家賃は未確認です。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="第17章で自分が保存したモデルを使います。")
    parser.add_argument("--model-path", type=Path, default=ROOT / "outputs" / "rent_pipeline.joblib")
    main(parser.parse_args().model_path)
