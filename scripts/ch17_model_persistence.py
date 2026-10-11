from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from ch13_csv_regression import FEATURES, ROOT, TARGET, build_model, load_rent_data
from common import print_header


def train_rent_model():
    data = load_rent_data()
    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES], data[TARGET], test_size=0.25, random_state=42
    )
    return build_model().fit(X_train, y_train), X_test, y_test


def example_rooms() -> pd.DataFrame:
    return pd.DataFrame([
        {"area_sqm": 35, "walk_min": 8, "age_years": 10, "zone": "suburb"},
        {"area_sqm": 35, "walk_min": np.nan, "age_years": 10, "zone": "suburb"},
        {"area_sqm": 35, "walk_min": 8, "age_years": 10, "zone": "coastal"},
    ])


def save_model(model, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "rent_pipeline.joblib"
    # 回帰だけでなく、空欄の補完・標準化・文字の変換も一緒に保存する。
    joblib.dump(model, model_path)
    metadata = {
        "features": FEATURES, "target": TARGET, "target_unit": "万円/月",
        "versions": {
            "python": platform.python_version(), "scikit-learn": sklearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "joblib": joblib.__version__,
        },
    }
    (output_dir / "rent_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return model_path


def main(output_dir: Path | None = None) -> None:
    print_header("Chapter 17: モデルを保存して、あとで使う")
    model, X_test, y_test = train_rent_model()
    print(f"第13章と同じ学習用90件 / 最終テスト30件のMAE: {mean_absolute_error(y_test, model.predict(X_test)):.2f}万円")
    model_path = save_model(model, output_dir or ROOT / "outputs")
    # 自分で作ったファイルだけを読み込む。読み込み後にfitはしない。
    restored = joblib.load(model_path)
    rooms = example_rooms()
    before, after = model.predict(rooms), restored.predict(rooms)
    result = rooms.copy()
    result["保存前の予想"] = before
    result["読み込み後の予想"] = after
    print(result.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"予想は一致した？ {np.allclose(before, after)}")
    print(f"保存しました: {model_path}")
    print("入力の列・単位・道具のバージョンは、隣のrent_metadata.jsonで確認できます。")
    print("coastalは学習時にない地域です。処理できても、その地域で当たることは未確認です。")
    print("次に scripts/predict_saved_rent.py を実行すると、学び直さず予想できます。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="前処理を含む家賃モデルを保存して読み戻します。")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    main(parser.parse_args().output_dir)
