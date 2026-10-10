from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import print_header


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "datasets" / "rent_practice.csv"
NUMERIC = ["area_sqm", "walk_min", "age_years"]
FEATURES = [*NUMERIC, "zone"]
TARGET = "rent_man"


def load_rent_data(path: Path = DATA_PATH) -> pd.DataFrame:
    data = pd.read_csv(path)
    missing = set(["room_id", *FEATURES, TARGET]) - set(data.columns)
    if missing:
        raise ValueError(f"CSVに必要な列がありません: {', '.join(sorted(missing))}")
    for column in [*NUMERIC, TARGET]:
        data[column] = pd.to_numeric(data[column], errors="raise")
    if data[TARGET].isna().any():
        raise ValueError("答えの家賃に空欄があります。答えは補完せず、データを見直してください。")
    return data


def build_model(features: list[str] | None = None) -> Pipeline:
    features = FEATURES if features is None else features
    numeric = [column for column in NUMERIC if column in features]
    numeric_steps = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    transformers = [("num", numeric_steps, numeric)]
    if "zone" in features:
        category_steps = Pipeline(steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", category_steps, ["zone"]))
    return Pipeline(steps=[
        ("prepare", ColumnTransformer(transformers=transformers)),
        ("reg", Ridge(alpha=1)),
    ])


def prediction_table(data: pd.DataFrame, y_test: pd.Series, pred) -> pd.DataFrame:
    result = data.loc[y_test.index, ["room_id", TARGET]].copy()
    result["predicted_man"] = pred
    result["absolute_error_man"] = (result[TARGET] - result["predicted_man"]).abs()
    return result


def main() -> None:
    print_header("Chapter 13: CSVから家賃を予測する")
    data = load_rent_data()
    print(f"読み込んだ部屋: {len(data)}件（学習用の架空データ）")
    print(data.head(3).to_string(index=False))
    print("\n手がかりの空欄の数:")
    print(data[FEATURES].isna().sum().to_string())
    # 部屋の番号と、当てたい家賃を手がかりに入れない。
    X, y = data[FEATURES], data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = build_model()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
    print(f"\n学習用{len(y_train)}件 / 最終テスト{len(y_test)}件")
    print(f"平均を答える基準のMAE: {mean_absolute_error(y_test, baseline.predict(X_test)):.2f}万円")
    print(f"モデルのMAE: {mean_absolute_error(y_test, pred):.2f}万円")

    new_room = pd.DataFrame([{"area_sqm": 35, "walk_min": 8, "age_years": 10, "zone": "suburb"}])
    print(f"35㎡・駅から8分・築10年・suburbの予想: {model.predict(new_room)[0]:.2f}万円")
    print("新しい部屋の実際の家賃は未確認です。予想のずれはまだ計算できません。")
    output = ROOT / "outputs" / "rent_predictions.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    prediction_table(data, y_test, pred).to_csv(output, index=False)
    print(f"最終テストの答え・予測・ずれを保存しました: {output}")


if __name__ == "__main__":
    main()
