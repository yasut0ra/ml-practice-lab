from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ch13_csv_regression import FEATURES, ROOT, TARGET, build_model, load_rent_data, prediction_table
from common import print_header


def main() -> None:
    print_header("Chapter 13 Answer: 手がかりを比べ、ずれを調べる")
    data = load_rent_data()
    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES], data[TARGET], test_size=0.25, random_state=42
    )
    # 問2: 比較は学習用の中で行う。前処理を含めたPipelineをCVに渡す。
    models = {"2列": build_model(["area_sqm", "walk_min"]), "4列": build_model()}
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error")
        rows.append({"手がかり": name, "CVのMAE（万円）": -scores.mean()})
    result = pd.DataFrame(rows).sort_values(["CVのMAE（万円）", "手がかり"])
    print(result.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("neg_mean_absolute_errorはMAEにマイナスを付けた値なので、表示では符号を戻します。")
    best_name = result.iloc[0]["手がかり"]
    model = models[best_name].fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"選んだ手がかり: {best_name} / 最後のMAE: {mean_absolute_error(y_test, pred):.2f}万円")

    # 問1: 同じ広さの2部屋で、駅からの時間だけを変える。
    rooms = pd.DataFrame([
        {"area_sqm": 35, "walk_min": minutes, "age_years": 10, "zone": "suburb"}
        for minutes in [5, 20]
    ])
    rooms["predicted_man"] = model.predict(rooms)
    print("\n新しい部屋の予想:")
    print(rooms.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("これは架空データから学んだモデル内の関係で、実際の相場や原因を示しません。")

    # 問3: 最終テストのずれを大きい順に並べる。これを見てモデルを選び直さない。
    errors = prediction_table(data, y_test, pred).sort_values("absolute_error_man", ascending=False)
    print("\nずれが大きい5件:")
    print(errors.head(5).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    output = ROOT / "outputs" / "rent_errors.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(output, index=False)
    print(f"保存しました: {output}")


if __name__ == "__main__":
    main()
