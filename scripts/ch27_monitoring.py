from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from ch13_csv_regression import FEATURES, ROOT, build_model
from common import print_header


def make_market(seed: int, size: int, shifted: bool = False):
    rng = np.random.default_rng(seed)
    area = rng.uniform(35, 85, size) if shifted else rng.uniform(18, 70, size)
    walk, age = rng.integers(1, 26, size), rng.integers(0, 41, size)
    zone = rng.choice(["central", "suburb"], size=size)
    index = [f"{seed}-{i}" for i in range(size)]
    X = pd.DataFrame({"area_sqm": area, "walk_min": walk, "age_years": age, "zone": zone}, index=index)[FEATURES]
    rent = 3 + 0.17 * area - 0.08 * walk - 0.04 * age + 1.8 * (zone == "central") + rng.normal(0, 0.65, size)
    return X, pd.Series(rent, index=index, name="rent_man")


def monitoring_data():
    X_train, y_train = make_market(41, 300)
    X_reference, y_reference = make_market(42, 120)
    X_shift, y_shift = make_market(43, 120, shifted=True)
    # 後の二つは同じ入力を使う仮想の比較。二つ目だけ家賃の関係が2.5万円変わる。
    batches = {"reference": (X_reference, y_reference), "input_shift": (X_shift, y_shift),
               "relationship_shift": (X_shift, y_shift + 2.5)}
    return X_train, y_train, batches


def summarize_batch(model, X, y, reference_area: float, name: str) -> pd.DataFrame:
    # 答えがまだ来ていなければ、誤差を0とせず「未評価」にする。
    labels = pd.Series(np.nan, index=X.index) if y is None else y.reindex(X.index)
    pred = pd.Series(model.predict(X), index=X.index)
    rows = []
    for group, mask in [("all", pd.Series(True, index=X.index)),
                        ("central", X["zone"] == "central"), ("suburb", X["zone"] == "suburb")]:
        observed = mask & labels.notna()
        mae = mean_absolute_error(labels[observed], pred[observed]) if observed.any() else np.nan
        area_mean = X.loc[mask, "area_sqm"].mean()
        rows.append({"batch": name, "group": group, "n_rows": int(mask.sum()), "n_labeled": int(observed.sum()),
            "area_mean_sqm": area_mean, "MAE_man": mae,
            "input_alert": abs(area_mean - reference_area) > 10 if group == "all" else None,
            "error_alert": "未評価" if not observed.any() else ("確認" if mae > 1.5 else "基準内")})
    return pd.DataFrame(rows)


def main() -> None:
    print_header("Chapter 27: 使い始めたあとも確かめる")
    X_train, y_train, batches = monitoring_data()
    model = build_model().fit(X_train, y_train)
    reference_area = batches["reference"][0]["area_sqm"].mean()
    table = pd.concat([summarize_batch(model, X, y, reference_area, name) for name, (X, y) in batches.items()], ignore_index=True)
    print("架空の新しい3状況 / モデルは最初の300件で固定。後の二つは同じ入力を使う仮想の比較です。")
    print("事前に決めた説明用の目安: 平均の広さの差が10㎡を超える／MAEが1.5万円を超えたら確認")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    output = ROOT / "outputs" / "monitoring_report.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print("入力が変わっても、必ず精度が下がるとは限りません。実際の答えが来てから誤差も確かめます。")
    print("地域ごとの誤差と件数も見ます。同じMAEだけで公平性を判断できるわけではありません。")


if __name__ == "__main__":
    main()
