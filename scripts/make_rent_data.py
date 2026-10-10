"""第13章の架空CSVを再生成する。実在の物件情報は含まない。"""

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    rng = np.random.default_rng(42)
    count = 120
    area = rng.integers(18, 71, count)
    walk = rng.integers(1, 26, count)
    age = rng.integers(0, 41, count)
    zone = rng.choice(["central", "suburb"], count)
    # 説明用の規則 + 偶然のずれ。現実の家賃を計算する式ではない。
    rent = 3 + 0.17 * area - 0.08 * walk - 0.04 * age
    rent = rent + (zone == "central") * 1.8 + rng.normal(0, 0.65, count)
    data = pd.DataFrame({
        "room_id": [f"R{index:03d}" for index in range(1, count + 1)],
        "area_sqm": area.astype(float),
        "walk_min": walk.astype(float),
        "age_years": age.astype(float),
        "zone": zone,
        "rent_man": np.round(rent, 2),
    })
    # 手がかりの空欄だけを後から作る。答えは空欄にしない。
    for column, missing_count in [("area_sqm", 7), ("walk_min", 8), ("age_years", 6), ("zone", 5)]:
        data.loc[rng.choice(count, missing_count, replace=False), column] = np.nan
    output = Path(__file__).resolve().parents[1] / "datasets" / "rent_practice.csv"
    data.to_csv(output, index=False)
    print(f"架空の{count}件を保存しました: {output}")


if __name__ == "__main__":
    main()
