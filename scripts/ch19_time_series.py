from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import print_header


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["day_index", "weekday_sin", "weekday_cos", "lag_1", "lag_7"]


def make_sales() -> pd.Series:
    dates = pd.date_range("2025-01-01", periods=210, freq="D")
    rng = np.random.default_rng(42)
    angle = 2 * np.pi * dates.dayofweek.to_numpy() / 7
    # 日ごとのゆるやかな増加と、曜日の波と、偶然のずれを持つ架空の販売個数。
    counts = 100 + 0.18 * np.arange(len(dates)) + 18 * np.sin(angle) + 8 * np.cos(angle) + rng.normal(0, 6, len(dates))
    return pd.Series(np.rint(counts).astype(int), index=dates, name="sales")


def make_features(sales: pd.Series) -> pd.DataFrame:
    angle = 2 * np.pi * sales.index.dayofweek.to_numpy() / 7
    return pd.DataFrame({
        "day_index": np.arange(len(sales)), "weekday_sin": np.sin(angle), "weekday_cos": np.cos(angle),
        # 今日の予想には、昨日までにわかっている値だけを使う。
        "lag_1": sales.shift(1), "lag_7": sales.shift(7),
    }, index=sales.index).dropna()


def split_sales():
    sales = make_sales()
    X = make_features(sales)
    y = sales.loc[X.index]
    # 先頭7日は先週の値がない。残り203日から、最後の30日を残す。
    return X.iloc[:-30], X.iloc[-30:], y.iloc[:-30], y.iloc[-30:]


def build_model(alpha: float = 1) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("reg", Ridge(alpha=alpha))])


def chronological_folds(X_train):
    return list(TimeSeriesSplit(n_splits=3, test_size=30).split(X_train))


def prediction_table(X_test, y_test, pred) -> pd.DataFrame:
    return pd.DataFrame({"actual": y_test, "predicted": pred, "yesterday_baseline": X_test["lag_1"]}).rename_axis("date")


def main() -> None:
    print_header("Chapter 19: 時間の順番を守って予測する")
    X_train, X_test, y_train, y_test = split_sales()
    print(f"学習用{len(y_train)}日: {X_train.index.min():%m/%d}〜{X_train.index.max():%m/%d}")
    print(f"最終テスト{len(y_test)}日: {X_test.index.min():%m/%d}〜{X_test.index.max():%m/%d}")
    print("\n学習用の中で、過去から後の30日を確かめる3通りの分割:")
    for number, (train, valid) in enumerate(chronological_folds(X_train), start=1):
        print(f"{number}: 学習{len(train)}日（〜{X_train.index[train[-1]]:%m/%d}）→検証{len(valid)}日（{X_train.index[valid[0]]:%m/%d}〜）")
    model = build_model().fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"\n昨日と同じ個数を答える基準のMAE: {mean_absolute_error(y_test, X_test['lag_1']):.2f}個")
    print(f"モデルのMAE: {mean_absolute_error(y_test, pred):.2f}個")
    print(prediction_table(X_test, y_test, pred).head(5).to_string(float_format=lambda x: f"{x:.2f}"))
    print("毎日、前日までの実際の販売個数を受け取り、当日分を予想する評価です。モデルは固定します。")
    print("この30日分は、30日先まで一度に予想した成績ではありません。")


if __name__ == "__main__":
    main()
