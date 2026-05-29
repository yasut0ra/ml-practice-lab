# Chapter 2: 回帰

## 短い説明

回帰では、数値を予測します。予測の良し悪しは MAE、RMSE、R2 などで確認します。
Ridge は係数が大きくなりすぎることを抑える線形回帰です。

## 実行

```bash
python scripts/ch02_regression.py
```

## 練習問題

1. `Ridge(alpha=0.1)`, `Ridge(alpha=1.0)`, `Ridge(alpha=10.0)` を比較してください。
2. 各モデルの MAE、RMSE、R2 を表形式で表示してください。
3. `test_size=0.3` に変えたとき、結果がどのくらい変わるか確認してください。

## ヒント

```python
for alpha in [0.1, 1.0, 10.0]:
    model = Ridge(alpha=alpha)
```

解答例は `answers/ch02_answer.py` にあります。
