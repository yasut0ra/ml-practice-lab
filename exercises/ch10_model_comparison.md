# Chapter 10: 簡単なモデル比較

## 短い説明

モデル比較では、同じデータ、同じ分割方法、同じ評価指標を使うことが重要です。
条件をそろえない比較は、モデルそのものの差なのか評価方法の差なのか分かりにくくなります。

## 実行

```bash
python scripts/ch10_model_comparison.py
```

## 練習問題

1. scoring に `recall` を追加してください。
2. F1 が最も高いモデル名を表示してください。
3. 評価結果を `outputs/model_comparison.csv` に保存してください。

## ヒント

```python
scores = cross_validate(model, X, y, scoring={"accuracy": "accuracy", "f1": "f1", "recall": "recall"})
result.to_csv("outputs/model_comparison.csv", index=False)
```

解答例は `answers/ch10_answer.py` にあります。
