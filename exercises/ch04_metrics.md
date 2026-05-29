# Chapter 4: 評価指標

## 短い説明

分類の評価では、accuracy だけでは不十分なことがあります。
見逃しを減らしたい場合は recall、誤検知を減らしたい場合は precision を重視します。

## 実行

```bash
python scripts/ch04_metrics.py
```

## 練習問題

1. 予測確率に対してしきい値 `0.3`, `0.5`, `0.7` を試してください。
2. 各しきい値で precision、recall、F1 を表示してください。
3. 悪性の見逃しを減らしたい場合、どのしきい値がよさそうか考えてください。

## ヒント

```python
pred = (prob >= threshold).astype(int)
```

解答例は `answers/ch04_answer.py` にあります。
