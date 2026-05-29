# Chapter 5: train/test split と cross validation

## 短い説明

train/test split は一度だけデータを分けて評価します。
cross validation は複数回に分けて評価するため、モデルの安定性を見やすくなります。

## 実行

```bash
python scripts/ch05_split_cv.py
```

## 練習問題

1. `n_splits=3`, `5`, `10` で cross validation を比較してください。
2. accuracy だけでなく F1 でも cross validation を実行してください。
3. 平均値だけでなく標準偏差も見て、評価のばらつきを確認してください。

## ヒント

```python
cross_val_score(model, X, y, cv=cv, scoring="f1")
```

解答例は `answers/ch05_answer.py` にあります。
