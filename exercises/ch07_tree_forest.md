# Chapter 7: 決定木・ランダムフォレスト

## 短い説明

決定木は分岐条件を学習するため、直感的に理解しやすいモデルです。
ただし深すぎる木は過学習しやすいため、`max_depth` などで制御します。

## 実行

```bash
python scripts/ch07_tree_forest.py
```

## 練習問題

1. `DecisionTreeClassifier(max_depth=2, 4, None)` を比較してください。
2. `RandomForestClassifier(n_estimators=50, 150)` を比較してください。
3. feature importance の top 10 を表示してください。

## ヒント

```python
model.feature_importances_
```

解答例は `answers/ch07_answer.py` にあります。
