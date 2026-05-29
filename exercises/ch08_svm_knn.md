# Chapter 8: SVM と k-NN

## 短い説明

SVM はクラスの境界を学習し、k-NN は近くのデータの多数決で分類します。
どちらも特徴量のスケールに影響を受けやすいため、標準化が重要です。

## 実行

```bash
python scripts/ch08_svm_knn.py
```

## 練習問題

1. SVM の `C` を `0.5`, `3`, `10` で比較してください。
2. k-NN の `n_neighbors` を `3`, `5`, `11` で比較してください。
3. 標準化を外すと accuracy が変わるか確認してください。

## ヒント

```python
Pipeline([("scale", StandardScaler()), ("clf", SVC(C=3))])
```

解答例は `answers/ch08_answer.py` にあります。
