# Chapter 9: クラスタリング

## 短い説明

クラスタリングは、正解ラベルを使わずに似ているデータをグループ化します。
KMeans ではクラスタ数 `n_clusters` を指定する必要があります。

## 実行

```bash
python scripts/ch09_clustering.py
```

## 練習問題

1. `n_clusters=2`, `3`, `4`, `5` を比較してください。
2. 各クラスタ数の silhouette score を表示してください。
3. true label がある人工データなので、adjusted rand score も確認してください。

## ヒント

```python
KMeans(n_clusters=k, n_init=10, random_state=42)
```

解答例は `answers/ch09_answer.py` にあります。
