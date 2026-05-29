# Chapter 6: 前処理

## 短い説明

実データには欠損値、数値特徴量、カテゴリ特徴量が混ざります。
`Pipeline` と `ColumnTransformer` を使うと、前処理とモデルをまとめて扱えます。

## 実行

```bash
python scripts/ch06_preprocessing.py
```

## 練習問題

1. 新しい顧客データを 1 件作り、アップグレード確率を予測してください。
2. 数値特徴量の欠損値補完を `mean` に変更して結果を確認してください。
3. one-hot encoding 後の特徴量名を表示してください。

## ヒント

```python
model.predict_proba(new_customer)[:, 1]
model.named_steps["preprocess"].get_feature_names_out()
```

解答例は `answers/ch06_answer.py` にあります。
