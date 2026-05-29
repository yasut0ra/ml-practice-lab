# Chapter 3: 分類

## 短い説明

分類では、入力データがどのクラスに属するかを予測します。
Iris データセットでは、3 種類のアヤメの品種を予測します。

## 実行

```bash
python scripts/ch03_classification.py
```

## 練習問題

1. accuracy に加えて confusion matrix を表示してください。
2. `stratify=y` を外して実行し、テストデータのクラス分布がどう変わるか見てください。
3. `sepal length (cm)` と `sepal width (cm)` だけを使った場合の accuracy を確認してください。

## ヒント

```python
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, pred))
```

解答例は `answers/ch03_answer.py` にあります。
