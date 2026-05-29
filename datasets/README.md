# Datasets

この教材はローカルで確実に動かすため、外部ファイルではなく scikit-learn 付属データセットを中心に使います。

## 利用する主なデータセット

| データセット | 使う章 | 内容 |
| --- | --- | --- |
| `load_diabetes` | 1, 2 | 糖尿病の進行度を予測する回帰データ |
| `load_iris` | 3, 9 | アヤメの品種分類データ |
| `load_breast_cancer` | 4, 5, 7, 10 | 乳がん診断の二値分類データ |
| `load_digits` | 8 | 手書き数字画像の分類データ |
| `make_blobs` | 9 | クラスタリング用の人工データ |

## 自分の CSV を使う場合

`pandas.read_csv()` で読み込み、説明変数 `X` と目的変数 `y` に分けると、この教材の多くのコードを再利用できます。

```python
import pandas as pd

df = pd.read_csv("datasets/my_data.csv")
X = df.drop(columns=["target"])
y = df["target"]
```
