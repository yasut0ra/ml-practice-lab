# Datasets

この教材はローカルで確実に動かすため、外部ファイルではなく scikit-learn 付属データセットを中心に使います。

## 利用する主なデータセット

| データセット | 使う章 | 内容 |
| --- | --- | --- |
| `load_diabetes` | 1, 2 | 糖尿病の進行度を予測する回帰データ |
| `load_iris` | 3 | アヤメの品種分類データ |
| `load_breast_cancer` | 4, 5, 7, 10 | 乳がん診断の二値分類データ |
| `load_digits` | 8 | 手書き数字画像の分類データ |
| `make_blobs` | 9 | クラスタリング用の人工データ |

第6章は、コード内で作った16件の架空の顧客データを使います。

## 「1行」「1列」「答え」の意味

Iris を例にすると、1行は1つの花、4つの列は花の測定値、答えはその花の品種です。

```python
from sklearn.datasets import load_iris

data = load_iris(as_frame=True)
print(data.data.head(3))  # 最初の3つの花の測定値
print(data.target.head(3))  # 同じ3つの花の品種番号
print(data.target_names)  # 品種番号に対応する名前
```

`data.data` と `data.target` の同じ行番号が1組になります。
糖尿病データの特徴量は加工済みの尺度なので、BMI や血圧の普段の単位としては読みません。
診断データでは元の番号が「0＝悪性、1＝良性」です。
第4・5・7・10章では悪性を陽性として評価するため「1＝悪性、0＝良性」に付け直しています。

データの詳しい定義は[糖尿病データの公式説明](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html)を参照してください。

## 自分の CSV を使う場合

`pandas.read_csv()` で読み込み、説明変数 `X` と目的変数 `y` に分けると、この教材の多くのコードを再利用できます。

```python
import pandas as pd

df = pd.read_csv("datasets/my_data.csv")
X = df.drop(columns=["target"])
y = df["target"]
```
