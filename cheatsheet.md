# scikit-learn Cheatsheet

## 基本の流れ

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)
pred = model.predict(X_test)
print(accuracy_score(y_test, pred))
```

## よく使う import

```python
from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
```

## 教師あり学習

| 種類 | 目的 | 例 |
| --- | --- | --- |
| 回帰 | 数値を予測する | 価格、気温、売上 |
| 分類 | ラベルを予測する | 良性/悪性、品種、合否 |

## 教師なし学習

| 種類 | 目的 | 例 |
| --- | --- | --- |
| クラスタリング | 似ているデータをグループ化する | 顧客分類、文書分類 |
| 次元削減 | 特徴量を少ない軸に圧縮する | 可視化、前処理 |

## 評価指標

| タスク | 指標 | 意味 |
| --- | --- | --- |
| 回帰 | MAE | 誤差の絶対値の平均 |
| 回帰 | RMSE | 大きい誤差を重く見る |
| 回帰 | R2 | 1 に近いほど説明できている |
| 分類 | accuracy | 全体の正解率 |
| 分類 | precision | 陽性と予測した中で本当に陽性の割合 |
| 分類 | recall | 本当の陽性をどれだけ拾えたか |
| 分類 | F1 | precision と recall のバランス |
| クラスタリング | silhouette | クラスタのまとまりと分離の良さ |

## train/test split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,  # 分類ではクラス比率を保ちたいときに使う
)
```

## cross validation

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print(scores.mean(), scores.std())
```

## Pipeline

前処理とモデルをまとめると、train/test split や cross validation でデータ漏洩を避けやすくなります。

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = Pipeline(
    steps=[
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ]
)
```

## モデル選びの目安

| モデル | 向いている場面 | 注意点 |
| --- | --- | --- |
| LinearRegression | 回帰の最初の基準 | 非線形関係は苦手 |
| LogisticRegression | 分類の基準モデル | スケーリングが効く |
| DecisionTree | 解釈しやすい | 過学習しやすい |
| RandomForest | 強い基準モデル | 解釈性は少し落ちる |
| SVM | 小〜中規模データで強い | スケーリングが重要 |
| k-NN | 直感的で簡単 | データ数が増えると遅い |
| KMeans | クラスタリングの基準 | クラスタ数を指定する必要がある |
