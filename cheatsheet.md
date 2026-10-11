# scikit-learn Cheatsheet

迷ったときに「この言葉・この行は何をしている？」を確認するページです。
初めてなら[はじめてのガイド](docs/getting-started.md)から読んでください。

## 用語を日本語で読む

| コード・用語 | かんたんな意味 |
| --- | --- |
| `X` / 特徴量 | 予想の手がかりの表。行は1件、列は1種類の手がかり |
| `y` / 目的変数 | 当てたい答え。`X` と同じ行が対応する |
| `model` / モデル | 例から学んで予想する仕組み |
| `fit` / 学習 | 入力と答えから規則を見つける |
| `predict` / 予測 | 入力から答えを予想する |
| `train` | 学習に使うデータ |
| `test` | 最後に実力を確かめるためのデータ |
| `validation` | モデルや設定を選ぶ途中で確認するためのデータ |
| `pred` | モデルが出した予測値 |
| `random_state=42` | 乱数の出発点を固定する。42自体に特別な意味はない |
| `stratify=y` | 学習用と確認用のクラスの割合を近くする |
| `max_iter` | 計算の繰り返し回数の上限。データの件数ではない |
| 過学習 | 練習した例に合わせすぎて、新しい例でうまくいかないこと |
| データ漏洩 | 確認用の情報を、学習や設定選びに先に使ってしまうこと |
| ベースライン | 複雑なモデルと比べるための、単純な予想の基準 |
| ハイパーパラメータ | モデルが学ぶ前に、人が候補を決める設定 |
| GridSearchCV | 用意した設定の組み合わせを、同じCVで比べる道具 |
| PCA / 主成分分析 | 特徴量を組み合わせ、広がりを残せる少ない軸にまとめる方法 |
| 寄与率 | PCAの各軸に残した、学習用のばらつきの割合。正解率とは別 |
| CSV | 表を、列の間をカンマで区切って保存したファイル |
| 学習曲線 | 学習する件数を変え、学習用と検証用の成績を比べる図 |
| 不均衡な分類 | 種類ごとの件数が大きく違う分類。正解率だけでは読み違えやすい |
| しきい値 | 予測確率がこの値以上なら陽性と判断する、という境目 |
| Permutation Importance | 学習済みモデルで1列を入れ替え、成績の悪化を測る重要度 |
| モデルの保存 | 学習済みの仕組みをファイルに残し、あとで読み込んで使うこと |
| n-gram | 文章を、重なりのあるn文字などのかたまりに切ったもの |
| TF-IDF | 文章内の出現回数と、多くの文章に共通するかを使って付ける重み |
| macro F1 | 種類ごとのF1を、同じ重みで平均した値 |
| 時系列 | 日付や時刻の順番を持つ記録 |
| ラグ | 昨日や先週など、過去の値を今日の手がかりにした列 |
| TimeSeriesSplit | 過去で学び、その後の期間で確かめるためのCVの分割 |

## 基本の流れ

以下は、そのまま実行できる分類の最小例です。

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_iris(return_X_y=True)  # 手がかりと答えを用意する
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)  # 学習用で学ぶ
pred = model.predict(X_test)  # 確認用の入力だけで予想する
print(accuracy_score(y_test, pred))  # 取っておいた答えと比べる
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
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve
from sklearn.inspection import permutation_importance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import TimeSeriesSplit
import joblib
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
| 回帰 | MAE | ずれの大きさの平均。小さいほど良い |
| 回帰 | RMSE | 大きいずれを重く見る。小さいほど良い |
| 回帰 | R2 | 1に近いほど良い。0は平均を答えるのと同程度で、負にもなる |
| 分類 | accuracy | 全体の正解率。0.9なら90% |
| 分類 | precision | 陽性と予測した中で本当に陽性の割合 |
| 分類 | recall | 本当の陽性をどれだけ拾えたか |
| 分類 | F1 | precision と recall の調和平均。高いほど良い |
| 分類 | macro F1 | 各種類を対象にしたF1の単純平均。高いほど良い |
| 分類 | ROC-AUC | 陽性を陰性より高いスコアにする力。1に近いほど良く、ランダムなら約0.5 |
| クラスタリング | silhouette | 同じグループで近く、別グループから離れているか。-1〜1 |

「陽性」は見つけたい対象です。[第4章の4マスの図](exercises/ch04_metrics.md)で、
precision と recall の分母が違うことを確認できます。

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

確認する部分を入れ替えて、毎回新しく学習します。[第5章の図解](exercises/ch05_split_cv.md)も参照してください。
モデルや設定を比べるときは、最終テストを取っておき、学習用の中で CV を行います。
以下は上の例の続きです。

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
print(scores.mean(), scores.std())
```

## Pipeline

**前処理 → モデル**の順番をひとまとめにする仕組みです。
`fit(X_train, y_train)` なら、平均などの前処理の基準も学習用だけから求めます。
`predict(X_test)` では、その基準を使って変換してから予想します。
前処理を学習用と確認用で混ぜないためにも役立ちます。

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

## モデルより先に、単純な基準を作る

回帰なら「学習用の答えの平均をいつも答える」、分類なら「学習用で最も多い種類をいつも答える」と比べます。
複雑なモデルの数値だけでは、良い予想か判断しにくいためです。

```python
from sklearn.dummy import DummyRegressor, DummyClassifier

regression_baseline = DummyRegressor(strategy="mean")
classification_baseline = DummyClassifier(strategy="most_frequent")
```

基準も学習用だけで `fit` し、同じ確認用で評価します。[第2章](exercises/ch02_regression.md)と[第3章](exercises/ch03_classification.md)に実行例があります。
R2の説明でいう「いつも平均を答える方法」は評価対象の答えの平均を使う数学上の基準です。
DummyRegressorの予測は学習用の平均なので、テストでのR2がちょうど0になるとは限りません。
詳しくは[単純な基準の公式説明](https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators)を参照してください。

## 発展編のコードを読む

| 書き方 | 意味 | 実行できる例 |
| --- | --- | --- |
| `"clf__C": [0.1, 1, 10]` | Pipelineの`clf`というモデルのCを3通り試す。`__`は2個 | [第11章](exercises/ch11_parameter_search.md) |
| `search.best_params_` | CVで選んだ設定 | [第11章](scripts/ch11_parameter_search.py) |
| `projection.transform(X_check)` | 決めた尺度とPCAの軸で、新しい例も変換する | [第12章](scripts/ch12_pca.py) |
| `pd.read_csv(path)` | CSVを表として読み込む | [第13章](scripts/ch13_csv_regression.py) |
| `neg_mean_absolute_error` | 「大きいほど良い」にそろえるため、MAEにマイナスを付けたCV指標 | [第13章の解答](answers/ch13_answer.py) |
| `learning_curve(..., train_sizes=[0.2, ..., 1.0])` | CVの各回の学習側から使う割合を変える | [第14章](scripts/ch14_learning_curves.py) |
| `probabilities >= threshold` | 同じモデルの確率に、判断の境目を当てる | [第15章](scripts/ch15_imbalanced_classification.py) |
| `permutation_importance(..., n_repeats=10)` | 1列を入れ替え、成績を測る操作を列ごとに10回行う | [第16章](scripts/ch16_feature_importance.py) |
| `joblib.dump(model, path)` / `joblib.load(path)` | 前処理を含めたモデルを保存／読み戻す。自分のファイルを同じ環境で使う | [第17章](scripts/ch17_model_persistence.py) |
| `TfidfVectorizer(analyzer="char", ngram_range=(2, 3))` | 学習用の2〜3文字のかたまりを辞書にし、重みで数値にする | [第18章](scripts/ch18_text_classification.py) |
| `sales.shift(1)` | 1行前の値を今日の行へ移す。この教材では昨日の値 | [第19章](scripts/ch19_time_series.py) |
| `TimeSeriesSplit(n_splits=3, test_size=30)` | 過去の学習期間を伸ばし、その後の30行を検証する | [第19章の解答](answers/ch19_answer.py) |

MAEとして説明するなら、最後の指標の符号を戻します。たとえばCVの平均が-0.8ならMAEは0.8です。
単位は答えの単位と同じで、第13章では万円です。
第16章の重要度では、`-MAE`の採点値の差を取るので、返る値は「入れ替え後MAE − 元のMAE」です。
これはMAEの増加（万円）なので、さらにマイナスを付けません。

## もう少し詳しく知りたいとき

英語ですが、次は scikit-learn の公式説明です。

- [学習用とテスト用の分割・交差検証](https://scikit-learn.org/stable/modules/cross_validation.html)
- [評価指標の定義](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [前処理とデータ漏洩でよくある間違い](https://scikit-learn.org/stable/common_pitfalls.html)
