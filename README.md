# ML Practice Lab

「機械学習って何をしているの？」から始める、図解つきの入門教材です。
身近な例でイメージをつかみ、短い Python コードを動かし、結果を見ながら少しずつ学びます。

機械学習は、**過去の例からパターンを見つけ、新しい例の答えを予想する方法**です。
たとえば「これまでの部屋の広さと家賃」を見て、別の部屋の家賃を予想します。

![過去の入力と答えを使ってモデルを学習し、新しい入力から予測を作り、実際の答えと比べて評価する流れ](docs/images/learning-flow.svg)

図の家賃は説明用の架空の例です。実際のコードでは、付属の花・手書き数字などのデータを使います。

## 初めての人はここから

1. [はじめての機械学習ガイド](docs/getting-started.md)を読む。「入力」「答え」「モデル」の意味から説明します。
2. 下の手順で**第1章だけ**動かす。最初から全部のコードを理解する必要はありません。
3. [第1章の図解と演習](exercises/ch01_basic_concepts.md)で、画面に出た数字の意味を確認する。

前提は「Python のファイルを実行できること」です。数学の公式や機械学習の知識は、最初は必要ありません。
Python も初めてなら、ガイドの[コードを読むための最低限の約束](docs/getting-started.md#コードを読むための最低限の約束)を先に見てください。

## セットアップ

Python 3.10 以上を用意してください。ターミナル（コマンドを入力するアプリ）を開き、
この教材のフォルダに移動して、次を1行ずつ実行します。`$` などの記号は入力しません。

macOS / Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/ch01_basic_concepts.py
```

Windows（PowerShell）:

```powershell
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts/ch01_basic_concepts.py
```

1行目は教材専用の Python 環境を作り、2行目は必要な道具を入れ、3行目は第1章を実行します。
環境を作る作業とインストールは初回だけで大丈夫です。
つまずいたら、ガイドの[よくあるつまずき](docs/getting-started.md#よくあるつまずき)を確認してください。

`make` が使える環境では、次のショートカットも利用できます。

```bash
make setup
```

すべての章のサンプルコードを実行します。

```bash
make run
```

構文チェックと短い実行確認を行います。

```bash
make check
```

解答例をまとめて実行します。

```bash
make answers
```

## 構成

```text
.
├── README.md
├── Makefile
├── requirements.txt
├── cheatsheet.md
├── docs/
│   ├── getting-started.md  # 身近な例・用語・最初の実行手順
│   └── images/            # 各章の図解（インストール不要で読めます）
├── datasets/
│   └── README.md
├── scripts/
│   ├── ch01_basic_concepts.py
│   ├── ch02_regression.py
│   ├── ch03_classification.py
│   ├── ch04_metrics.py
│   ├── ch05_split_cv.py
│   ├── ch06_preprocessing.py
│   ├── ch07_tree_forest.py
│   ├── ch08_svm_knn.py
│   ├── ch09_clustering.py
│   ├── ch10_model_comparison.py
│   ├── visualize.py       # 実際のデータをグラフにする
│   └── run_all.py
├── exercises/
│   └── chXX_*.md
└── answers/
    └── chXX_answer.py
```

## 学習の進め方

各章の演習ページには、身近な例・図解・コードの読み方・結果の見方があります。

1. 演習ページの「この章でわかること」と図を見る。
2. 「実行」のコマンドでサンプルを動かす。
3. 「結果の見方」を読み、数字と意味を結びつける。
4. 「まずはここだけ」で、小さな変更を1つ試す。
5. 慣れてきたら練習問題に進み、必要なときに解答例を見る。

最初の区切りは**第1〜3章**です。「データから学ぶ」「数値を予想する」「種類を当てる」が説明できれば十分です。
第4〜6章で予測の確かめ方を学び、第7〜10章でさまざまな方法を比べます。

## 実際のデータをグラフで見る

```bash
.venv/bin/python scripts/visualize.py
```

Windows では `.venv\Scripts\python.exe scripts/visualize.py` を使います。
`make` が使える場合は `make visuals` でも実行できます。

`outputs/figures/` に6枚の PNG 画像ができます。画像を開いて、各章の説明と見比べてください。
画面を表示できない環境でも、画像ファイルとして保存できます。

| 画像 | 見るポイント | 関連する章 |
| --- | --- | --- |
| `01_regression.png` | 点は実際のデータ、線は予測。点と線の縦の距離が誤差 | 1・2 |
| `02_classification.png` | 花の種類ごとに点の位置がどう違うか | 3 |
| `03_confusion_matrix.png` | 当たった件数と、間違えた件数の場所 | 4 |
| `04_cross_validation.png` | 学習する部分と確認する部分を入れ替える仕組み | 5 |
| `05_digits.png` | 手書きの数字が、8 × 8 個の数値として扱われること | 8 |
| `06_clustering.png` | 同じ点が、似たもの同士のグループに分かれる様子 | 9 |

日本語フォントがある環境では日本語で表示します。フォントがない場合は英語になり、
各章の「グラフで見る」で対応する日本語を確認できます。

## 章一覧

| 章 | テーマ | 実行コード | 演習 | 解答 |
| --- | --- | --- | --- | --- |
| 1 | データから学ぶってどういうこと？ | [コード](scripts/ch01_basic_concepts.py) | [図解と演習](exercises/ch01_basic_concepts.md) | [解答](answers/ch01_answer.py) |
| 2 | 数値を予想する（回帰） | [コード](scripts/ch02_regression.py) | [図解と演習](exercises/ch02_regression.md) | [解答](answers/ch02_answer.py) |
| 3 | 種類を当てる（分類） | [コード](scripts/ch03_classification.py) | [図解と演習](exercises/ch03_classification.md) | [解答](answers/ch03_answer.py) |
| 4 | 予想はどれくらい当たった？（評価指標） | [コード](scripts/ch04_metrics.py) | [図解と演習](exercises/ch04_metrics.md) | [解答](answers/ch04_answer.py) |
| 5 | 初めて見る問題で確かめる（分割と交差検証） | [コード](scripts/ch05_split_cv.py) | [図解と演習](exercises/ch05_split_cv.md) | [解答](answers/ch05_answer.py) |
| 6 | データを使いやすく整える（前処理） | [コード](scripts/ch06_preprocessing.py) | [図解と演習](exercises/ch06_preprocessing.md) | [解答](answers/ch06_answer.py) |
| 7 | 質問を重ねて判断する（決定木・森） | [コード](scripts/ch07_tree_forest.py) | [図解と演習](exercises/ch07_tree_forest.md) | [解答](answers/ch07_answer.py) |
| 8 | 境界で分ける・近い例を見る（SVM・k-NN） | [コード](scripts/ch08_svm_knn.py) | [図解と演習](exercises/ch08_svm_knn.md) | [解答](answers/ch08_answer.py) |
| 9 | 似たものを集める（クラスタリング） | [コード](scripts/ch09_clustering.py) | [図解と演習](exercises/ch09_clustering.md) | [解答](answers/ch09_answer.py) |
| 10 | 同じ条件で方法を比べる（モデル比較） | [コード](scripts/ch10_model_comparison.py) | [図解と演習](exercises/ch10_model_comparison.md) | [解答](answers/ch10_answer.py) |

## 使うデータ

主に scikit-learn 付属データセットを使います。外部ダウンロードなしで動きます。

- `load_diabetes`: 回帰
- `load_iris`: 花の品種を当てる分類
- `load_breast_cancer`: 二値分類、評価指標、モデル比較
- `load_digits`: 画像分類
- `make_blobs`: クラスタリングの人工データ

## 学習メモ

- 最初は精度の高さより、データの形、学習、予測、評価の流れを理解することを優先してください。
- `random_state=42` は結果を再現しやすくするための固定値です。
- 章のコードは短くしてあります。慣れてきたら特徴量、モデル、評価指標を変更して結果を比較してください。
- わからない言葉は[日本語つきの用語早見表](cheatsheet.md)で確認できます。
