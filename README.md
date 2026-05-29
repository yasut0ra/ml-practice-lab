# ML Practice Lab

Python と scikit-learn で機械学習の基礎を手を動かしながら学ぶ教材です。
対象は Python の基礎を学んだあと、機械学習に入門する学生です。

このリポジトリは Notebook ではなく `scripts/` 形式にしています。エディタでコードを読み、実行して、演習で少しずつ書き換える流れです。

## セットアップ

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
│   └── run_all.py
├── exercises/
│   └── chXX_*.md
└── answers/
    └── chXX_answer.py
```

## 学習の進め方

1. `scripts/chXX_*.py` を読む
2. `python scripts/chXX_*.py` で実行する
3. `exercises/chXX_*.md` の練習問題に取り組む
4. 必要なときだけ `answers/chXX_answer.py` を見る

## 章一覧

| 章 | テーマ | 実行コード | 演習 | 解答 |
| --- | --- | --- | --- | --- |
| 1 | 機械学習の基本概念 | `scripts/ch01_basic_concepts.py` | `exercises/ch01_basic_concepts.md` | `answers/ch01_answer.py` |
| 2 | 回帰 | `scripts/ch02_regression.py` | `exercises/ch02_regression.md` | `answers/ch02_answer.py` |
| 3 | 分類 | `scripts/ch03_classification.py` | `exercises/ch03_classification.md` | `answers/ch03_answer.py` |
| 4 | 評価指標 | `scripts/ch04_metrics.py` | `exercises/ch04_metrics.md` | `answers/ch04_answer.py` |
| 5 | train/test split と cross validation | `scripts/ch05_split_cv.py` | `exercises/ch05_split_cv.md` | `answers/ch05_answer.py` |
| 6 | 前処理 | `scripts/ch06_preprocessing.py` | `exercises/ch06_preprocessing.md` | `answers/ch06_answer.py` |
| 7 | 決定木・ランダムフォレスト | `scripts/ch07_tree_forest.py` | `exercises/ch07_tree_forest.md` | `answers/ch07_answer.py` |
| 8 | SVM と k-NN | `scripts/ch08_svm_knn.py` | `exercises/ch08_svm_knn.md` | `answers/ch08_answer.py` |
| 9 | クラスタリング | `scripts/ch09_clustering.py` | `exercises/ch09_clustering.md` | `answers/ch09_answer.py` |
| 10 | 簡単なモデル比較 | `scripts/ch10_model_comparison.py` | `exercises/ch10_model_comparison.md` | `answers/ch10_answer.py` |

## 使うデータ

主に scikit-learn 付属データセットを使います。外部ダウンロードなしで動きます。

- `load_diabetes`: 回帰
- `load_iris`: 多クラス分類とクラスタリング
- `load_breast_cancer`: 二値分類、評価指標、モデル比較
- `load_digits`: 画像分類
- `make_blobs`: クラスタリングの人工データ

## 学習メモ

- 最初は精度の高さより、データの形、学習、予測、評価の流れを理解することを優先してください。
- `random_state=42` は結果を再現しやすくするための固定値です。
- 章のコードは短くしてあります。慣れてきたら特徴量、モデル、評価指標を変更して結果を比較してください。
