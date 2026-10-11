from __future__ import annotations

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from common import print_header


LABELS = ["shipping", "payment"]
LABEL_NAMES = {"shipping": "配送", "payment": "支払い"}


def make_messages() -> pd.DataFrame:
    # すべて教材用の架空の問い合わせ。実際の顧客の文章は使わない。
    shipping = [
        "荷物がまだ届きません", "配達の予定日を教えてください", "発送はいつになりますか",
        "届け先の住所を変更したいです", "配送状況を確認できますか", "追跡番号が知りたいです",
        "不在票をなくしたので再配達をお願いします", "商品を受け取る時間を指定したいです",
        "荷物の箱が破れて届きました", "注文品が別の住所へ配達されています", "発送済みなのに追跡できません",
        "届ける日を来週に変更できますか", "宅配ボックスに商品が入っていません", "配送先の名前を間違えました",
        "荷物が営業所に止まっています", "配達員と連絡を取りたいです", "海外への発送は可能ですか",
        "商品は何日で到着しますか", "配送の遅れについて問い合わせます", "受け取り場所をコンビニにしたいです",
        "二つの注文を一緒に発送できますか", "届いた荷物が違う人のものでした", "運送会社を教えてください",
        "再配達の申し込み方法がわかりません",
    ]
    payment = [
        "カードで支払いができません", "請求金額が注文時と違います", "返金はいつ行われますか",
        "領収書を発行してほしいです", "銀行振込の口座を教えてください", "決済方法を変更したいです",
        "クレジットカードが二重に請求されています", "請求書の宛名を会社名にしてください", "入金が反映されていません",
        "支払い期限を延ばせますか", "返金先の銀行口座を登録したいです", "決済エラーで注文できません",
        "分割払いは利用できますか", "振込の金額を間違えました", "領収書を再発行できますか",
        "代金の引き落とし日を確認したいです", "請求書がメールで届きません", "カードの有効期限を更新したいです",
        "支払ったのに未払いと表示されます", "返金額が不足しています", "コンビニ払いの番号を教えてください",
        "現金で決済する方法はありますか", "入金の確認をお願いします", "カード明細に知らない請求があります",
    ]
    return pd.DataFrame({"text": shipping + payment, "label": ["shipping"] * 24 + ["payment"] * 24})


def split_messages():
    data = make_messages()
    return train_test_split(data["text"], data["label"], test_size=0.25, stratify=data["label"], random_state=42)


def build_model(ngram_range: tuple[int, int] = (2, 3)) -> Pipeline:
    return Pipeline([
        ("text", TfidfVectorizer(analyzer="char", ngram_range=ngram_range)),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])


def prediction_table(texts: pd.Series, labels: pd.Series, predictions) -> pd.DataFrame:
    return pd.DataFrame({"text": texts, "actual": labels, "predicted": predictions})


def main() -> None:
    print_header("Chapter 18: 日本語の問い合わせを分類する")
    X_train, X_test, y_train, y_test = split_messages()
    model = build_model().fit(X_train, y_train)
    pred = model.predict(X_test)
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train.to_frame(), y_train)
    print(f"学習用{len(y_train)}件 / 最終テスト{len(y_test)}件（各種類6件）")
    for name, guesses in [("最多の種類を答える基準", baseline.predict(X_test.to_frame())), ("文章から学ぶモデル", pred)]:
        print(f"{name}: 正解率={accuracy_score(y_test, guesses):.3f}, macro F1={f1_score(y_test, guesses, average='macro'):.3f}")
    print("\n答え・予想の名前: shipping=配送 / payment=支払い")
    print(prediction_table(X_test, y_test, pred).to_string(index=False))
    print("\n行が答え、列が予想の件数（順番は配送、支払い）:")
    print(confusion_matrix(y_test, pred, labels=LABELS))
    vectorizer = model.named_steps["text"]
    print(f"\n学習した文字のかたまりの種類: {len(vectorizer.vocabulary_)}")
    print(f"『荷物が』を切った例: {vectorizer.build_analyzer()('荷物が')}")
    new_texts = pd.Series(["配達の時間を変更できますか", "請求書の金額を確認したいです", "送料を返金してほしいです"])
    print("\n新しい文章の予想（正解は未確認）:")
    print(pd.DataFrame({"text": new_texts, "predicted": model.predict(new_texts)}).to_string(index=False))
    print("配送と支払いが混ざる文章でも、必ずどちらかを答えます。文章を理解できた証明ではありません。")


if __name__ == "__main__":
    main()
