"""3件の架空の家賃データで、学習・予測・評価を体験する。"""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from common import print_header


def main() -> None:
    print_header("Chapter 0: 家賃を予想してみよう")
    print("部屋の広さ（㎡）から家賃（万円）を予想します。すべて教材用の架空のデータです。")

    # 1. 各行が1つの部屋。入力の表と、答えの並びは同じ順番にする。
    X_train = [[20], [40], [60]]
    y_train = [6, 10, 14]
    print("学習用: 20㎡ → 6万円、40㎡ → 10万円、60㎡ → 14万円")

    # 2. 広さと家賃の関係をデータから学ぶ。
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 3. 学習に使っていない35㎡の部屋を予想する。答えは渡さない。
    new_area = 35
    prediction = model.predict([[new_area]])[0]
    print(f"{new_area}㎡の部屋の予想: {prediction:.1f}万円")

    # 4. 別に取っておいた部屋で、実際の家賃と予想を比べる。
    X_test = [[30], [50]]
    y_test = [9.0, 12.0]
    pred = model.predict(X_test)
    for room, actual, predicted in zip(X_test, y_test, pred):
        print(f"確認用: {room[0]}㎡ / 実際 {actual:.1f}万円 / 予想 {predicted:.1f}万円")
    print(f"ずれの平均（MAE）: {mean_absolute_error(y_test, pred):.2f}万円")
    print("3件で学び、別の2件で確認しました。この少数の例だけで実際の家賃を予想できるとは言えません。")


if __name__ == "__main__":
    main()
