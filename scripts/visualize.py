"""実際のデータを11枚の図にする。実行: .venv/bin/python scripts/visualize.py"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

# 画面のない環境でも実行できるよう、画像への保存を使う。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch, Rectangle
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits, load_iris, make_blobs, make_moons
from sklearn.linear_model import LinearRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ch05_split_cv import build_model
from ch11_parameter_search import build_search, comparison_table
from ch12_pca import build_projection
from ch13_csv_regression import FEATURES, TARGET, build_model as build_rent_model, load_rent_data


COLORS = ["#2765a1", "#28806b", "#b06a20"]
MARKERS = ["o", "s", "^"]

# 日本語フォントがなければ、四角い文字になる代わりに英語で描く。
available_fonts = {font.name for font in font_manager.fontManager.ttflist}
JAPANESE_FONT = next(
    (name for name in ["Hiragino Sans", "Yu Gothic", "Meiryo", "Noto Sans CJK JP",
                       "Noto Sans JP", "IPAexGothic", "IPAGothic"] if name in available_fonts),
    None,
)
if JAPANESE_FONT:
    plt.rcParams.update({"font.family": JAPANESE_FONT, "axes.unicode_minus": False})


def caption(japanese: str, english: str) -> str:
    return japanese if JAPANESE_FONT else english


def regression() -> plt.Figure:
    """1つの手がかりで予想して、点と線の距離を見る。"""
    data = load_diabetes(as_frame=True)
    X = data.data[["bmi"]]
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression().fit(X_train, y_train)
    order = np.argsort(X_test["bmi"].to_numpy())
    bmi = X_test["bmi"].to_numpy()[order]
    actual = y_test.to_numpy()[order]
    pred = model.predict(X_test)[order]

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ax.scatter(bmi, actual, color=COLORS[0], alpha=0.8,
               label=caption("実際の値（確認用）", "Actual test data"))
    ax.plot(bmi, pred, color=COLORS[1], linewidth=2.5,
            label=caption("学習した予測", "Learned prediction"))

    # 中央付近で、予測とのずれが見える1件を例として示す。
    candidates = np.arange(len(bmi) // 3, 2 * len(bmi) // 3)
    index = candidates[np.argmax(np.abs(actual[candidates] - pred[candidates]))]
    ax.plot([bmi[index], bmi[index]], [pred[index], actual[index]],
            color=COLORS[2], linewidth=2.5,
            label=caption("1件の予測のずれ", "Prediction error (one example)"))
    ax.set(title=caption("回帰：予想と実際の答えを比べる",
                         "Regression: compare a prediction with the real answer"),
           xlabel=caption("BMI（加工済みの値）", "BMI (scaled)"),
           ylabel=caption("病気の進行度", "Progression score"))
    ax.legend(loc="upper left", fontsize=10)
    return fig


def classification() -> plt.Figure:
    """ラベル付きの花を、2つの特徴量で眺める（予測の図ではない）。"""
    data = load_iris()
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for group, name in enumerate(data.target_names):
        selected = data.target == group
        ax.scatter(data.data[selected, 2], data.data[selected, 3],
                   color=COLORS[group], marker=MARKERS[group], alpha=0.8, label=name)
    ax.set(title=caption("分類：花の測定値と、本当の品種を見る",
                         "Classification: flowers have known variety labels"),
           xlabel=caption("花びらの長さ（cm）", "Petal length (cm)"),
           ylabel=caption("花びらの幅（cm）", "Petal width (cm)"))
    ax.legend(title=caption("本当の品種", "Actual variety"))
    return fig


def classification_errors() -> plt.Figure:
    """第4章と同じ条件で予測し、混同行列を描く。"""
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = (data.target == 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = build_model()
    model.fit(X_train, y_train)
    matrix = confusion_matrix(y_test, model.predict(X_test), labels=[0, 1])

    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    display = ConfusionMatrixDisplay(
        matrix, display_labels=[caption("良性（0）", "Benign (0)"),
                                caption("悪性（1）", "Malignant (1)")]
    )
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    labels = [caption("TN：正しい判定", "TN: correct"), caption("FP：誤検知", "FP: false alarm"),
              caption("FN：見逃し", "FN: missed"), caption("TP：正しい検出", "TP: found")]
    for item, label, count in zip(display.text_.flat, labels, matrix.flat):
        item.set_text(f"{item.get_text()}\n{label}")
        item.set_fontsize(12)
        item.set_color("white" if count > matrix.max() * 0.7 else "#19324d")
    ax.set(title=caption("評価：どんな間違いをしたか、件数で見る",
                         "Evaluation: which predictions were wrong?"),
           xlabel=caption("予想した答え", "Predicted label"),
           ylabel=caption("実際の答え", "Actual label"))
    return fig


def cross_validation() -> plt.Figure:
    """5分割の模式図。最終テストはCVとは別に残す。"""
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    training_color, validation_color, test_color = "#dce9f5", "#f4c788", "#e3e7ec"
    for row in range(5):
        for column in range(5):
            validation = row == column
            ax.add_patch(Rectangle((column, row), 0.94, 0.85,
                                   facecolor=validation_color if validation else training_color))
            ax.text(column + 0.47, row + 0.425,
                    caption("検証", "Check") if validation else caption("学習", "Learn"),
                    ha="center", va="center", fontsize=12)
    ax.add_patch(Rectangle((5.4, 0), 1.5, 4.85, facecolor=test_color))
    ax.text(6.15, 2.425, caption("最終テスト\n\nCVには\n使わない", "Final test\n\nKept aside\nfrom CV"),
            ha="center", va="center", fontsize=12)
    ax.set(xlim=(-0.1, 7), ylim=(5, -0.6),
           title=caption("交差検証：確認する部分を交代し、毎回学び直す",
                         "Cross-validation: learn again for each round"),
           yticks=np.arange(5) + 0.425, yticklabels=[caption(f"{i}回目", f"Round {i}") for i in range(1, 6)],
           xticks=np.arange(5) + 0.47, xticklabels=[caption(f"区画{i}", f"Part {i}") for i in range(1, 6)])
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(handles=[Patch(facecolor=training_color, label=caption("学習用", "Training")),
                       Patch(facecolor=validation_color, label=caption("検証用", "Validation")),
                       Patch(facecolor=test_color, label=caption("最終テスト", "Final test"))],
              loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=3)
    return fig


def digits() -> plt.Figure:
    """画像とラベルを見て、64個の数値で表されることを知る。"""
    data = load_digits()
    fig, axes = plt.subplots(2, 5, figsize=(9, 5), layout="constrained")
    for number, ax in enumerate(axes.flat):
        index = np.flatnonzero(data.target == number)[0]
        ax.imshow(data.images[index], cmap="gray_r", vmin=0, vmax=16, interpolation="nearest")
        ax.set_title(caption(f"本当の数字：{number}", f"Actual digit: {number}"))
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(caption("画像1枚 = 8 × 8ピクセル = 64個の数値",
                         "One image = 8 x 8 pixels = 64 numerical features"))
    return fig


def clustering() -> plt.Figure:
    """同じ点を分ける前と分けた後を、同じ座標で比べる。"""
    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.4, random_state=42)
    X_scaled = StandardScaler().fit_transform(X)
    model = KMeans(n_clusters=3, n_init=10, random_state=42)
    labels = model.fit_predict(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True, layout="constrained")
    axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], color="#62788e", alpha=0.6, s=24)
    axes[0].set_title(caption("分ける前：答えのラベルなし", "Before grouping: no labels given"))
    for group in range(3):
        selected = labels == group
        axes[1].scatter(X_scaled[selected, 0], X_scaled[selected, 1],
                        color=COLORS[group], marker=MARKERS[group], alpha=0.7,
                        s=24, label=caption(f"グループ{group}", f"Group {group}"))
    axes[1].scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
                    marker="x", color="#19324d", s=160, linewidths=3,
                    label=caption("中心", "Centers"))
    axes[1].set_title(caption("分けた後：3つのグループ", "After KMeans: 3 groups"))
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.set_xlabel(caption("手がかり1（標準化済み）", "Feature 1 (scaled)"))
    axes[0].set_ylabel(caption("手がかり2（標準化済み）", "Feature 2 (scaled)"))
    return fig


def overfitting() -> plt.Figure:
    """ノイズのある人工データで、細かい分岐と学習・検証の成績を比べる。"""
    X, y = make_moons(n_samples=220, noise=0.3, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.35, random_state=42, stratify=y
    )
    x_grid, y_grid = np.meshgrid(
        np.linspace(X[:, 0].min() - 0.2, X[:, 0].max() + 0.2, 160),
        np.linspace(X[:, 1].min() - 0.2, X[:, 1].max() + 0.2, 160),
    )
    grid = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    fig = plt.figure(figsize=(11, 8), layout="constrained")
    panels = fig.add_gridspec(2, 3)
    for column, depth in enumerate([1, 3, None]):
        ax = fig.add_subplot(panels[0, column])
        model = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(X_train, y_train)
        regions = model.predict(grid).reshape(x_grid.shape)
        ax.contourf(x_grid, y_grid, regions, levels=[-0.5, 0.5, 1.5], colors=COLORS[:2], alpha=0.15)
        for group in [0, 1]:
            selected = y_train == group
            ax.scatter(X_train[selected, 0], X_train[selected, 1], s=20,
                       color=COLORS[group], marker=MARKERS[group], label=caption(f"種類{group}", f"Class {group}"))
        depth_label = caption("上限なし", "Unlimited") if depth is None else str(depth)
        ax.set(title=caption(f"質問の深さ：{depth_label}", f"Tree depth: {depth_label}"),
               xlabel=caption("手がかり1", "Feature 1"), ylabel=caption("手がかり2", "Feature 2"))
        if column == 0:
            ax.legend(fontsize=10)

    depths = list(range(1, 11)) + [None]
    training_scores, validation_scores = [], []
    for depth in depths:
        model = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(X_train, y_train)
        training_scores.append(model.score(X_train, y_train))
        validation_scores.append(model.score(X_valid, y_valid))
    ax = fig.add_subplot(panels[1, :])
    ax.plot(range(len(depths)), training_scores, marker="o", color=COLORS[0],
            label=caption("学習用の正解率", "Training accuracy"))
    ax.plot(range(len(depths)), validation_scores, marker="s", color=COLORS[2],
            label=caption("検証用の正解率", "Validation accuracy"))
    ax.set(xticks=range(len(depths)), xticklabels=[str(d) if d else caption("上限なし", "None") for d in depths],
           xlabel=caption("質問の深さ", "Tree depth"), ylabel=caption("正解率", "Accuracy"), ylim=(0, 1.05))
    ax.legend(loc="lower right")
    fig.suptitle(caption("過学習：練習した例で満点でも、別の例では？",
                         "Overfitting: perfect training answers do not guarantee new answers"))
    return fig


def decision_boundaries() -> plt.Figure:
    """花びらの2特徴量で、SVMとk-NNが学ぶ境界を同じデータ上で見る。"""
    data = load_iris()
    X, y = data.data[:, 2:4], data.target
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    x_grid, y_grid = np.meshgrid(
        np.linspace(X[:, 0].min() - 0.3, X[:, 0].max() + 0.3, 160),
        np.linspace(X[:, 1].min() - 0.2, X[:, 1].max() + 0.2, 160),
    )
    grid = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    models = [make_pipeline(StandardScaler(), SVC(kernel="linear", C=1)),
              make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True, layout="constrained")
    for ax, name, model in zip(axes, ["SVM (linear)", "k-NN (k=5)"], models):
        model.fit(X_train, y_train)
        regions = model.predict(grid).reshape(x_grid.shape)
        ax.contourf(x_grid, y_grid, regions, levels=[-0.5, 0.5, 1.5, 2.5], colors=COLORS, alpha=0.14)
        ax.contour(x_grid, y_grid, regions, levels=[0.5, 1.5], colors="#53708d", linewidths=0.8)
        for group, variety in enumerate(data.target_names):
            selected = y_train == group
            ax.scatter(X_train[selected, 0], X_train[selected, 1], color=COLORS[group],
                       marker=MARKERS[group], s=22, alpha=0.7, label=variety)
            selected = y_valid == group
            ax.scatter(X_valid[selected, 0], X_valid[selected, 1], color=COLORS[group],
                       marker=MARKERS[group], s=65, edgecolor="#19324d", linewidth=1.2)
        accuracy = model.score(X_valid, y_valid)
        ax.set_title(name + caption(f" / 確認用 {accuracy:.1%}", f" / Validation {accuracy:.1%}"))
        ax.set_xlabel(caption("花びらの長さ（cm）", "Petal length (cm)"))
    axes[0].set_ylabel(caption("花びらの幅（cm）", "Petal width (cm)"))
    axes[0].legend(title=caption("点の色・形：本当の品種", "Color and shape: actual variety"), fontsize=9)
    fig.suptitle(caption("分類の境界を見る：大きな枠付きの点が確認用",
                         "Classification boundaries: large outlined points are validation data"))
    return fig


def parameter_search() -> plt.Figure:
    """第11章と同じCVで、設定ごとの平均と標準偏差を描く。"""
    data = load_iris(as_frame=True)
    X_train, _, y_train, _ = train_test_split(
        data.data, data.target, test_size=0.25, random_state=42, stratify=data.target
    )
    result = comparison_table(build_search().fit(X_train, y_train))
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    for kernel, color, marker in zip(["linear", "rbf"], COLORS, MARKERS):
        selected = result[result["kernel"] == kernel].sort_values("C")
        ax.errorbar(selected["C"].astype(float), selected["CV平均"],
                    yerr=selected["CVばらつき"], label=kernel, color=color,
                    marker=marker, markersize=8, capsize=6, linewidth=2)
    ax.set(xscale="log", xticks=[0.1, 1, 10], xticklabels=["0.1", "1", "10"],
           xlabel=caption("C（対数目盛）", "C (log scale)"),
           ylabel=caption("CVの平均正解率", "Mean CV accuracy"), ylim=(0.8, 1.04),
           title=caption("設定を比べる：点は平均、線は5回の標準偏差",
                         "Compare settings: mean accuracy with +/- one standard deviation"))
    ax.legend(title="kernel")
    ax.grid(alpha=0.2)
    return fig


def pca_projection() -> plt.Figure:
    """第12章と同じ尺度・軸で、花の位置と累積寄与率を描く。"""
    data = load_iris(as_frame=True)
    X_train, X_check, y_train, y_check = train_test_split(
        data.data, data.target, test_size=0.25, random_state=42, stratify=data.target
    )
    projection = build_projection().fit(X_train)
    train_points, check_points = projection.transform(X_train), projection.transform(X_check)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for group, name in enumerate(data.target_names):
        selected = y_train.to_numpy() == group
        axes[0].scatter(train_points[selected, 0], train_points[selected, 1],
                        color=COLORS[group], marker=MARKERS[group], s=24, alpha=0.6, label=name)
        selected = y_check.to_numpy() == group
        axes[0].scatter(check_points[selected, 0], check_points[selected, 1],
                        color=COLORS[group], marker=MARKERS[group], s=65,
                        edgecolor="#19324d", linewidth=1.2)
    axes[0].set(title=caption("4列を組み合わせて2軸に", "Combine four features into two axes"),
                xlabel=caption("第1軸", "Principal component 1"),
                ylabel=caption("第2軸", "Principal component 2"))
    axes[0].legend(fontsize=9, title=caption("本当の品種", "Actual variety"))
    full_projection = build_projection(4).fit(X_train)
    cumulative = np.cumsum(full_projection.named_steps["pca"].explained_variance_ratio_)
    axes[1].plot([1, 2, 3, 4], cumulative, color=COLORS[0], marker="o", linewidth=2)
    for count, ratio in enumerate(cumulative, start=1):
        axes[1].annotate(f"{ratio:.1%}", (count, ratio), xytext=(0, -22),
                         textcoords="offset points", ha="center", fontsize=11)
    axes[1].set(xticks=[1, 2, 3, 4], ylim=(0, 1.1),
                xlabel=caption("残す軸の数", "Number of components"),
                ylabel=caption("累積寄与率", "Cumulative explained variance"),
                title=caption("標準化した学習用のばらつきを残す割合", "Fraction of standardized training variance retained"))
    axes[1].grid(alpha=0.2)
    fig.suptitle(caption("PCA：枠付きは確認用の花。色は本当の品種、予測ではない",
                         "PCA: outlined points are check data; colors show actual varieties"))
    return fig


def csv_predictions() -> plt.Figure:
    """第13章と同じ固定モデルの最終テストの予測とずれを描く。"""
    data = load_rent_data()
    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES], data[TARGET], test_size=0.25, random_state=42
    )
    pred = build_rent_model().fit(X_train, y_train).predict(X_test)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    lower, upper = min(y_test.min(), pred.min()) - 0.5, max(y_test.max(), pred.max()) + 0.5
    axes[0].scatter(y_test, pred, color=COLORS[0], s=40, alpha=0.8)
    axes[0].plot([lower, upper], [lower, upper], color=COLORS[1], linestyle="--",
                 label=caption("予測と実際が一致", "Prediction equals actual"))
    axes[0].set(xlim=(lower, upper), ylim=(lower, upper),
                xlabel=caption("実際の家賃（万円）", "Actual rent (10,000 JPY)"),
                ylabel=caption("予測した家賃（万円）", "Predicted rent (10,000 JPY)"))
    axes[0].legend(fontsize=10)
    axes[1].scatter(y_test, pred - y_test, color=COLORS[2], s=40, alpha=0.8)
    axes[1].axhline(0, color="#53708d", linestyle="--")
    axes[1].set(xlabel=caption("実際の家賃（万円）", "Actual rent (10,000 JPY)"),
                ylabel=caption("予測 − 実際（万円）", "Predicted - actual (10,000 JPY)"),
                title=caption("0より上は高め、下は安めの予想", "Above zero: too high; below zero: too low"))
    fig.suptitle(caption("CSVからの家賃予想：架空データの最終テスト30件",
                         "Rent predictions from CSV: 30 final test cases from fictional data"))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="教材のデータを11枚のPNGグラフにして保存します。")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "outputs" / "figures",
                        help="画像の保存先（既定: 教材フォルダの outputs/figures）")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    figures = [
        ("01_regression.png", regression),
        ("02_classification.png", classification),
        ("03_confusion_matrix.png", classification_errors),
        ("04_cross_validation.png", cross_validation),
        ("05_digits.png", digits),
        ("06_clustering.png", clustering),
        ("07_overfitting.png", overfitting),
        ("08_decision_boundaries.png", decision_boundaries),
        ("09_parameter_search.png", parameter_search),
        ("10_pca.png", pca_projection),
        ("11_csv_prediction.png", csv_predictions),
    ]
    for filename, draw in figures:
        fig = draw()
        destination = args.output_dir / filename
        fig.savefig(destination, dpi=160, facecolor="white")
        plt.close(fig)
        print(f"保存しました: {destination}")
    print("画像を開き、各章の『グラフで見る』と見比べてください。")


if __name__ == "__main__":
    main()
