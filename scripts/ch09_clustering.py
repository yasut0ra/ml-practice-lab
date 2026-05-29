from __future__ import annotations

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from common import print_header


def main() -> None:
    print_header("Chapter 9: クラスタリング")
    print("クラスタリングは正解ラベルなしで、似たデータをグループに分ける教師なし学習です。")

    X, true_labels = make_blobs(
        n_samples=300,
        centers=3,
        cluster_std=1.4,
        random_state=42,
    )
    X_scaled = StandardScaler().fit_transform(X)

    model = KMeans(n_clusters=3, n_init=10, random_state=42)
    cluster_labels = model.fit_predict(X_scaled)

    print(f"silhouette score: {silhouette_score(X_scaled, cluster_labels):.4f}")
    print(f"adjusted rand score: {adjusted_rand_score(true_labels, cluster_labels):.4f}")
    print("cluster centers:")
    for i, center in enumerate(model.cluster_centers_):
        print(f"  cluster {i}: [{center[0]:.3f}, {center[1]:.3f}]")


if __name__ == "__main__":
    main()
