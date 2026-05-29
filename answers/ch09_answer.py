from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from common import print_header, print_table
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


def main() -> None:
    print_header("Answer 9: クラスタリング")

    X, true_labels = make_blobs(
        n_samples=300,
        centers=3,
        cluster_std=1.4,
        random_state=42,
    )
    X_scaled = StandardScaler().fit_transform(X)

    rows = []
    for k in [2, 3, 4, 5]:
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = model.fit_predict(X_scaled)
        rows.append(
            {
                "k": k,
                "silhouette": f"{silhouette_score(X_scaled, labels):.4f}",
                "adjusted_rand": f"{adjusted_rand_score(true_labels, labels):.4f}",
            }
        )

    print_table(rows, ["k", "silhouette", "adjusted_rand"])


if __name__ == "__main__":
    main()
