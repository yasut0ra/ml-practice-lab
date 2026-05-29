from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CHAPTERS = [
    "ch01_basic_concepts.py",
    "ch02_regression.py",
    "ch03_classification.py",
    "ch04_metrics.py",
    "ch05_split_cv.py",
    "ch06_preprocessing.py",
    "ch07_tree_forest.py",
    "ch08_svm_knn.py",
    "ch09_clustering.py",
    "ch10_model_comparison.py",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Reserved for quick checks. Current scripts are already lightweight.",
    )
    parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"

    for chapter in CHAPTERS:
        path = scripts_dir / chapter
        print(f"\n>>> python {path.relative_to(repo_root)}", flush=True)
        subprocess.run([sys.executable, str(path)], cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
