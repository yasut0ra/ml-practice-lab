from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import print_header


def build_search(c_values: list[float] | None = None) -> GridSearchCV:
    model = Pipeline(steps=[("scale", StandardScaler()), ("clf", SVC())])
    # clf は Pipeline 内の名前。__ の後ろが、そのモデルの設定名。
    parameters = {
        "clf__C": [0.1, 1, 10] if c_values is None else c_values,
        "clf__kernel": ["linear", "rbf"],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return GridSearchCV(model, parameters, cv=cv, scoring="accuracy", refit=True)


def comparison_table(search: GridSearchCV) -> pd.DataFrame:
    results = search.cv_results_
    return pd.DataFrame({
        "C": results["param_clf__C"],
        "kernel": results["param_clf__kernel"],
        "CV平均": results["mean_test_score"],
        "CVばらつき": results["std_test_score"],
        "順位": results["rank_test_score"],
    }).sort_values("順位", kind="stable")


def main() -> None:
    print_header("Chapter 11: モデルの設定を選ぶ")
    print("同じSVMでも、設定で予想が変わります。学習用のCVで6通りを比べます。")
    data = load_iris(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.25, random_state=42, stratify=data.target
    )

    search = build_search()
    # 標準化も毎回のCVの学習側だけから学ぶ。最終テストは渡さない。
    search.fit(X_train, y_train)
    print(comparison_table(search).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n選んだ設定: {search.best_params_}")
    print(f"設定選びに使ったCVの平均: {search.best_score_:.4f}")
    print("同率なら、候補を調べた順で最初の設定が選ばれます。")

    # refit=True により、選んだ設定で学習用すべてを学び直している。
    pred = search.predict(X_test)
    print(f"最後に取っておいた{len(y_test)}件の正解率: {accuracy_score(y_test, pred):.4f}")
    print("このテストを見て設定を選び直すと、最後の確認としては使えなくなります。")


if __name__ == "__main__":
    main()
