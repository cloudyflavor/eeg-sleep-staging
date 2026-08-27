"""Phase 3 — 하이퍼파라미터 탐색과 특징 수 곡선. 근거: docs/05-experiments.md.

탐색은 5-fold, 최종 보고는 10-fold 로 다시 잰다. 20개 중 최고를 고르는 행위가
그 점수를 부풀리므로, 고르는 데 쓴 숫자를 결과로 쓰면 안 된다.
"""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sleepstage.config import Config, config_hash
from sleepstage.evaluation.split import fold_masks, load_folds
from sleepstage.evaluation.train import (
    META_COLS,
    SUBSETS,
    _fit,
    _metrics,
    load_table,
)

#: 기본값을 중심에 두고 양쪽으로 벌린 격자. 유효 표본이 78명이라 얕고 규제 센 쪽을 더 본다.
#: iterations 2000 은 뺐다 — 실측상 조합 하나가 5-fold 에 37분이라 예산이 안 맞고,
#: 트리를 늘리는 방향은 learning_rate 를 낮추는 것으로 대신 탐색된다.
CATBOOST_SPACE = {
    "depth": [4, 6, 8],
    "iterations": [500, 1000],
    "learning_rate": [0.03, 0.1],
    "l2_leaf_reg": [1, 3, 10],
}


def sample_space(space: dict, n: int, seed: int) -> list[dict]:
    """무작위 탐색. 기본값 조합을 반드시 첫 후보로 넣어 비교 기준을 남긴다."""
    rng = np.random.default_rng(seed)
    default = {"depth": 6, "iterations": 1000, "learning_rate": 0.1, "l2_leaf_reg": 3}
    seen, out = {json.dumps(default, sort_keys=True)}, [default]
    while len(out) < n:
        cand = {k: rng.choice(v).item() for k, v in space.items()}
        key = json.dumps(cand, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(cand)
    return out


def _catboost(params: dict, seed: int):
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        random_seed=seed, thread_count=8, verbose=False, loss_function="MultiClass", **params
    )


def _evaluate(X, y, table, folds, params, seed, weighted) -> float:
    """주어진 fold 들에서 pooled macro-F1. 탐색용이라 예측은 저장하지 않는다."""
    from sklearn.utils.class_weight import compute_sample_weight

    trues, preds = [], []
    for fold in folds:
        train_mask, test_mask = fold_masks(table, fold)
        sw = compute_sample_weight("balanced", y[train_mask]) if weighted else None
        model = _fit(_catboost(params, seed), X[train_mask], y[train_mask], sw)
        trues.append(y[test_mask])
        preds.append(model.predict(X[test_mask]).ravel().astype(int))
    return _metrics(np.concatenate(trues), np.concatenate(preds))["macro_f1"]


def search(cfg: Config, n_trials: int = 20, search_folds: int = 5) -> dict[str, Any]:
    """무작위 탐색. 결과는 순위표만 남기고, 우승 설정의 10-fold 평가는 따로 돌린다."""
    p = cfg["train"]
    features_path = Path(p["features"])
    table, _ = load_table(features_path, p["drop_missing"])
    folds = load_folds(p["splits"])[:search_folds]
    subset = p.get("feature_subset", "all")
    cols = [c for c in table.columns if c not in META_COLS and SUBSETS[subset](c)]
    X, y = table[cols].to_numpy(dtype=np.float32), table["stage"].to_numpy()

    results = []
    for i, params in enumerate(sample_space(CATBOOST_SPACE, n_trials, p["seed"])):
        t0 = time.perf_counter()
        score = _evaluate(X, y, table, folds, params, p["seed"], p["class_weight"] == "balanced")
        results.append({**params, "macro_f1": score, "sec": round(time.perf_counter() - t0, 1)})
        tag = " (기본값)" if i == 0 else ""
        print(f"  [{i + 1}/{n_trials}] {score:.4f}  {params}{tag}", flush=True)

    results.sort(key=lambda r: -r["macro_f1"])
    out = (
        Path(p["out_dir"]) / f"search_{config_hash({'f': features_path.stem, 'n': n_trials})}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"search_folds": search_folds, "results": results}, indent=2), encoding="utf-8"
    )
    return {
        "best": results[0],
        "default": results[0] if len(results) == 1 else None,
        "out": str(out),
    }


def feature_curve(cfg: Config, counts: list[int], importance_csv: str) -> dict[str, Any]:
    """중요도 상위 N개만 남기고 재학습 — "몇 개로 충분한가" 곡선.

    lasso 규제 경로 대신 실제 배포 모델(부스팅) 기준으로 같은 질문에 답한다.
    """
    p = cfg["train"]
    table, _ = load_table(Path(p["features"]), p["drop_missing"])
    folds = load_folds(p["splits"])
    ranked = pd.read_csv(importance_csv)["feature"].tolist()
    y = table["stage"].to_numpy()

    points = []
    for n in counts:
        cols = [c for c in ranked[:n] if c in table.columns]
        X = table[cols].to_numpy(dtype=np.float32)
        t0 = time.perf_counter()
        score = _evaluate(X, y, table, folds, {}, p["seed"], p["class_weight"] == "balanced")
        points.append(
            {"n_features": len(cols), "macro_f1": score, "sec": round(time.perf_counter() - t0, 1)}
        )
        print(f"  상위 {len(cols):>3}개  macro-F1 {score:.4f}", flush=True)

    out = Path(p["out_dir"]) / "feature_curve.json"
    out.write_text(json.dumps({"points": points}, indent=2), encoding="utf-8")
    return {"points": points, "out": str(out)}
