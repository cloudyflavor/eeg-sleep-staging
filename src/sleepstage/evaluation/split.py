"""피험자 단위 fold 를 만들어 파일로 고정한다. 모든 실험이 같은 분할을 쓴다.

근거: docs/05-experiments.md, docs/02-prior-art.md §2.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

STAGE_NAMES = ("W", "LS", "DS", "R")


def make_folds(
    table: pd.DataFrame,
    n_splits: int = 10,
    seed: int = 0,
) -> tuple[list[dict[str, list[str]]], dict[str, Any]]:
    """피험자를 n_splits 개 fold 로. 같은 피험자가 양쪽에 있으면 예외."""
    subjects = table["subject"].to_numpy()
    stages = table["stage"].to_numpy()

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds, diagnostics = [], []

    for train_idx, test_idx in splitter.split(table, stages, groups=subjects):
        train_subjects = sorted(set(subjects[train_idx]))
        test_subjects = sorted(set(subjects[test_idx]))

        overlap = set(train_subjects) & set(test_subjects)
        if overlap:
            raise AssertionError(f"같은 피험자가 양쪽에 있습니다: {sorted(overlap)}")

        folds.append({"train": train_subjects, "test": test_subjects})
        counts = Counter(stages[test_idx].tolist())
        total = sum(counts.values())
        diagnostics.append(
            {
                "n_test_subjects": len(test_subjects),
                "n_test_epochs": total,
                "test_class_ratio": {
                    STAGE_NAMES[k]: round(counts.get(k, 0) / total, 4) for k in range(4)
                },
            }
        )

    return folds, {"folds": diagnostics, **_balance_summary(diagnostics)}


def _balance_summary(diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    """fold 간 클래스 비율 폭 — 희소 클래스 쏠림 확인용."""
    spread = {}
    for stage in STAGE_NAMES:
        ratios = [d["test_class_ratio"][stage] for d in diagnostics]
        spread[stage] = {
            "min": min(ratios),
            "max": max(ratios),
            "range": round(max(ratios) - min(ratios), 4),
        }
    return {"class_ratio_spread": spread}


def write_folds(
    features_path: str | Path,
    out_path: str | Path,
    n_splits: int = 10,
    seed: int = 0,
) -> dict[str, Any]:
    """fold 를 JSON 으로 저장. 행 번호가 아니라 피험자 이름을 저장한다."""
    features_path, out_path = Path(features_path), Path(out_path)
    table = pd.read_parquet(features_path, columns=["subject", "stage"])

    folds, diagnostics = make_folds(table, n_splits, seed)
    manifest = {
        "n_splits": n_splits,
        "seed": seed,
        "n_subjects": int(table["subject"].nunique()),
        "n_epochs": len(table),
        "source": features_path.name,
        "folds": folds,
        "diagnostics": diagnostics,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def load_folds(path: str | Path) -> list[dict[str, list[str]]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["folds"]


def fold_masks(table: pd.DataFrame, fold: dict[str, list[str]]) -> tuple[np.ndarray, np.ndarray]:
    """피험자 이름 목록을 행 마스크로 바꾼다."""
    subjects = table["subject"]
    return subjects.isin(fold["train"]).to_numpy(), subjects.isin(fold["test"]).to_numpy()
