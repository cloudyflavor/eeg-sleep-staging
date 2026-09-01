"""데이터를 사람 단위로 나눠 파일에 고정한다.

한 사람이 여러 밤을 잤을 수 있다. 같은 사람의 데이터가 학습과 평가에 나뉘어
들어가면 모델이 그 사람을 기억한 채로 평가받게 되어 성능이 부풀려진다.

나눈 결과를 파일로 남기는 이유는 모든 실험이 같은 기준으로 평가받아야 하기
때문이다. 실행할 때마다 다시 나누면 조건이 달라져 비교가 무의미해진다.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from sleepstage.core.epochs import STAGE_NAMES


def make_folds(
    table: pd.DataFrame,
    n_splits: int = 10,
    seed: int = 0,
) -> tuple[list[dict[str, list[str]]], dict[str, Any]]:
    """사람들을 n_splits 개 묶음으로 나눈다.

    한 사람이 학습과 평가 양쪽에 들어가면 오류를 낸다. 묶음마다 수면 단계 비율이
    얼마나 다른지도 함께 돌려준다.
    """
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
    """묶음별로 각 수면 단계의 비율이 얼마나 차이 나는지 요약한다.

    드문 단계가 한 묶음에 몰리면 그 묶음만 점수가 낮게 나와 해석이 어려워진다.
    """
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
    """나눈 결과를 파일로 저장한다.

    행 번호가 아니라 사람 이름을 저장한다. 행 번호는 데이터가 바뀌면 의미를
    잃지만 이름은 어떤 데이터에도 그대로 적용된다.
    """
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
    """사람 이름 목록을 표의 행 선택 조건으로 바꾼다."""
    subjects = table["subject"]
    return subjects.isin(fold["train"]).to_numpy(), subjects.isin(fold["test"]).to_numpy()
