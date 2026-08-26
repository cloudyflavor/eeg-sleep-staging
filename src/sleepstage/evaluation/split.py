"""피험자 단위 fold 배정을 만들어 파일로 고정한다.

배정을 파일로 굳히는 이유는 **모든 실험이 같은 분할을 써야 비교가 성립하기** 때문이다.
학습할 때마다 다시 계산하면 파일 목록 순서나 라이브러리 버전이 달라졌을 때 조용히
다른 분할이 되고, 그러면 조합 9개가 서로 다른 데이터로 평가된 것이 된다.

References
----------
Memar, P., & Faradji, F. (2018). A novel multi-class EEG-based sleep stage classification
system. IEEE TNSRE, 26(1), 84-95. 같은 데이터·모델로 에포크 단위 CV 95.31 % 대
피험자 단위 CV 86.64 % 를 보고한다.
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
    """피험자를 ``n_splits`` 개 fold 로 나눈다. ``(fold 목록, 진단)``.

    Notes
    -----
    나누는 단위는 **에포크가 아니라 피험자**다. 78명 중 75명이 이틀 밤을 잤으므로
    같은 사람의 두 밤이 학습과 평가로 갈리면 모델이 그 사람을 기억한 채 평가받는다.

    ``StratifiedGroupKFold`` 는 그룹을 유지하면서 클래스 비율도 맞추려 하지만,
    한 사람이 네 클래스를 모두 갖고 있어 완벽한 층화는 불가능하다. 그래서 결과로
    나온 fold 별 클래스 비율을 진단에 담아 **얼마나 치우쳤는지 눈으로 확인**하게 한다.
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
    """fold 사이에서 클래스 비율이 얼마나 흔들리는지.

    희소 클래스(DS 6.7 %)는 한 fold 에 몰리기 쉽다. 그 fold 의 점수만 낮게 나오면
    모델 문제인지 분할 문제인지 구분할 수 없으므로, 미리 폭을 확인해 둔다.
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
    """특징 표에서 fold 를 만들어 JSON 으로 저장한다.

    저장하는 것은 **피험자 이름 목록**이지 행 번호가 아니다. 행 번호는 특징 표가
    바뀌면 의미를 잃지만, 피험자 이름은 어떤 표에도 그대로 적용된다.
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
    """피험자 이름 목록을 행 마스크로 바꾼다."""
    subjects = table["subject"]
    return subjects.isin(fold["train"]).to_numpy(), subjects.isin(fold["test"]).to_numpy()
