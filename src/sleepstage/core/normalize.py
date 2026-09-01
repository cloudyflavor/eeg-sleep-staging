"""사람마다 다른 뇌파 진폭을 같은 기준으로 맞춘다.

같은 델타파 세기 40 이 어떤 사람에게는 깊은 수면이고 다른 사람에게는 얕은 수면일
수 있다. 각자의 그날 밤 분포에서 어느 위치인지로 바꿔 이 차이를 없앤다.
"""

import numpy as np
import pandas as pd

#: 통계를 믿기 전에 필요한 최소 표본 수, 20 에포크는 10분이다
MIN_PERIODS = 20

#: 분포 폭이 이보다 좁으면 0 으로 나누는 것과 같아 결측으로 둔다
_MIN_SCALE = 1e-12


def expanding_robust(feats: pd.DataFrame, min_periods: int = MIN_PERIODS) -> pd.DataFrame:
    """지금까지 쌓인 값만으로 기준을 잡아 변환한다.

    미래를 보지 않으므로 실시간으로 계산할 수 있다. 밤이 깊어질수록 표본이 쌓여
    밤 전체로 계산한 값에 가까워진다.
    """
    exp = feats.expanding(min_periods=min_periods)
    scale = exp.quantile(0.95) - exp.quantile(0.05)
    return (feats - exp.median()) / scale.where(scale > _MIN_SCALE)


def posthoc_robust(feats: pd.DataFrame) -> pd.DataFrame:
    """밤 전체를 보고 기준을 잡아 변환한다. 녹음이 끝난 뒤에만 계산할 수 있다."""
    scale = feats.quantile(0.95) - feats.quantile(0.05)
    return (feats - feats.median()) / scale.where(scale > _MIN_SCALE)


#: 방식 이름 -> (함수, 필요한 미래 에포크 수). None 은 밤 전체가 있어야 한다는 뜻이다.
NORMALIZERS = {"expanding": (expanding_robust, 0), "posthoc": (posthoc_robust, None)}


def missing_report(normalized: pd.DataFrame, kept_from: int) -> dict[str, int | float]:
    """변환하지 못한 행이 몇 개인지 센다.

    표본이 부족한 초반 구간에서 결측이 생긴다. 학습에 실제로 들어가는 구간의
    결측 수를 따로 세는 이유는 결측을 받지 못하는 모델이 있기 때문이다.
    """
    missing = normalized.isna().any(axis=1)
    kept = int(missing.iloc[kept_from:].sum())
    return {
        "rows_missing_total": int(missing.sum()),
        "rows_missing_kept": kept,
        "ratio_kept": round(kept / max(len(normalized) - kept_from, 1), 6),
    }


def as_float32(df: pd.DataFrame) -> pd.DataFrame:
    """저장 크기와 계산 속도를 위해 정밀도를 낮춘다."""
    return df.astype(np.float32)
