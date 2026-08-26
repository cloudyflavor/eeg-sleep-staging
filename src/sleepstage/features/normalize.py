"""녹음 단위 정규화. 근거: docs/04-stage2.md D11."""

import numpy as np
import pandas as pd

MIN_PERIODS = 20  # 이보다 적은 표본의 분위수는 잡음이다
_MIN_SCALE = 1e-12  # 분모 0 방지


def expanding_robust(feats: pd.DataFrame, min_periods: int = MIN_PERIODS) -> pd.DataFrame:
    """(x − 중앙값) / (95분위 − 5분위), 과거 통계만. warmup 포함 창에서 시작할 것."""
    exp = feats.expanding(min_periods=min_periods)
    scale = exp.quantile(0.95) - exp.quantile(0.05)
    return (feats - exp.median()) / scale.where(scale > _MIN_SCALE)


def posthoc_robust(feats: pd.DataFrame) -> pd.DataFrame:
    """밤 전체 통계로 정규화. 실시간 불가 — 상한선 측정용."""
    scale = feats.quantile(0.95) - feats.quantile(0.05)
    return (feats - feats.median()) / scale.where(scale > _MIN_SCALE)


#: 방식 → (함수, lookahead 에포크). None = 밤 전체가 있어야 한다.
NORMALIZERS = {"expanding": (expanding_robust, 0), "posthoc": (posthoc_robust, None)}


def missing_report(normalized: pd.DataFrame, kept_from: int) -> dict[str, int | float]:
    """결측 행 수 — 학습 구간(kept)과 전체를 나눠 센다. 선형 모델 비교 때 필요."""
    missing = normalized.isna().any(axis=1)
    kept = int(missing.iloc[kept_from:].sum())
    return {
        "rows_missing_total": int(missing.sum()),
        "rows_missing_kept": kept,
        "ratio_kept": round(kept / max(len(normalized) - kept_from, 1), 6),
    }


def as_float32(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype(np.float32)
