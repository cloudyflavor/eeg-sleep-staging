"""녹음 단위 정규화. 왜 누적 통계인지는 `docs/04-stage2.md` D11.

References
----------
Vallat, R., & Walker, M. P. (2021). An open-source, high-performance tool for automated
sleep staging. eLife, 10, e70092. https://doi.org/10.7554/eLife.70092
"""

import numpy as np
import pandas as pd

#: 통계를 믿기 전에 필요한 과거 길이. 20 에포크 = 10분.
MIN_PERIODS = 20

#: 분위 폭이 이보다 작으면 0 으로 나누는 것과 같아 결측으로 둔다.
_MIN_SCALE = 1e-12


def expanding_robust(feats: pd.DataFrame, min_periods: int = MIN_PERIODS) -> pd.DataFrame:
    """``(x − 중앙값) / (95분위 − 5분위)`` 로 스케일한다. 과거만 쓴다.

    Notes
    -----
    **절단 전 표에 적용해야 한다.** 절단 후에 적용하면 수면 구간 첫 에포크가 과거가
    없는 것처럼 취급되는데, 실제 기기는 착용한 순간부터 쌓고 있다
    (실측 ``crop_lo`` 중앙값 839 에포크 ≈ 7시간).
    """
    exp = feats.expanding(min_periods=min_periods)
    scale = exp.quantile(0.95) - exp.quantile(0.05)
    return (feats - exp.median()) / scale.where(scale > _MIN_SCALE)


def posthoc_robust(feats: pd.DataFrame) -> pd.DataFrame:
    """밤 전체 통계로 스케일한다 (YASA 방식).

    실시간에는 못 쓴다. 도달 가능한 상한선을 재는 용도로만 둔다.
    """
    scale = feats.quantile(0.95) - feats.quantile(0.05)
    return (feats - feats.median()) / scale.where(scale > _MIN_SCALE)


def missing_report(normalized: pd.DataFrame, crop: tuple[int, int]) -> dict[str, int | float]:
    """결측 행 수를 절단 전후로 나눠 센다.

    Notes
    -----
    학습에 들어가는 건 절단 후 결측뿐이다. 이 숫자가 중요한 이유는 **선형 baseline 이
    결측을 못 받기 때문**이다 — 그쪽만 대치하면 모델 비교가 대치 방식을 재게 된다.
    """
    missing = normalized.isna().any(axis=1)
    lo, hi = crop
    after = int(missing.iloc[lo:hi].sum())
    return {
        "rows_before_crop": int(missing.sum()),
        "rows_after_crop": after,
        "ratio_after_crop": round(after / max(hi - lo, 1), 6),
    }


def as_float32(df: pd.DataFrame) -> pd.DataFrame:
    """float32 로 낮춘다. float64 는 skew/kurtosis 시간과 parquet 크기를 둘 다 두 배로 만든다."""
    return df.astype(np.float32)
