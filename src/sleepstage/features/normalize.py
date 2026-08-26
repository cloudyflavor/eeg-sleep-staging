"""녹음 단위 정규화 — 개인 간 진폭 차이를 나눠서 없앤다.

사람마다 뇌파 진폭이 다르다. eLife 논문(Vallat & Walker 2021)의 표현으로
*"each individual has a unique EEG fingerprint"* 다. 절대 델타파워 40 이
A 씨에겐 깊은 수면이고 B 씨에겐 얕은 수면일 수 있다.

그래서 **원본과 정규화본을 둘 다** 낸다. 논문 근거:

- 정규화본 — 개인차를 흡수한다
- 원본 — *"increase the temporal specificity and keep absolute values that can be
  compared across individuals"*

**단, YASA 처럼 밤 전체 통계를 쓰지 않는다.** 중앙값·분위수는 전체를 줄 세워야 나오는 값이라
그 밤이 끝나야 계산된다. 실시간에는 존재하지 않는다.
→ **누적(expanding) 통계**로 간다. lookahead 0 이면서, 밤이 깊어질수록 사후 값에 수렴한다.
"""

import numpy as np
import pandas as pd

#: 통계가 이만큼 쌓이기 전에는 결측으로 둔다. 20 에포크 = 10분.
#: 표본 두세 개로 낸 분위수는 값이 아니라 잡음이다.
MIN_PERIODS = 20

#: 분위 폭이 이보다 작으면 0 으로 나누는 것과 같다 → 결측으로 둔다.
_MIN_SCALE = 1e-12


def expanding_robust(feats: pd.DataFrame, min_periods: int = MIN_PERIODS) -> pd.DataFrame:
    """누적 robust 정규화. ``(x − 중앙값) / (95분위 − 5분위)``, 과거만 사용.

    `sklearn.preprocessing.robust_scale` 과 같은 식이지만 통계 범위가 다르다.

    =========== ================= ============ ==========================
    방식        통계 범위          lookahead    문제
    =========== ================= ============ ==========================
    사후 (YASA) 밤 전체            밤 끝까지    실시간 불가
    **누적**    1 ~ 지금           **0**        밤 초반 불안정
    이동창      최근 N            0            사후에 수렴하지 않는다
    =========== ================= ============ ==========================

    누적을 고른 이유는 **밤이 깊어질수록 사후 값에 수렴하기 때문**이다.
    에포크 600 시점이면 표본 600개라 사후와 거의 같다. 이동창은 아무리 시간이 지나도
    최근 구간만 본다 — "실시간의 대가" 가 밤 전체에 퍼지는 대신 **밤 초반에 집중**된다.

    .. important::
       **절단(crop) 전 전체 배열에 적용해야 한다.** 절단 후에 적용하면 절단 구간의 첫
       에포크가 "과거가 없는" 것처럼 취급되는데, 실제 기기는 착용한 순간부터 쌓고 있다.
       실측상 녹음 153개의 ``crop_lo`` 중앙값이 839 라, 절단 전에 적용하면 151개 녹음에서
       결측이 아예 생기지 않는다.
    """
    exp = feats.expanding(min_periods=min_periods)
    center = exp.median()
    scale = exp.quantile(0.95) - exp.quantile(0.05)
    scale = scale.where(scale > _MIN_SCALE)  # 분모 0 → 결측
    return (feats - center) / scale


def posthoc_robust(feats: pd.DataFrame) -> pd.DataFrame:
    """밤 전체 통계로 정규화 — YASA 방식.

    **실시간에는 쓸 수 없다.** 도달 가능한 상한선을 재는 용도로만 둔다.
    :func:`expanding_robust` 와의 차이가 곧 "실시간의 대가" 다.
    """
    center = feats.median()
    scale = feats.quantile(0.95) - feats.quantile(0.05)
    scale = scale.where(scale > _MIN_SCALE)
    return (feats - center) / scale


def missing_report(normalized: pd.DataFrame, crop: tuple[int, int]) -> dict[str, int | float]:
    """결측이 얼마나 생겼는지. **절단 전후를 나눠서** 센다.

    절단 전 결측은 대부분 warmup 구간(첫 ``min_periods`` 에포크)이고,
    절단하면 사라진다 — 실측상 녹음 153개의 ``crop_lo`` 중앙값이 839 라
    warmup 이 절단 구간에 들어오는 녹음은 2개뿐이다.

    **절단 후 결측이 실제로 학습에 들어가는 것**이라 이 숫자가 중요하다.
    결측 자체가 "밤 초반" 이라는 신호가 되고(결측 여부 = 에포크 인덱스 < min_periods),
    무엇보다 **선형 baseline 은 결측을 받지 못한다** — 별도 대치를 하면 부스팅과 서로 다른
    데이터를 보게 되어 "모델 차이" 인지 "결측 처리 차이" 인지 구분할 수 없게 된다.
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
    """float32 로 낮춰 저장한다.

    측정 결과 float64 캐스팅은 scipy 의 skew/kurtosis 를 두 배 느리게 하고,
    parquet 크기도 두 배가 된다. 특징 값의 유효 자릿수에 float32 로 충분하다.
    """
    return df.astype(np.float32)
