"""시간 영역 특징 — 파형의 모양을 숫자로 요약한다.

FFT 없이 계산되므로 주파수 영역 특징보다 훨씬 싸다. 웨어러블 관점에서 중요한 성질이다.

`antropy` 에 의존하지 않고 직접 구현한다. 이유 두 가지:

1. **`antropy` 의 엔트로피·프랙탈 함수는 1차원만 받아** ``np.apply_along_axis`` 로 루프를
   돌아야 한다. 벡터화하면 배치 처리가 크게 빨라진다.
2. Hjorth·MMD·Esis 는 정의가 짧아서, 직접 쓰는 편이 "이 숫자가 무엇인지" 를 드러낸다.

측정 결과(2026-08-26, SC400N1): 에포크 하나당 2채널 전체가 11.7 ms.
30초 예산의 0.039 % 라 **연산 비용은 실시간 제약이 되지 않는다.**
"""

import numpy as np
from scipy import stats as sp_stats


def basic_stats(epochs: np.ndarray) -> dict[str, np.ndarray]:
    """진폭 분포의 모양. ``(n_epochs, n_samples)`` → 각 5개.

    - ``std`` 진폭 크기. 깊은 수면은 진폭이 크다
    - ``iqr`` std 와 비슷하나 **이상치에 강하다.** 잡음 1초가 껴도 안 흔들린다
    - ``skew`` 위아래 비대칭. K-복합체 같은 비대칭 파형을 잡는다
    - ``kurt`` 뾰족함. **K-복합체·방추 같은 순간적 사건**이 있으면 높아진다
    - ``nzc`` 0선을 몇 번 지나가나 = **주파수의 초싼 대용품**

    ``float32`` 를 유지한다 — float64 로 캐스팅하면 scipy 의 skew/kurtosis 가 두 배 느려진다.
    """
    q25, q75 = np.quantile(epochs, [0.25, 0.75], axis=-1)
    return {
        "std": epochs.std(axis=-1).astype("float32"),
        "iqr": (q75 - q25).astype("float32"),
        "skew": sp_stats.skew(epochs, axis=-1).astype("float32"),
        "kurt": sp_stats.kurtosis(epochs, axis=-1).astype("float32"),
        "nzc": (np.diff(np.signbit(epochs), axis=-1).sum(axis=-1)).astype("float32"),
    }


def hjorth(epochs: np.ndarray) -> dict[str, np.ndarray]:
    """Hjorth 파라미터 (Hjorth 1970). 분산 계산 몇 번으로 주파수 정보를 근사한다.

    - ``hmob`` mobility = √(1차 미분의 분산 / 원 분산) ≈ **평균 주파수**
    - ``hcomp`` complexity = mobility(미분) / mobility(원본) ≈ **주파수 대역폭**

    Activity(=분산)는 :func:`basic_stats` 의 ``std`` 와 같은 것이라 따로 내지 않는다.

    **실측한 중복도** (SC400N1, 상대 대역파워와의 상관):
    ``hmob`` ↔ beta **0.84**, ``nzc`` ↔ beta **0.86** — 셋 다 "얼마나 빠른 파형인가" 를
    재므로 사실상 겹친다. 반면 ``hcomp`` 은 어느 대역과도 |r| < 0.6 이라 독립적이다.
    트리 모델은 상관된 특징이 있어도 틀리지는 않지만, 차원이 늘고 중요도가 분산된다.
    → **ablation 으로 확인할 지점이다.**
    """
    d1 = np.diff(epochs, axis=-1)
    d2 = np.diff(d1, axis=-1)
    v0 = epochs.var(axis=-1)
    v1 = d1.var(axis=-1)
    v2 = d2.var(axis=-1)
    mob = np.sqrt(v1 / v0)
    return {
        "hmob": mob.astype("float32"),
        "hcomp": (np.sqrt(v2 / v1) / mob).astype("float32"),
    }


def distance_energy(epochs: np.ndarray, n_windows: int = 10) -> dict[str, np.ndarray]:
    """MMD 와 Esis — 선행 연구의 시간영역 특징.

    출처: Aboalayon et al., *Entropy* 18(9):272, 2016.

    - ``mmd`` Maximum-Minimum Distance. 에포크를 ``n_windows`` 개로 쪼개고, 각 창에서
      최댓값 지점과 최솟값 지점 사이의 피타고라스 거리 √(Δ진폭² + Δ시간²) 를 구해 합한다
    - ``esis`` EnergySis. 진폭² 합 × 속도² 합

    **기여도를 측정하는 대상이지 당연히 넣는 특징이 아니다.** 정의상 기존 특징과 겹칠 여지가 크다.

    ``mmd`` ≈ 진폭 범위(``std``·``iqr``) + 시간 간격(``nzc`` 와 비슷한 축)
    ``esis`` ≈ Hjorth Activity × Mobility

    올라가면 개선이고 안 올라가면 중복임을 실증한 것이다. 둘 다 결과다.
    비용은 거의 0 이고 **미래를 보지 않는다** (lookahead 0).
    """
    n_ep, n_samp = epochs.shape[0], epochs.shape[-1]
    width = n_samp // n_windows
    win = epochs[..., : width * n_windows].reshape(*epochs.shape[:-1], n_windows, width)

    imax = win.argmax(axis=-1)
    imin = win.argmin(axis=-1)
    vmax = np.take_along_axis(win, imax[..., None], axis=-1).squeeze(-1)
    vmin = np.take_along_axis(win, imin[..., None], axis=-1).squeeze(-1)
    mmd = np.hypot(vmax - vmin, (imax - imin).astype("float32")).sum(axis=-1)

    velocity = np.diff(epochs, axis=-1)
    esis = (epochs**2).sum(axis=-1) * (velocity**2).sum(axis=-1)

    assert n_ep == mmd.shape[0]
    return {"mmd": mmd.astype("float32"), "esis": esis.astype("float32")}
