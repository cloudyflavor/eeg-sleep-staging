"""주파수 영역 특징 — Welch PSD 와 대역파워.

수면 단계의 판정 기준 자체가 주파수로 정의돼 있다 (AASM 채점 매뉴얼).
예: "0.5–2 Hz 서파가 에포크의 20 % 이상이면 N3", "알파가 50 % 미만이면 N1 시작".
그래서 대역파워가 가장 중요한 갈래다.
"""

import numpy as np
from scipy import signal as sp_signal

#: YASA 와 동일한 대역 정의. sleep-linear 도 `yasa.bandpower` 를 그대로 쓴다.
BANDS = {
    "sdelta": (0.4, 1.0),  # 아주 느린 진동
    "fdelta": (1.0, 4.0),  # DS 판정의 결정적 근거
    "theta": (4.0, 8.0),  # N1 의 특징
    "alpha": (8.0, 12.0),  # W 판정 근거 (눈 감은 각성)
    "sigma": (12.0, 16.0),  # 수면방추 = N2 의 결정적 근거
    "beta": (16.0, 30.0),  # 각성 · 근육 긴장
}

#: 광대역. `abspow` 와 상대파워의 분모.
BROAD = (0.4, 30.0)

#: 비율 특징. 절대 파워는 전극 접촉·두개골·나이에 따라 사람마다 몇 배씩 다르지만,
#: 비율은 그 배수가 분자·분모에서 상쇄된다.
RATIOS = {
    "dt": ("delta", "theta"),  # 수면 깊이
    "ds": ("delta", "sigma"),  # DS(델타 우세) vs N2(방추 우세)
    "db": ("delta", "beta"),  # 임상에서도 쓰는 깊이 지표
    "at": ("alpha", "theta"),  # W vs N1 — 이 전환이 N1 정의 그 자체
}


def welch_psd(epochs: np.ndarray, sfreq: float, window_sec: float = 5.0) -> tuple:
    """에포크 배열의 PSD. ``(freqs, psd)`` 를 돌려준다.

    **평균이 아니라 중앙값으로 집계한다** (``average="median"``).
    Welch 는 신호를 겹치는 조각으로 나눠 각각의 스펙트럼을 구한 뒤 합치는데,
    평균을 쓰면 30초 중 1초짜리 큰 잡음이 결과를 끌고 간다. 중앙값은 거의 안 움직인다.

    YASA 가 수면 단계 분류 경로에서 아티팩트 제거를 아예 하지 않고도 견디는 장치가 이것이다.
    우리도 같은 이유로 아티팩트를 제거하지 않고 지표만 저장했다 (1단계 D7).
    """
    return sp_signal.welch(
        epochs,
        sfreq,
        window="hamming",
        nperseg=int(window_sec * sfreq),
        average="median",
        axis=-1,
    )


def band_powers(freqs: np.ndarray, psd: np.ndarray) -> dict[str, np.ndarray]:
    """상대 대역파워 6개 + `abspow` + 비율 4개 = 11개.

    상대파워로 내는 이유는 비율과 같다 — 개인 간 진폭 차이를 나눠서 없앤다.
    `abspow` 만 절대값으로 남긴다 (진폭 수준 자체가 정보다).
    """
    dfreq = float(freqs[1] - freqs[0])

    def integrate(lo: float, hi: float) -> np.ndarray:
        mask = (freqs >= lo) & (freqs < hi)
        return np.trapezoid(psd[..., mask], dx=dfreq, axis=-1)

    total = integrate(*BROAD)
    out = {"abspow": total}
    for name, (lo, hi) in BANDS.items():
        out[name] = integrate(lo, hi) / total

    delta = out["sdelta"] + out["fdelta"]
    parts = {"delta": delta, **out}
    for name, (num, den) in RATIOS.items():
        out[name] = parts[num] / parts[den]

    return {k: v.astype("float32") for k, v in out.items()}
