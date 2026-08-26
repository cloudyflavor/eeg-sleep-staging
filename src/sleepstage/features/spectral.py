"""주파수 영역 특징 — Welch PSD 와 대역파워.

대역 정의와 비율을 쓰는 이유는 `docs/04-stage2.md` D13.
"""

import numpy as np
from scipy import signal as sp_signal

#: YASA 와 같은 경계. sleep-linear 은 `yasa.bandpower` 를 그대로 import 한다.
BANDS = {
    "sdelta": (0.4, 1.0),
    "fdelta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),
    "beta": (16.0, 30.0),
}

BROAD = (0.4, 30.0)

RATIOS = {
    "dt": ("delta", "theta"),
    "ds": ("delta", "sigma"),
    "db": ("delta", "beta"),
    "at": ("alpha", "theta"),
}


def welch_psd(epochs: np.ndarray, sfreq: float, window_sec: float = 5.0) -> tuple:
    """에포크별 PSD. 조각을 평균이 아니라 **중앙값**으로 합친다.

    1초짜리 잡음이 스펙트럼을 끌고 가지 못하므로 아티팩트 제거를 건너뛸 수 있다.
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
    """상대 대역파워 6개, 총파워 1개, 비율 4개. ``abspow`` 만 절대값이다."""
    dfreq = float(freqs[1] - freqs[0])

    def integrate(lo: float, hi: float) -> np.ndarray:
        mask = (freqs >= lo) & (freqs < hi)
        return np.trapezoid(psd[..., mask], dx=dfreq, axis=-1)

    total = integrate(*BROAD)
    out = {"abspow": total}
    for name, (lo, hi) in BANDS.items():
        out[name] = integrate(lo, hi) / total

    parts = {"delta": out["sdelta"] + out["fdelta"], **out}
    for name, (num, den) in RATIOS.items():
        out[name] = parts[num] / parts[den]

    return {k: v.astype("float32") for k, v in out.items()}
