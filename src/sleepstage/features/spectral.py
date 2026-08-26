"""주파수 영역 특징 — Welch PSD 와 대역파워. 근거: docs/04-stage2.md D13."""

import numpy as np
from scipy import signal as sp_signal

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
    """에포크별 PSD. 조각 집계는 중앙값 — 잡음 1초가 스펙트럼을 못 끌게 한다."""
    return sp_signal.welch(
        epochs,
        sfreq,
        window="hamming",
        nperseg=int(window_sec * sfreq),
        average="median",
        axis=-1,
    )


def band_powers(freqs: np.ndarray, psd: np.ndarray) -> dict[str, np.ndarray]:
    """상대 대역파워 6, 총파워 1, 비율 4. ``abspow`` 만 절대값."""
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
