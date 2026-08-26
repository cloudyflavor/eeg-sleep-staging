"""밴드패스 필터. 근거: docs/04-stage2.md D10."""

import numpy as np
from scipy import signal as sp_signal

BAND = (0.4, 30.0)
ORDER = 5


def _sos(sfreq: float):
    return sp_signal.butter(ORDER, BAND, btype="bandpass", fs=sfreq, output="sos")


def _causal(signal: np.ndarray, sfreq: float) -> np.ndarray:
    return sp_signal.sosfilt(_sos(sfreq), signal, axis=-1).astype(np.float32)


def _zerophase(signal: np.ndarray, sfreq: float) -> np.ndarray:
    return sp_signal.sosfiltfilt(_sos(sfreq), signal, axis=-1).astype(np.float32)


#: 방식 → (함수, 필요 미래 초). 방식 추가는 여기 한 줄.
FILTERS = {
    "none": (None, 0.0),
    "causal": (_causal, 0.0),
    "zerophase": (_zerophase, 5.0),
}


def apply_filter(signal: np.ndarray, sfreq: float, mode: str) -> np.ndarray:
    """(n_channels, n_samples) 밴드패스. 절단 전(warmup 포함) 신호에 적용할 것."""
    if mode not in FILTERS:
        raise ValueError(f"모르는 필터 방식입니다: {mode!r} ({'/'.join(FILTERS)})")
    fn, _ = FILTERS[mode]
    return signal if fn is None else fn(signal, sfreq)
