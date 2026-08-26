"""밴드패스 필터. 세 방식을 비교하는 근거는 `docs/04-stage2.md` D10."""

import numpy as np
from scipy import signal as sp_signal

BAND = (0.4, 30.0)
ORDER = 5

#: 각 방식이 요구하는 미래 시간. 지연 예산 장부에 쓴다.
LOOKAHEAD_SEC = {"none": 0.0, "causal": 0.0, "zerophase": 5.0}


def apply_filter(signal: np.ndarray, sfreq: float, mode: str) -> np.ndarray:
    """``(n_channels, n_samples)`` 에 밴드패스를 건다.

    ``none`` 안 걸음, ``causal`` 과거만 본다(실시간 가능·위상 왜곡),
    ``zerophase`` 양방향으로 건다(위상 보존·미래 필요).

    Notes
    -----
    **절단 전 신호에 걸어야 한다.** 인과 필터는 과거를 참조하므로 절단 후에 걸면
    첫 몇 초가 과거 없이 시작한다. 실제 기기에는 없는 경계 왜곡이다.
    """
    if mode == "none":
        return signal
    sos = sp_signal.butter(ORDER, BAND, btype="bandpass", fs=sfreq, output="sos")
    if mode == "causal":
        out = sp_signal.sosfilt(sos, signal, axis=-1)
    elif mode == "zerophase":
        out = sp_signal.sosfiltfilt(sos, signal, axis=-1)
    else:
        raise ValueError(f"모르는 필터 방식입니다: {mode!r} (none / causal / zerophase)")
    return out.astype(np.float32)
