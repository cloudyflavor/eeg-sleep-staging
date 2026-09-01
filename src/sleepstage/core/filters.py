"""뇌파 대역 통과 필터.

수면 판정에 쓰는 0.4~30Hz 만 남기고 전극 흔들림(저주파)과 근육 잡음(고주파)을 깎는다.
"""

import numpy as np
from scipy import signal as sp_signal

BAND = (0.4, 30.0)
ORDER = 5


def _sos(sfreq: float):
    return sp_signal.butter(ORDER, BAND, btype="bandpass", fs=sfreq, output="sos")


def _causal(signal: np.ndarray, sfreq: float) -> np.ndarray:
    """과거 방향으로만 거른다. 파형이 뒤로 밀리지만 미래 데이터가 필요 없다."""
    return sp_signal.sosfilt(_sos(sfreq), signal, axis=-1).astype(np.float32)


def _zerophase(signal: np.ndarray, sfreq: float) -> np.ndarray:
    """앞뒤 양방향으로 걸러 밀림을 상쇄한다. 뒤쪽 데이터가 조금 필요하다."""
    return sp_signal.sosfiltfilt(_sos(sfreq), signal, axis=-1).astype(np.float32)


#: 방식 이름 -> (함수, 필요한 미래 시간(초))
FILTERS = {
    "none": (None, 0.0),
    "causal": (_causal, 0.0),
    "zerophase": (_zerophase, 5.0),
}


def apply_filter(signal: np.ndarray, sfreq: float, mode: str) -> np.ndarray:
    """(채널, 샘플) 신호에 대역 통과 필터를 건다.

    잘라내기 전 신호에 적용해야 한다. 잘라낸 뒤에 걸면 시작 부분이 과거 없이
    시작해 값이 왜곡된다.
    """
    if mode not in FILTERS:
        raise ValueError(f"모르는 필터 방식입니다: {mode!r} ({'/'.join(FILTERS)})")
    fn, _ = FILTERS[mode]
    return signal if fn is None else fn(signal, sfreq)
