"""짧게 나타나는 사건을 세어 특징으로 만듭니다.

30초 평균으로는 순간적인 사건이 지워집니다. 1초짜리 방추 하나는 30분의 1로
희석되어 평균을 거의 움직이지 못합니다. 실제로 재보면 얕은 잠은 조각당 방추가
평균 4개이고 REM 은 0.1개인데, 시그마 대역의 평균 세기로는 이 차이가 흐려집니다.

문턱은 지나온 조각들로만 정합니다. 밤 전체를 보고 정하면 실제 기기에서 쓸 수
없습니다. 사람마다 뇌파 진폭이 몇 배씩 다르므로 고정된 값도 쓸 수 없습니다.
"""

import numpy as np
from scipy import signal as sp_signal

#: 수면방추가 나타나는 주파수
SPINDLE_BAND = (11.0, 16.0)

#: 방추로 인정하는 길이. 짧으면 잡음이고 길면 다른 파형이다.
SPINDLE_SECONDS = (0.5, 2.0)

#: 문턱으로 쓸 분위수
PERCENTILE = 92

#: 문턱을 다시 계산하는 주기. 매 조각마다 하면 느리고 얻는 것이 없다.
REFRESH = 20

#: 문턱 계산에 쓰는 표본 간격. 전부 쓰면 밤이 깊어질수록 느려진다.
DECIMATE = 5


def envelope(signal: np.ndarray, sfreq: float, band=SPINDLE_BAND) -> np.ndarray:
    """해당 주파수 성분이 순간마다 얼마나 센지를 낸다."""
    sos = sp_signal.butter(4, band, btype="bandpass", fs=sfreq, output="sos")
    return np.abs(sp_signal.hilbert(sp_signal.sosfiltfilt(sos, signal)))


def count_runs(env: np.ndarray, threshold: float, sfreq: float, seconds) -> tuple[int, float]:
    """문턱을 넘어 정해진 시간만큼 이어지는 구간의 개수와 총 길이를 센다."""
    above = env > threshold
    if not above.any():
        return 0, 0.0
    edges = np.diff(above.astype(np.int8))
    starts = np.flatnonzero(edges == 1) + 1
    ends = np.flatnonzero(edges == -1) + 1
    if above[0]:
        starts = np.r_[0, starts]
    if above[-1]:
        ends = np.r_[ends, len(above)]
    dur = (ends - starts) / sfreq
    keep = (dur >= seconds[0]) & (dur <= seconds[1])
    return int(keep.sum()), float(dur[keep].sum())


class SpindleCounter:
    """조각을 순서대로 받아 그 조각의 방추를 셉니다.

    학습과 서빙이 같은 값을 내야 하므로 양쪽이 이 객체를 똑같이 씁니다.
    조각을 건너뛰거나 순서를 바꿔 넣으면 문턱이 달라져 값이 조용히 어긋납니다.
    """

    def __init__(self, sfreq: float, band=SPINDLE_BAND, seconds=SPINDLE_SECONDS):
        self.sfreq = sfreq
        self.band = band
        self.seconds = seconds
        self.seen: list[np.ndarray] = []
        self.threshold: float | None = None
        self.n = 0

    def update(self, epoch: np.ndarray) -> tuple[int, float]:
        """조각 하나를 넣고 그 안의 방추 개수와 총 길이를 돌려준다."""
        env = envelope(epoch, self.sfreq, self.band)
        self.seen.append(env[::DECIMATE])
        self.n += 1
        if self.threshold is None or self.n % REFRESH == 0:
            self.threshold = float(np.percentile(np.concatenate(self.seen), PERCENTILE))
        return count_runs(env, self.threshold, self.sfreq, self.seconds)


def spindles_for_recording(epochs: np.ndarray, sfreq: float) -> np.ndarray:
    """녹음 하나의 모든 조각에 대해 방추 개수와 길이를 낸다.

    돌려주는 배열의 모양은 (조각 수, 채널 수, 2) 이고 마지막 축은 개수와 길이다.
    """
    n_ep, n_ch, _ = epochs.shape
    out = np.zeros((n_ep, n_ch, 2), dtype="float32")
    for ch in range(n_ch):
        counter = SpindleCounter(sfreq)
        for i in range(n_ep):
            out[i, ch] = counter.update(epochs[i, ch])
    return out
