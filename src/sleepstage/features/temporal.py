"""시간 영역 특징 — FFT 없이 파형의 모양을 잡는다.

`antropy` 대신 직접 구현했다. antropy 는 1차원만 받아 파이썬 루프를 강제한다.
각 특징이 무엇을 잡는지는 `docs/04-stage2.md` D13.

References
----------
Hjorth, B. (1970). EEG analysis based on time domain properties.
Electroencephalography and Clinical Neurophysiology, 29(3), 306-310.

Aboalayon, K. A. I., Faezipour, M., Almuhammadi, W. S., & Moslehpour, S. (2016).
Sleep stage classification using EEG signal analysis. Entropy, 18(9), 272.
"""

import numpy as np
from scipy import stats as sp_stats


def basic_stats(epochs: np.ndarray) -> dict[str, np.ndarray]:
    q25, q75 = np.quantile(epochs, [0.25, 0.75], axis=-1)
    return {
        "std": epochs.std(axis=-1).astype("float32"),
        "iqr": (q75 - q25).astype("float32"),
        "skew": sp_stats.skew(epochs, axis=-1).astype("float32"),
        "kurt": sp_stats.kurtosis(epochs, axis=-1).astype("float32"),
        "nzc": np.diff(np.signbit(epochs), axis=-1).sum(axis=-1).astype("float32"),
    }


def hjorth(epochs: np.ndarray) -> dict[str, np.ndarray]:
    """Hjorth mobility(≈ 평균 주파수) 와 complexity(≈ 대역폭).

    Activity 는 분산이라 ``std`` 와 같은 것이므로 내지 않는다.
    """
    d1 = np.diff(epochs, axis=-1)
    d2 = np.diff(d1, axis=-1)
    mob = np.sqrt(d1.var(axis=-1) / epochs.var(axis=-1))
    return {
        "hmob": mob.astype("float32"),
        "hcomp": (np.sqrt(d2.var(axis=-1) / d1.var(axis=-1)) / mob).astype("float32"),
    }


def distance_energy(epochs: np.ndarray, n_windows: int = 10) -> dict[str, np.ndarray]:
    """MMD(최대점-최소점 거리)와 Esis(진폭² × 속도²).

    위 특징들과 많이 겹친다. 도움이 된다고 보고 넣는 게 아니라 기여도를 재려고 넣는다.
    """
    n_samp = epochs.shape[-1]
    width = n_samp // n_windows
    win = epochs[..., : width * n_windows].reshape(*epochs.shape[:-1], n_windows, width)

    imax, imin = win.argmax(axis=-1), win.argmin(axis=-1)
    vmax = np.take_along_axis(win, imax[..., None], axis=-1).squeeze(-1)
    vmin = np.take_along_axis(win, imin[..., None], axis=-1).squeeze(-1)
    mmd = np.hypot(vmax - vmin, (imax - imin).astype("float32")).sum(axis=-1)

    velocity = np.diff(epochs, axis=-1)
    esis = (epochs**2).sum(axis=-1) * (velocity**2).sum(axis=-1)
    return {"mmd": mmd.astype("float32"), "esis": esis.astype("float32")}
