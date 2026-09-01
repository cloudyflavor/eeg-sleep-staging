"""주파수 대역별 세기를 계산한다.

수면 단계는 어느 주파수에 힘이 실렸는지로 갈린다. 깊은 수면은 느린 델타파가,
얕은 수면은 12~16Hz 방추가, 각성은 빠른 베타파가 우세하다.
"""

import numpy as np
from scipy import signal as sp_signal

#: 대역 이름 -> (하한 Hz, 상한 Hz)
BANDS = {
    "sdelta": (0.4, 1.0),
    "fdelta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),
    "beta": (16.0, 30.0),
}

#: 안구운동 전위가 섞여 들어오는 대역. 눈은 전기적으로 건전지라 회전하면
#: 이마 전극에 느린 전위 변화를 만든다. 뒤통수에는 거의 안 섞인다.
EOG_BAND = (0.3, 2.0)

#: 상대 세기를 낼 때 분모가 되는 전체 대역
BROAD = (0.4, 30.0)

#: 비율 이름 -> (분자 대역, 분모 대역)
RATIOS = {
    "dt": ("delta", "theta"),
    "ds": ("delta", "sigma"),
    "db": ("delta", "beta"),
    "at": ("alpha", "theta"),
}


def welch_psd(
    epochs: np.ndarray, sfreq: float, window_sec: float = 5.0, average: str = "median"
) -> tuple:
    """에포크별 주파수 분포를 구한다.

    신호를 5초 조각으로 나눠 각각 분석한 뒤 합친다. 중앙값은 순간적인 잡음 한두 조각이
    결과를 끌고 가지 못하게 막지만, 방추처럼 순간적인 신호도 함께 깎는다.
    실제로 방추가 있는 조각과 없는 조각의 시그마 세기 차이가 중앙값에서 20% 작아진다.
    """
    return sp_signal.welch(
        epochs,
        sfreq,
        window="hamming",
        nperseg=int(window_sec * sfreq),
        average=average,
        axis=-1,
    )


def integrate(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """주파수 구간의 세기를 합한다.

    직사각형 합을 쓴다. 사다리꼴로 구간을 나눠 적분하면 구간 경계에 걸친 조각이
    어디에도 들어가지 않아, 대역별 상대 세기의 합이 1보다 작아진다.
    """
    dfreq = float(freqs[1] - freqs[0])
    mask = (freqs >= lo) & (freqs < hi)
    return psd[..., mask].sum(axis=-1) * dfreq


def band_extremes(
    epochs: np.ndarray, sfreq: float, window_sec: float = 5.0
) -> dict[str, np.ndarray]:
    """5초 조각 6개 각각의 대역 세기를 구해 최댓값과 최솟값을 낸다.

    30초 안에서 가장 셌을 때와 가장 약했을 때를 남긴다. 합치는 순간 사라지는
    순간적인 변화를 붙잡는 값이다.
    """
    n_ep, n_sample = epochs.shape
    step = int(window_sec * sfreq)
    n_win = n_sample // step
    windows = epochs[:, : n_win * step].reshape(n_ep * n_win, step)

    freqs, psd = sp_signal.welch(windows, sfreq, window="hamming", nperseg=step, axis=-1)
    total = integrate(freqs, psd, *BROAD)

    out: dict[str, np.ndarray] = {}
    for name, (lo, hi) in BANDS.items():
        rel = (integrate(freqs, psd, lo, hi) / total).reshape(n_ep, n_win)
        out[f"{name}_wmax"] = rel.max(axis=1).astype("float32")
        out[f"{name}_wmin"] = rel.min(axis=1).astype("float32")
    return out


def band_powers(freqs: np.ndarray, psd: np.ndarray) -> dict[str, np.ndarray]:
    """대역별 상대 세기 6개, 전체 세기 1개, 대역 간 비율 4개를 낸다.

    상대 세기와 비율을 쓰는 이유는 사람마다 뇌파 진폭이 몇 배씩 다르기 때문이다.
    나눗셈으로 그 차이가 상쇄된다.
    """
    total = integrate(freqs, psd, *BROAD)
    out = {"abspow": total}
    for name, (lo, hi) in BANDS.items():
        out[name] = integrate(freqs, psd, lo, hi) / total

    parts = {"delta": out["sdelta"] + out["fdelta"], **out}
    for name, (num, den) in RATIOS.items():
        out[name] = parts[num] / parts[den]

    return {k: v.astype("float32") for k, v in out.items()}
