"""파형의 모양을 숫자로 요약한다.

주파수 분석 없이 계산하므로 훨씬 싸다. 진폭이 얼마나 큰지, 얼마나 뾰족한지,
얼마나 불규칙한지 같은 성질을 잡는다.
"""

import numpy as np
from scipy import signal as sp_signal
from scipy import stats as sp_stats


def basic_stats(epochs: np.ndarray) -> dict[str, np.ndarray]:
    """진폭 분포의 기본 통계.

    std 와 iqr 은 진폭 크기, skew 는 위아래 비대칭, kurt 는 뾰족함,
    nzc 는 0선을 지나간 횟수로 파형이 얼마나 빠른지를 나타낸다.
    """
    q25, q75 = np.quantile(epochs, [0.25, 0.75], axis=-1)
    return {
        "std": epochs.std(axis=-1).astype("float32"),
        "iqr": (q75 - q25).astype("float32"),
        "skew": sp_stats.skew(epochs, axis=-1).astype("float32"),
        "kurt": sp_stats.kurtosis(epochs, axis=-1).astype("float32"),
        "nzc": np.diff(np.signbit(epochs), axis=-1).sum(axis=-1).astype("float32"),
    }


def hjorth(epochs: np.ndarray) -> dict[str, np.ndarray]:
    """파형의 평균 속도와 복잡도.

    미분의 분산을 이용해 주파수 정보를 근사한다. hmob 은 대략 평균 주파수,
    hcomp 은 주파수가 얼마나 넓게 퍼져 있는지를 나타낸다.
    """
    d1 = np.diff(epochs, axis=-1)
    d2 = np.diff(d1, axis=-1)
    mob = np.sqrt(d1.var(axis=-1) / epochs.var(axis=-1))
    return {
        "hmob": mob.astype("float32"),
        "hcomp": (np.sqrt(d2.var(axis=-1) / d1.var(axis=-1)) / mob).astype("float32"),
    }


def distance_energy(epochs: np.ndarray, n_windows: int = 10) -> dict[str, np.ndarray]:
    """구간별 진폭 변화폭과 신호 에너지.

    에포크를 작은 구간으로 나눠 각 구간의 최고점과 최저점 사이 거리를 재고(mmd),
    진폭과 변화 속도를 곱해 에너지를 낸다(esis).
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


def band_correlation(front: np.ndarray, back: np.ndarray, sfreq: float, bands: dict) -> dict:
    """두 채널의 파형이 대역마다 얼마나 닮았는지 낸다.

    지금 채널 비교 특징은 세기 비율뿐이라 "이마가 더 세다" 까지만 안다. 서파도
    이마가 더 세고 안구운동도 이마가 더 세서 그것만으로는 안 갈린다.

    닮았는지를 함께 보면 갈린다. 서파는 머리 전체가 같이 출렁여 두 채널이 닮고,
    이마에서만 나는 사건은 안 닮는다. 방추도 두 채널에 같이 나타나므로 시그마
    대역이 닮았는지가 곧 방추가 있었는지가 된다.

    실측 R 대 LS 판별력은 시그마 0.686, 안구운동 대역 0.641, 저주파 0.631 이다.
    """
    out = {"corr_broad": _pearson(front, back)}
    for name, (lo, hi) in bands.items():
        sos = sp_signal.butter(4, (lo, hi), btype="bandpass", fs=sfreq, output="sos")
        out[f"corr_{name}"] = _pearson(
            sp_signal.sosfiltfilt(sos, front, axis=-1), sp_signal.sosfiltfilt(sos, back, axis=-1)
        )
    return out


def _pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ac = a - a.mean(axis=-1, keepdims=True)
    bc = b - b.mean(axis=-1, keepdims=True)
    scale = np.sqrt((ac**2).sum(axis=-1) * (bc**2).sum(axis=-1))
    # 한쪽이 완전히 평평하면 닮았다고도 아니라고도 할 수 없다
    return np.where(
        scale > 0, (ac * bc).sum(axis=-1) / np.where(scale > 0, scale, 1), np.nan
    ).astype("float32")
