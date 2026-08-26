"""시간 영역 특징. 정의·근거: docs/04-stage2.md D13, docs/06-primer.md §5."""

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
    """Hjorth mobility·complexity. Activity 는 std 와 같아 내지 않는다."""
    d1 = np.diff(epochs, axis=-1)
    d2 = np.diff(d1, axis=-1)
    mob = np.sqrt(d1.var(axis=-1) / epochs.var(axis=-1))
    return {
        "hmob": mob.astype("float32"),
        "hcomp": (np.sqrt(d2.var(axis=-1) / d1.var(axis=-1)) / mob).astype("float32"),
    }


def distance_energy(epochs: np.ndarray, n_windows: int = 10) -> dict[str, np.ndarray]:
    """MMD 와 Esis — 기여도 측정 대상."""
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
