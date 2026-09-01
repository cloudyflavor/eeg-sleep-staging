"""신경망에 넣을 데이터를 준비한다.

특징을 뽑지 않고 신호를 그대로 넣는다. 신경망이 어떤 패턴을 볼지 스스로 배우게
하는 방식이라, 사람이 정한 특징과 성능을 비교할 수 있다.

비교가 성립하려면 입력 형태 말고는 전부 같아야 한다. 그래서 나누는 기준, 잘라내는
범위, 거는 필터, 진폭을 맞추는 방식까지 특징 쪽과 똑같이 맞춘다.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sleepstage.core import filters
from sleepstage.experiment.pipeline.features import WARMUP_EPOCHS

#: 진폭 폭을 잴 때 쓰는 분포 범위
_Q = (5, 95)

#: 0 으로 나누는 것을 막는 하한
_MIN_SCALE = 1e-6


def _expanding_normalize(epochs: np.ndarray) -> np.ndarray:
    """지금까지 지나온 조각들만으로 기준을 잡아 진폭을 맞춘다.

    미래를 보지 않으므로 실제 기기에서도 같은 값을 만들 수 있다. 특징 쪽에서
    쓰는 방식과 원리가 같아서, 두 방식의 차이가 입력 형태에서만 오게 된다.
    """
    center = np.median(epochs, axis=2)
    lo, hi = np.percentile(epochs, _Q, axis=2)
    spread = np.maximum(hi - lo, _MIN_SCALE)

    def running(a):
        return pd.DataFrame(a).expanding(min_periods=1).median().to_numpy()

    run_center = running(center)[:, :, None]
    run_scale = np.maximum(running(spread), _MIN_SCALE)[:, :, None]
    return ((epochs - run_center) / run_scale).astype("float32")


def load_recording(
    path: Path, filter_mode: str = "none", normalize_mode: str = "expanding"
) -> tuple[np.ndarray, np.ndarray, str]:
    """녹음 하나에서 신호와 라벨과 피험자 번호를 읽는다.

    본 구간 앞의 준비 구간까지 함께 읽어 기준값을 쌓은 뒤 본 구간만 남긴다.
    준비 구간이 없으면 밤 초반의 기준값이 표본 부족으로 흔들린다.
    """
    z = np.load(path, allow_pickle=False)
    sfreq = float(z["sfreq"])
    lo, hi = int(z["crop_lo"]), int(z["crop_hi"])
    start = max(0, lo - WARMUP_EPOCHS)
    keep_from = lo - start

    epochs = z["data"][start:hi]
    labels = z["labels"][start:hi]

    if filter_mode != "none":
        n_ch = epochs.shape[1]
        joined = epochs.transpose(1, 0, 2).reshape(n_ch, -1)
        epochs = (
            filters.apply_filter(joined, sfreq, filter_mode)
            .reshape(n_ch, len(labels), -1)
            .transpose(1, 0, 2)
        )
    if normalize_mode != "none":
        epochs = _expanding_normalize(epochs)

    return epochs[keep_from:].astype("float32"), labels[keep_from:], str(z["subject"])


class EpochDataset(Dataset):
    """30초 조각 하나를 학습 표본 하나로 다룬다."""

    def __init__(self, signals: np.ndarray, labels: np.ndarray):
        self.signals = signals
        self.labels = labels.astype("int64")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        return torch.from_numpy(self.signals[i]), int(self.labels[i])


class SequenceDataset(Dataset):
    """이어진 조각 여러 개를 학습 표본 하나로 다룬다.

    묶음은 녹음 안에서만 만든다. 다른 사람의 밤이 한 묶음에 섞이면 누수다.

    묶음마다 채점할 자리를 함께 들고 다닌다. 가운데 조각만 채점하면 녹음의 처음과
    끝이 어디에서도 채점되지 않으므로, 양 끝 묶음이 그 자리까지 맡는다. 이렇게
    하면 모든 조각이 정확히 한 번씩 채점된다.
    """

    def __init__(
        self,
        chunks: list[tuple[np.ndarray, np.ndarray]],
        length: int,
        stride: int = 1,
        score_all: bool = False,
    ):
        self.length = length
        self.labels = np.concatenate([labels for _, labels in chunks])
        self.windows = []
        half = length // 2

        for signals, labels in chunks:
            last = len(labels) - length
            for start in range(0, last + 1, stride):
                mask = np.zeros(length, dtype=bool)
                if not score_all:
                    mask[:] = True
                else:
                    mask[half] = True
                    if start == 0:
                        mask[:half] = True
                    if start == last:
                        mask[half:] = True
                self.windows.append((signals, labels, start, mask))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int):
        signals, labels, start, mask = self.windows[i]
        end = start + self.length
        return (
            torch.from_numpy(signals[start:end].copy()),
            torch.from_numpy(labels[start:end].astype("int64").copy()),
            torch.from_numpy(mask.copy()),
        )


def _read_group(
    epochs_dir, subjects: set[str], filter_mode: str, normalize_mode: str, limit: int | None
) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    for path in sorted(Path(epochs_dir).glob("*.npz")):
        if path.stem.split("N")[0] not in subjects:
            continue
        signals, labels, _ = load_recording(path, filter_mode, normalize_mode)
        out.append((signals, labels))
        if limit and len(out) >= limit:
            break
    if not out:
        raise ValueError("읽어온 녹음이 없습니다")
    return out


def split_subjects(fold: dict[str, list[str]], val_ratio: float, seed: int) -> tuple[set, set, set]:
    """학습용 피험자 일부를 검증용으로 떼어낸다.

    학습 회차를 고를 때 평가 데이터를 보면 성능이 부풀려진다. 그래서 학습 쪽에서만
    떼어낸 사람들로 회차를 고르고, 평가 데이터는 마지막에 한 번만 쓴다.
    """
    train = sorted(fold["train"])
    n_val = max(1, round(len(train) * val_ratio))
    val = set(np.random.default_rng(seed).choice(train, size=n_val, replace=False))
    return set(train) - val, val, set(fold["test"])


def build_epoch_split(epochs_dir, fold, val_ratio, seed, filter_mode, normalize_mode, limit=None):
    """조각 하나를 표본으로 쓰는 학습용, 검증용, 평가용 데이터를 만든다."""
    groups = split_subjects(fold, val_ratio, seed)
    caps = (limit, max(1, limit // 4), max(1, limit // 4)) if limit else (None, None, None)
    out = []
    for subjects, cap in zip(groups, caps, strict=True):
        chunks = _read_group(epochs_dir, subjects, filter_mode, normalize_mode, cap)
        out.append(
            EpochDataset(
                np.concatenate([s for s, _ in chunks]), np.concatenate([lab for _, lab in chunks])
            )
        )
    return tuple(out)


def build_sequence_split(
    epochs_dir, fold, val_ratio, seed, filter_mode, normalize_mode, length, stride, limit=None
):
    """묶음을 표본으로 쓰는 학습용, 검증용, 평가용 데이터를 만든다.

    ``stride`` 는 학습용에만 쓴다. 검증용과 평가용은 한 칸씩 밀며 만들어 모든
    조각이 한 번씩 채점되게 한다.
    """
    groups = split_subjects(fold, val_ratio, seed)
    caps = (limit, max(1, limit // 4), max(1, limit // 4)) if limit else (None, None, None)
    strides = (stride, 1, 1)
    score_all = (False, True, True)
    out = []
    for subjects, cap, st, sa in zip(groups, caps, strides, score_all, strict=True):
        chunks = _read_group(epochs_dir, subjects, filter_mode, normalize_mode, cap)
        chunks = [c for c in chunks if len(c[1]) >= length]
        if not chunks:
            raise ValueError(f"묶음 길이 {length} 보다 긴 녹음이 없습니다")
        out.append(SequenceDataset(chunks, length, st, sa))
    return tuple(out)
