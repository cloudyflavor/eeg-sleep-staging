"""에포크 격자 위의 계산 — 라벨 코드, 신호 자르기, 절단 경계, 품질 지표.

여기서 에포크를 버리지 않는다. 표시와 경계를 계산해 줄 뿐이다.
"""

import numpy as np

#: 지우기로 한 에포크를 표시하는 값. 실제 클래스는 0 이상이다.
DROP = -1


def to_codes(
    per_epoch: np.ndarray,
    label_map: dict[str, int],
    drop: list[str],
) -> tuple[np.ndarray, dict[str, int]]:
    """주석 문자열을 클래스 코드로. 지울 것은 :data:`DROP`, 사유는 따로 센다.

    모르는 주석은 예외를 던진다. 조용히 흘려보내면 나중에 못 찾는다.
    """
    codes = np.full(len(per_epoch), DROP, dtype=np.int8)
    dropped = {}
    for desc in set(per_epoch.tolist()):
        mask = per_epoch == desc
        if desc in label_map:
            codes[mask] = label_map[desc]
        elif desc in drop:
            dropped[desc] = int(mask.sum())
        else:
            raise KeyError(f"모르는 주석입니다: {desc!r} ({int(mask.sum())}개)")
    return codes, dropped


def epoch_signal(signal: np.ndarray, samples_per_epoch: int) -> np.ndarray:
    """``(n_ch, n_samples)`` → ``(n_epochs, n_ch, samples_per_epoch)``. 꼬리는 버린다."""
    n_ch, n_samples = signal.shape
    n_ep = n_samples // samples_per_epoch
    return (
        signal[:, : n_ep * samples_per_epoch]
        .reshape(n_ch, n_ep, samples_per_epoch)
        .transpose(1, 0, 2)
        .copy()
    )


def crop_bounds(labels: np.ndarray, wake_code: int, edge_epochs: int) -> tuple[int, int]:
    """수면 구간 앞뒤로 ``edge_epochs`` 만 남기는 경계 ``[lo, hi)``.

    자르지 않고 경계만 돌려준다. 절단은 정답 라벨을 봐야 정할 수 있어 배포 시점에는
    재현할 수 없고, "절단 vs 미절단" 을 재려면 원본이 남아 있어야 한다.
    """
    non_wake = np.flatnonzero(labels != wake_code)
    if non_wake.size == 0:
        return 0, len(labels)  # 전부 각성 — 자를 근거가 없다
    lo = max(0, int(non_wake[0]) - edge_epochs)
    hi = min(len(labels), int(non_wake[-1]) + edge_epochs + 1)
    return lo, hi


def quality_metrics(epochs: np.ndarray) -> dict[str, np.ndarray]:
    """에포크·채널별 품질 지표. 전부 ``(n_epochs, n_ch)`` float32.

    제거하지 않고 지표만 낸다. 어려운 에포크를 버리면 평가 대상이 달라져 실험 간
    비교가 깨진다.
    """
    return {
        "ptp": (epochs.max(-1) - epochs.min(-1)).astype("float32"),
        "std": epochs.std(-1).astype("float32"),
        # 연속한 두 표본이 완전히 같은 비율. 전극이 빠지면 1.0 에 가까워진다.
        "flat_ratio": (np.diff(epochs, axis=-1) == 0).mean(-1).astype("float32"),
    }
