"""신호를 30초 단위로 자르고 각 조각의 라벨과 품질을 정리한다.

여기서는 데이터를 버리지 않는다. 어디를 쓸지, 무엇을 뺄지는 표시만 해두고
실제 판단은 나중 단계에서 한다.
"""

import numpy as np

#: 지우기로 한 에포크를 표시하는 값. 실제 클래스는 0 이상이다.
DROP = -1

#: 4-class 이름과 순서. 여기 하나만 두고 나머지는 들여온다.
#: 두 군데 적어두면 한쪽만 고쳤을 때 라벨이 조용히 어긋난다.
STAGE_NAMES = ("W", "LS", "DS", "R")


def to_codes(
    per_epoch: np.ndarray,
    label_map: dict[str, int],
    drop: list[str],
) -> tuple[np.ndarray, dict[str, int]]:
    """판독 문자열을 숫자 라벨로 바꾼다.

    각성은 0, 얕은 수면은 1, 깊은 수면은 2, REM 은 3 이다. 판독 불가 구간은
    :data:`DROP` 으로 표시하고 사유별로 개수를 센다. 목록에 없는 문자열이
    나오면 오류를 낸다. 조용히 넘기면 나중에 원인을 찾을 수 없다.
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
    """연속 신호를 30초 조각으로 자른다. 마지막에 남는 자투리는 버린다."""
    n_ch, n_samples = signal.shape
    n_ep = n_samples // samples_per_epoch
    return (
        signal[:, : n_ep * samples_per_epoch]
        .reshape(n_ch, n_ep, samples_per_epoch)
        .transpose(1, 0, 2)
        .copy()
    )


def crop_bounds(labels: np.ndarray, wake_code: int, edge_epochs: int) -> tuple[int, int]:
    """실제로 잠든 구간의 앞뒤 여유를 포함한 범위를 찾는다.

    이 데이터는 하루 종일 녹음한 것이라 절반 이상이 낮에 깨어 있는 구간이다.
    잠든 구간 앞뒤로 일정 시간만 남기면 실제 기기가 재는 범위와 비슷해진다.

    범위만 돌려주고 자르지는 않는다.
    """
    non_wake = np.flatnonzero(labels != wake_code)
    if non_wake.size == 0:
        return 0, len(labels)  # 전부 각성 — 자를 근거가 없다
    lo = max(0, int(non_wake[0]) - edge_epochs)
    hi = min(len(labels), int(non_wake[-1]) + edge_epochs + 1)
    return lo, hi


def quality_metrics(epochs: np.ndarray) -> dict[str, np.ndarray]:
    """신호 품질을 나타내는 값들을 계산한다.

    진폭 범위(ptp), 표준편차(std), 값이 변하지 않는 비율(flat_ratio)을 낸다.
    전극이 빠지면 flat_ratio 가 1 에 가까워진다.

    이 값으로 에포크를 걸러내지는 않는다. 실험마다 다른 에포크가 빠지면
    성능을 비교할 수 없기 때문이다.
    """
    return {
        "ptp": (epochs.max(-1) - epochs.min(-1)).astype("float32"),
        "std": epochs.std(-1).astype("float32"),
        # 연속한 두 표본이 완전히 같은 비율. 전극이 빠지면 1.0 에 가까워진다.
        "flat_ratio": (np.diff(epochs, axis=-1) == 0).mean(-1).astype("float32"),
    }
