"""시간 문맥 — 이웃 에포크 정보를 열로. lookahead 도 함께 반환. 근거: docs/04-stage2.md D12."""

import numpy as np
import pandas as pd

CENTERED_WINDOW = 15  # 7.5분
TRAILING_WINDOW = 4  # 2분
SHIFTS = (-2, -1, 1, 2)


def smoothed(feats: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    center = (
        feats.rolling(CENTERED_WINDOW, center=True, min_periods=1, win_type="triang")
        .mean()
        .add_suffix("_c7min")
    )
    trail = feats.rolling(TRAILING_WINDOW, min_periods=1).mean().add_suffix("_p2min")

    lookahead = dict.fromkeys(center.columns, CENTERED_WINDOW // 2)
    lookahead.update(dict.fromkeys(trail.columns, 0))
    return pd.concat([center, trail], axis=1), lookahead


def shifted(
    feats: pd.DataFrame,
    epoch_idx: np.ndarray,
    shifts: tuple[int, ...] = SHIFTS,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """열을 시간축으로 민 사본. shift(+1)=과거이므로 이름·lookahead 는 부호를 뒤집는다.

    시간축 구멍은 결측으로 둔다. 녹음 하나씩 넘길 것 — 아니면 남의 밤을 참조한다.
    """
    idx = pd.Series(epoch_idx, index=feats.index)
    frames, lookahead = [], {}
    for s in shifts:
        moved = feats.shift(s)
        moved[idx.diff(s) != s] = np.nan
        offset = -s
        moved = moved.add_suffix(f"_ep{offset:+d}")
        frames.append(moved)
        lookahead.update(dict.fromkeys(moved.columns, max(offset, 0)))
    return pd.concat(frames, axis=1), lookahead


#: 방식 → 빌더 (feats, epoch_idx) → (열, lookahead). None = 문맥 없음.
CONTEXTS = {"none": None, "smooth": lambda feats, _idx: smoothed(feats), "shift": shifted}


def elapsed_hours(offset_epochs: np.ndarray, epoch_seconds: float) -> pd.Series:
    """창 시작 기준 경과 시간(시). 녹음 시작 기준이면 이 데이터의 녹음 시각대가 샌다."""
    return pd.Series(offset_epochs * epoch_seconds / 3600.0, name="time_hour", dtype="float32")
