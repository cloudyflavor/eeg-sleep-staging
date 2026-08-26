"""시간 문맥 — 이웃 에포크 정보를 열로 붙인다.

각 함수는 자기 열이 필요로 하는 미래 에포크 수도 함께 돌려준다(지연 예산 장부).
평활(YASA) vs 시프트(sleep-linear) 비교 근거는 `docs/04-stage2.md` D12.
"""

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
    """각 열을 시간축으로 민 사본을 붙인다.

    Notes
    -----
    **``shift(+1)`` 이 과거를 끌어온다.** 그래서 이름과 lookahead 는 부호를 뒤집어
    ``_ep-1`` 이 과거, ``_ep+1`` 이 미래다. 반대로 적으면 미래를 보는 열이 lookahead 0
    으로 기록돼 "실시간 가능" 이라고 잘못 보고하게 된다.

    ``epoch_idx`` 는 판독불가 제거로 생긴 구멍을 막는다 — 한 칸 위가 항상 30초 전은 아니다.

    호출하는 쪽이 **녹음 하나씩** 넘겨야 한 사람의 첫 에포크가 다른 사람의 마지막
    에포크를 참조하지 않는다.
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


def elapsed_hours(epoch_idx: np.ndarray, epoch_seconds: float) -> pd.Series:
    """녹음 시작부터의 경과 시간(시). 깊은 수면은 전반부에, REM 은 후반부에 몰린다.

    YASA 의 ``time_norm`` 은 밤 전체 길이를 알아야 해서 실시간에 계산할 수 없다.
    """
    return pd.Series(epoch_idx * epoch_seconds / 3600.0, name="time_hour", dtype="float32")
