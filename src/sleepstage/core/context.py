"""앞뒤 에포크의 정보를 열로 추가한다.

30초 조각 하나만 보면 얕은 수면과 REM 이 비슷해 구분하기 어렵다. 주변 몇 분의
흐름을 함께 보면 훨씬 잘 갈린다.

각 함수는 만든 열이 미래를 몇 에포크나 봐야 하는지도 함께 돌려준다. 실시간으로
쓸 수 있는 열을 고를 때 이 정보가 필요하다.
"""

import numpy as np
import pandas as pd

#: 앞뒤를 함께 보는 창의 크기, 15 에포크는 7.5분이다
CENTERED_WINDOW = 15

#: 과거만 보는 창의 크기, 4 에포크는 2분이다
TRAILING_WINDOW = 4

#: 몇 칸씩 밀어 붙일지, 음수는 미래 방향이다
SHIFTS = (-2, -1, 1, 2)


def smoothed(feats: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """주변 값의 평균을 열로 추가한다.

    앞뒤 7.5분 평균과 최근 2분 평균 두 가지를 만든다. 평균을 내면 순간적인
    흔들림이 줄고 단계 수준의 흐름만 남는다.
    """
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
    """이웃 에포크의 값을 그대로 열로 붙인다.

    평균과 달리 값의 순서가 보존되므로 늘어나는 중인지 줄어드는 중인지 알 수 있다.

    ``shift(+1)`` 은 한 칸 위의 값을 끌어오므로 과거를 뜻한다. 열 이름과 미래
    참조량은 이 부호를 뒤집어 붙인다. ``_ep-1`` 이 과거, ``_ep+1`` 이 미래다.

    ``epoch_idx`` 로 실제 시간 간격을 확인한다. 판독 불가로 빠진 에포크가 있으면
    한 칸 위가 30초 전이 아닐 수 있고, 그런 자리는 결측으로 둔다.

    호출하는 쪽이 녹음 하나씩 넘겨야 한 사람의 첫 에포크가 다른 사람의 마지막
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


#: 방식 이름 -> 열을 만드는 함수. None 은 문맥을 쓰지 않는다는 뜻이다.
CONTEXTS = {"none": None, "smooth": lambda feats, _idx: smoothed(feats), "shift": shifted}


def elapsed_hours(offset_epochs: np.ndarray, epoch_seconds: float) -> pd.Series:
    """측정을 시작한 뒤 몇 시간이 지났는지.

    깊은 수면은 밤 전반부에, REM 은 후반부에 몰리므로 시각 자체가 단서가 된다.
    녹음 시작이 아니라 측정 창 시작을 기준으로 삼아 실제 기기와 조건을 맞춘다.
    """
    return pd.Series(offset_epochs * epoch_seconds / 3600.0, name="time_hour", dtype="float32")
