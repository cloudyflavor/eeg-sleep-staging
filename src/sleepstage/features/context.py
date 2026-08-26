"""시간 문맥 — 이웃 에포크의 정보를 열로 추가한다.

수면 단계는 앞뒤 맥락 없이 정할 수 없다. 깊은 수면 직후의 얕은 수면과 REM 직전의 얕은 수면은
파형이 비슷하지만 **어디에서 왔는지**가 다르다.

YASA 는 평활을, sleep-linear 은 시프트를 쓴다. **둘 다 논문으로 나왔고 아무도 직접
비교하지 않았다.** 그래서 셋을 다 만들어 두고 설정으로 고른다.

각 방식이 요구하는 **미래 시간(lookahead)** 을 함께 돌려준다. 웨어러블에서 지연 예산 안에
드는 특징만 쓰려면 이 장부가 필요하다.
"""

import numpy as np
import pandas as pd

#: 중심 평활 창. 15 에포크 = 7.5분 (YASA 와 동일). 미래 7 에포크가 필요하다.
CENTERED_WINDOW = 15

#: 과거 평활 창. 4 에포크 = 2분 (YASA 와 동일). 미래가 필요 없다.
TRAILING_WINDOW = 4

#: 시프트 폭. 음수가 과거, 양수가 미래.
SHIFTS = (-2, -1, 1, 2)


def smoothed(feats: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """평활 방식 (YASA). 중심 7.5분 + 과거 2분.

    중심 평활은 삼각가중이다 — 가까운 에포크에 큰 가중을 준다.

    - ✅ 열이 특징 수 × 2 로 끝난다. 잡음이 눌린다
    - ❌ **평균이라 순서가 사라진다.** "올라가는 중" 과 "내려가는 중" 을 구분하지 못한다
    """
    center = feats.rolling(CENTERED_WINDOW, center=True, min_periods=1, win_type="triang").mean()
    trail = feats.rolling(TRAILING_WINDOW, min_periods=1).mean()
    out = pd.concat(
        [center.add_suffix("_c7min"), trail.add_suffix("_p2min")],
        axis=1,
    )
    lookahead = {c: CENTERED_WINDOW // 2 for c in center.add_suffix("_c7min").columns}
    lookahead.update({c: 0 for c in trail.add_suffix("_p2min").columns})
    return out, lookahead


def shifted(
    feats: pd.DataFrame,
    epoch_idx: np.ndarray,
    shifts: tuple[int, ...] = SHIFTS,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """시프트 방식 (sleep-linear). 특징 열을 시간축으로 밀어 붙인다.

    - ✅ **순서·방향이 보존된다.** "직전보다 델타가 올랐다" 를 트리가 직접 배운다
    - ❌ 열 폭증. 특징 50개 × 시프트 4개 = 200열. 유효 표본이 78명인데 위험하다

    ``epoch_idx`` 는 원본 에포크 번호다. **판독불가 제거로 시간축에 구멍이 1,104개 있어서**,
    단순히 한 칸 미는 것이 "30초 전" 이 아니라 "3분 전" 일 수 있다.
    실제 간격이 정확히 ``shift`` 만큼이 아니면 결측으로 둔다 — 없는 이웃을 있는 척하지 않는다.

    호출하는 쪽이 **녹음 하나씩** 넘겨야 한다. 그래야 A 라는 사람의 마지막 에포크가
    B 의 첫 에포크를 참조하는 일이 없다.
    """
    frames, lookahead = [], {}
    for s in shifts:
        moved = feats.shift(s)
        # 실제 시간 간격이 s 에포크가 맞는지 확인. 아니면 그 행은 결측.
        gap = pd.Series(epoch_idx, index=feats.index).diff(s)
        valid = (gap == s).to_numpy()
        moved = moved.where(valid[:, None])

        tag = f"_lag{s:+d}"
        moved = moved.add_suffix(tag)
        frames.append(moved)
        lookahead.update({c: max(s, 0) for c in moved.columns})

    return pd.concat(frames, axis=1), lookahead


def elapsed_hours(epoch_idx: np.ndarray, epoch_seconds: float) -> pd.Series:
    """녹음 시작부터의 경과 시간(시). 그 자체가 강한 사전정보다.

    **깊은 수면은 전반부에, REM 은 후반부에 몰린다.** 생리학적 사실이라 트리가 바로 쓴다.

    YASA 는 ``time_norm = times / times[-1]`` 도 쓰는데, **밤 길이를 알아야 하므로
    실시간에 계산할 수 없다.** 우리는 쓰지 않는다.
    """
    return pd.Series(epoch_idx * epoch_seconds / 3600.0, name="time_hour", dtype="float32")
