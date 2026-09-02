"""과거만 보고 단계 전이를 반영해 판정을 다듬는다.

모델은 조각마다 따로 답을 낸다. 그래서 각성 한가운데에 깊은 잠 한 칸이 튀어나와도
그대로 내보낸다. 단계가 바뀌는 확률표로 억지스러운 길에 벌점을 준다.

뒤쪽 조각을 보지 않으므로 판정이 밀리지 않는다. 실험 쪽의 일괄 복호화와 같은 답을
내야 하는데, 그쪽은 녹음 여러 개를 겹쳐 놓고 도는 빠른 판이라 코드를 합치지 않았다.
대신 두 판이 같은 답을 내는지 시험으로 고정한다.
"""

from __future__ import annotations

import numpy as np

#: 확률이 0 이면 로그가 발산한다. 아래로 자른다.
FLOOR = 1e-12


class PastOnlyDecoder:
    """조각을 하나씩 받아 지금까지의 흐름만으로 단계를 고른다."""

    def __init__(self, log_trans, weight: float = 1.0):
        table = np.asarray(log_trans, dtype="float64")
        if table.ndim != 2 or table.shape[0] != table.shape[1]:
            raise ValueError(f"전이 표가 정사각이 아닙니다: {table.shape}")
        self.log_trans = table * float(weight)
        self.weight = float(weight)
        self.alpha: np.ndarray | None = None

    @property
    def n_stages(self) -> int:
        return len(self.log_trans)

    def reset(self) -> None:
        self.alpha = None

    def step(self, proba) -> int:
        """조각 하나의 확률을 받아 단계 번호를 돌려준다."""
        values = np.asarray(proba, dtype="float64")
        if values.shape != (self.n_stages,):
            raise ValueError(f"확률 개수가 표와 다릅니다: {values.shape} 대 {self.n_stages}")
        emission = np.log(np.clip(values, FLOOR, None))
        if self.alpha is None:
            self.alpha = emission
        else:
            prev = self.alpha[:, None] + self.log_trans
            peak = prev.max(axis=0)
            self.alpha = emission + peak + np.log(np.exp(prev - peak).sum(axis=0))
        return int(self.alpha.argmax())

    def state(self) -> list[float] | None:
        """세션을 이어받을 때 넘길 값. 이것을 빼먹으면 재개한 뒤 흐름이 끊긴다."""
        return None if self.alpha is None else self.alpha.tolist()

    def load(self, values) -> None:
        self.alpha = None if values is None else np.asarray(values, dtype="float64")
