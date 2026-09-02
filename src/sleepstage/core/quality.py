"""녹음 하나의 신호 품질을 재고 배제할지 정한다.

조각마다 임계값을 걸면 EDFX 에서 아무것도 안 걸린다. 한 채널만 죽은 녹음은
그 채널의 진폭이 낮을 뿐 조각 하나하나는 정상 범위 안에 있기 때문이다.
그래서 단위가 조각이 아니라 녹음이고 채널이다.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def flat_fraction(ptp: np.ndarray, flat_uv: float) -> np.ndarray:
    """채널마다 진폭이 flat_uv 미만인 조각의 비율. ptp 는 (n_ep, n_ch)."""
    return (np.asarray(ptp) < float(flat_uv)).mean(axis=0).astype(float)


def _flat(ptp: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    frac = flat_fraction(ptp, params["flat_uv"])
    return frac > float(params["max_flat_frac"]), {"flat_fraction": [round(v, 4) for v in frac]}


#: 검사 이름 -> 함수. 새 검사는 여기 한 줄만 추가한다. 모르는 이름은 예외.
CHECKS = {"flat": _flat}


def assess(ptp: np.ndarray, channels: list[str], params: dict[str, Any]) -> dict[str, Any]:
    """녹음 하나를 검사한다. EEG 채널이 하나라도 걸리면 배제 대상이다.

    문헌은 모든 채널이 동시에 나빠야 배제하는데 그 규칙은 EOG 가 있을 때의 것이다.
    EEG 두 개만 쓰는 여기서는 한쪽만 죽어도 그 녹음의 절반이 없는 것과 같다.
    """
    names = list(params.get("checks") or CHECKS)
    bad: dict[str, list[str]] = {}
    detail: dict[str, Any] = {}
    for name in names:
        if name not in CHECKS:
            raise KeyError(f"모르는 품질 검사입니다: {name!r} ({'/'.join(CHECKS)})")
        hit, info = CHECKS[name](ptp, params)
        detail.update(info)
        for c in np.flatnonzero(hit):
            bad.setdefault(channels[int(c)], []).append(name)
    return {"exclude": bool(bad), "bad_channels": bad, **detail}
