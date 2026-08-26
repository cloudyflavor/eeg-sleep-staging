"""밴드패스 필터 — 셋 중 하나를 설정으로 고른다.

뇌파에는 판정에 쓸 대역(0.4–30 Hz) 밖의 것들이 섞여 있다.

=============== ================================== ====
주파수           정체                                쓰나
=============== ================================== ====
0.1 Hz 이하      땀·호흡으로 전극이 흔들리는 표류      ❌
0.4–30 Hz       델타·세타·알파·방추·베타             ✅
50/60 Hz        전원 잡음                           ❌
70 Hz 이상       근육 떨림                           ❌
=============== ================================== ====

**어느 방식이 맞는지 정해져 있지 않다.** 레퍼런스 셋이 셋 다 다르다.

- DeepSleepNet — **안 건다** (원신호를 그대로 CNN 에)
- YASA — **영위상** FIR 0.4–30 Hz
- sleep-linear — **인과** Butterworth 5차 0.4–30 Hz, 그런데 DeepSleepNet 을 이겼다

그래서 우리가 잰다. 셋 다 10분 지연 예산 안에 들어오므로 성능만 보고 고르면 된다.

.. important::
   **절단(crop) 전 전체 신호에 걸어야 한다.** 인과 필터는 과거를 참조하는데, 절단 후에
   걸면 첫 몇 초가 과거 없이 시작해 경계 왜곡이 생긴다. 실제 기기에는 그런 경계가 없다.
"""

import numpy as np
from scipy import signal as sp_signal

#: 통과 대역. YASA·sleep-linear 둘 다 동일.
BAND = (0.4, 30.0)

#: Butterworth 차수. sleep-linear 과 동일.
ORDER = 5


def apply_filter(signal: np.ndarray, sfreq: float, mode: str) -> np.ndarray:
    """``(n_channels, n_samples)`` 신호에 밴드패스를 건다.

    ``mode``:

    ``"none"``
        걸지 않는다. EDFX 는 하드웨어에서 이미 0.5–100 Hz 로 필터링됐고,
        대역파워를 Welch PSD 로 내면 시간영역 밴드패스가 **원리적으로 불필요**하다.
        "필터가 값어치가 없다" 는 결과가 나오면 웨어러블에서 연산을 통째로 뺄 수 있다.

    ``"causal"``
        과거만 본다 (``sosfilt``). **실시간 가능.** 대신 주파수마다 밀리는 양이 달라
        파형이 뒤로 밀린다(위상 왜곡). 30초 안의 통계량은 위상에 둔감하므로 문제가
        안 될 가능성이 크다 — sleep-linear 이 그 방증이다.

    ``"zerophase"``
        앞→뒤, 뒤→앞 두 번 걸어 밀림을 상쇄한다 (``sosfiltfilt``). 파형이 보존되지만
        **미래가 필요하다.** 블록 단위로 걸면 수 초면 되므로 10분 예산 안이다.
    """
    if mode == "none":
        return signal

    sos = sp_signal.butter(ORDER, BAND, btype="bandpass", fs=sfreq, output="sos")
    if mode == "causal":
        out = sp_signal.sosfilt(sos, signal, axis=-1)
    elif mode == "zerophase":
        out = sp_signal.sosfiltfilt(sos, signal, axis=-1)
    else:
        raise ValueError(f"모르는 필터 방식입니다: {mode!r} (none / causal / zerophase)")
    return out.astype(np.float32)


#: 각 방식이 요구하는 미래 시간(초). 지연 예산 회계에 쓴다.
LOOKAHEAD_SEC = {"none": 0.0, "causal": 0.0, "zerophase": 5.0}
