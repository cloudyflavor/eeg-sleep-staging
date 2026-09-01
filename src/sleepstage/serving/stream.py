"""신호가 30초마다 하나씩 들어오는 상황에서 수면 단계를 판정한다.

학습할 때는 밤 전체를 한꺼번에 놓고 계산했지만, 실제 기기는 조각을 하나씩 보낸다.
같은 결과를 내면서 조각 단위로 처리하도록 계산 방식을 바꾼 것이 이 모듈이다.

일부 특징은 뒤쪽 조각이 있어야 계산할 수 있다. 그런 특징을 쓰려면 판정을 그만큼
미뤄야 하므로, 지금 들어온 조각이 아니라 몇 개 전 조각의 답을 돌려준다.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from sleepstage.core import context, events, filters, normalize
from sleepstage.experiment.pipeline.features import epoch_features

#: 필터를 걸 때 앞뒤로 함께 넣는 조각 수. 신호가 갑자기 시작하거나 끝나는 데서
#: 생기는 왜곡을 피한다.
FILTER_PAD_EPOCHS = 2


def epoch_row(
    epoch: np.ndarray,
    sfreq: float,
    channels: list[str],
    settings: dict,
    spindles: list[events.SpindleCounter] | None,
) -> dict[str, float]:
    """조각 하나의 특징을 이름을 키로 하는 사전으로 낸다.

    학습과 같은 함수에 조각 하나짜리 배열을 넣는다. 여기서 따로 계산하면 학습과
    서빙이 갈라져 오류 없이 다른 값을 내게 된다.

    방추만 예외다. 문턱이 지나온 조각들로 정해지므로 조각마다 새로 세면 학습과
    다르다. 밤 전체를 기억하는 계수기가 대신 센다.
    """
    frame = epoch_features(
        epoch[None],
        sfreq,
        channels,
        settings.get("distance_features", True),
        welch_average=settings.get("welch_average", "median"),
        window_extremes=settings.get("window_extremes", False),
        spindle_events=False,
        channel_correlation=settings.get("channel_correlation", False),
    )
    row = {k: float(v) for k, v in frame.iloc[0].items()}
    if spindles is not None:
        for i, (name, counter) in enumerate(zip(channels, spindles, strict=True)):
            n, sec = counter.update(epoch[i])
            row[f"{name}__spindle_n"], row[f"{name}__spindle_sec"] = float(n), sec
    return row


@dataclass
class RunningStats:
    """지금까지 들어온 값들의 중앙값과 분포 폭을 기억한다.

    학습 때 쓴 방식과 같게 하려면 과거 값을 모두 보고 계산해야 한다. 하룻밤이
    천 개 남짓이라 그대로 쌓아둔다.
    """

    history: dict[str, list[float]] = field(default_factory=dict)

    def update(self, values: dict[str, float]) -> None:
        for name, value in values.items():
            self.history.setdefault(name, []).append(value)

    def normalize(self, values: dict[str, float], min_periods: int) -> dict[str, float]:
        out = {}
        for name, value in values.items():
            past = self.history[name]
            if len(past) < min_periods:
                out[f"{name}_norm"] = float("nan")
                continue
            arr = np.asarray(past)
            lo, hi = np.percentile(arr, [5, 95])
            scale = hi - lo
            out[f"{name}_norm"] = (
                float("nan") if scale <= 1e-12 else float((value - np.median(arr)) / scale)
            )
        return out


class StreamingStager:
    """조각을 하나씩 받아 준비되는 대로 판정을 돌려준다.

    ``push`` 가 ``None`` 을 돌려주면 아직 답할 수 없다는 뜻이다. 뒤쪽 조각이
    충분히 쌓이면 그때 지난 조각의 판정이 나온다.
    """

    def __init__(self, model, config: dict, feature_names: list[str]):
        self.model = model
        self.feature_names = feature_names
        self.channels = config["channels"]
        self.sfreq = float(config["sfreq"])
        self.stage_names = config["stage_names"]
        # 조각 길이를 여기 박아두면 표와 갈라진다. 배포 파일이 들고 있는 값을 쓴다.
        self.epoch_seconds = float(config.get("epoch_seconds", 30.0))
        self.preprocess = config["preprocess"]
        self.spindles = (
            [events.SpindleCounter(self.sfreq) for _ in self.channels]
            if self.preprocess.get("spindle_events")
            else None
        )

        self.uses_center = self.preprocess["context"] == "smooth"
        self.window = context.CENTERED_WINDOW if self.uses_center else 1
        self.delay = self.window // 2

        # 필터는 앞뒤 신호를 함께 봐야 정확하다. 조각 하나만 걸면 양끝이 왜곡되므로
        # 원신호를 몇 조각 들고 있다가 이어붙여 걸고 가운데만 꺼내 쓴다.
        self.filter_pad = 0 if self.preprocess["filter"] == "none" else FILTER_PAD_EPOCHS
        self.raw_buffer: deque[np.ndarray] = deque(maxlen=2 * self.filter_pad + 1)
        self.stats = RunningStats()
        # 판정 대상보다 앞뒤로 필요한 만큼을 모두 담을 수 있는 크기여야 한다.
        # 과거 평균은 대상 시점 기준이므로 대상 앞쪽 조각도 남겨둬야 한다.
        self.history_needed = context.TRAILING_WINDOW - 1
        self.buffer: deque[dict[str, float]] = deque(
            maxlen=self.window + self.history_needed if self.uses_center else 1
        )
        self.last_row: np.ndarray | None = None  # 방금 판정에 쓴 특징 한 줄
        self.index = 0  # 특징 계산이 끝난 조각 수
        self.received = 0  # 들어온 조각 수, 필터 여유분만큼 앞선다

    @property
    def lag(self) -> int:
        """판정이 나오기까지 밀리는 조각 수.

        창의 절반뿐 아니라 필터에 필요한 여유분까지 더해야 실제 지연이 된다.
        앞의 것만 세면 실제보다 짧게 보고하게 된다.
        """
        return self.delay + self.filter_pad

    def push(self, epoch: np.ndarray) -> dict[str, Any] | None:
        """조각 하나를 넣는다. 판정이 준비됐으면 결과를, 아니면 ``None`` 을 돌려준다."""
        if epoch.shape != (len(self.channels), int(self.sfreq * 30)):
            raise ValueError(f"신호 모양이 맞지 않습니다: {epoch.shape}")

        # 조각 번호는 들어온 순서로 센다. 필터 여유분 때문에 아직 처리하지 못한
        # 조각도 번호는 이미 받은 것이다.
        self.received += 1

        first_fill = self.filter_pad and len(self.raw_buffer) == self.raw_buffer.maxlen - 1
        filtered = self._filter_middle(epoch)
        if filtered is None:
            return None

        # 버퍼가 처음 찬 순간, 앞서 넘겼던 여유 조각들을 먼저 처리한다
        if first_fill:
            for early in self._catch_up_initial():
                self._absorb(early)
        self._absorb(filtered)

        if len(self.buffer) < self.window:
            return None
        # 지금 특징을 계산한 조각은 받은 것 중 filter_pad 만큼 이전 것이다
        current = self.received - 1 - self.filter_pad
        return self._decide(current - self.delay)

    def _filter_middle(self, epoch: np.ndarray) -> np.ndarray | None:
        """앞뒤 조각을 붙여 필터를 걸고 가운데 조각만 돌려준다.

        여유 조각이 모일 때까지는 ``None`` 을 돌려준다. 그만큼 판정이 늦어지지만,
        측정 초반 조각도 버리지 않고 여유분이 모인 뒤 한꺼번에 처리한다.
        """
        if self.filter_pad == 0:
            return epoch

        self.raw_buffer.append(epoch)
        if len(self.raw_buffer) < self.raw_buffer.maxlen:
            return None

        width = epoch.shape[-1]
        joined = np.concatenate(list(self.raw_buffer), axis=-1)
        filtered = filters.apply_filter(joined, self.sfreq, self.preprocess["filter"])
        start = self.filter_pad * width
        return filtered[:, start : start + width]

    def _catch_up_initial(self) -> list[np.ndarray]:
        """버퍼가 처음 찼을 때, 앞쪽 여유 조각들도 필터를 걸어 돌려준다.

        이 조각들은 앞에 이웃이 없어 왜곡이 남지만, 통계에서 빼면 기준값이
        학습 때와 달라진다. 학습에서도 같은 위치의 조각을 그대로 썼다.
        """
        width = self.raw_buffer[0].shape[-1]
        joined = np.concatenate(list(self.raw_buffer), axis=-1)
        filtered = filters.apply_filter(joined, self.sfreq, self.preprocess["filter"])
        return [filtered[:, i * width : (i + 1) * width] for i in range(self.filter_pad)]

    def _absorb(self, filtered: np.ndarray) -> None:
        """필터를 거친 조각에서 특징을 뽑아 통계와 버퍼에 넣는다."""
        raw = epoch_row(filtered, self.sfreq, self.channels, self.preprocess, self.spindles)
        self.stats.update(raw)

        combined = {**raw}
        if self.preprocess["normalize"] != "none":
            combined.update(self.stats.normalize(raw, normalize.MIN_PERIODS))

        self.buffer.append(combined)
        self.index += 1

    def flush(self) -> list[dict[str, Any]]:
        """남은 조각들을 마저 판정한다. 측정이 끝났을 때 호출한다.

        창이 다 차지 않아 미뤄뒀던 마지막 조각들을 처리한다. 뒤쪽 정보가 부족하므로
        평소보다 판단 근거가 적다.
        """
        results = []
        current = self.received - 1 - self.filter_pad
        for step in range(1, self.delay + 1):
            target = current - self.delay + step
            if 0 <= target <= current:
                results.append(self._decide(target, offset=step))
        return results

    def _decide(self, epoch_idx: int, offset: int = 0) -> dict[str, Any]:
        row = self._build_row(epoch_idx, offset)
        self.last_row = row
        proba = self._predict(row)
        best = int(np.argmax(proba))
        return {
            "epoch_idx": epoch_idx,
            "stage": self.stage_names[best],
            "confidence": float(proba[best]),
            "proba": {name: float(p) for name, p in zip(self.stage_names, proba, strict=True)},
        }

    def _build_row(self, epoch_idx: int, offset: int = 0) -> np.ndarray:
        """판정할 조각의 특징과 주변 흐름을 합쳐 모델 입력 한 줄을 만든다.

        버퍼는 판정 대상 앞뒤를 함께 담고 있다. ``offset`` 은 측정이 끝나 뒤쪽
        조각이 더 들어오지 않을 때, 대상 위치가 버퍼 안에서 얼마나 밀렸는지를 뜻한다.
        """
        if not self.uses_center:
            values = dict(self.buffer[-1])
            values["time_hour"] = epoch_idx * self.epoch_seconds / 3600.0
            return np.array(
                [values.get(name, np.nan) for name in self.feature_names], dtype=np.float32
            )

        # 버퍼 안에서 판정 대상의 위치. 앞쪽에는 과거 평균용 조각이 더 있다.
        at = min(self.history_needed + self.delay + offset, len(self.buffer) - 1)
        values = dict(self.buffer[at])

        frame = pd.DataFrame(list(self.buffer))
        centered, _ = context.smoothed(frame)
        values.update(centered.iloc[at].to_dict())

        values["time_hour"] = epoch_idx * self.epoch_seconds / 3600.0
        return np.array([values.get(name, np.nan) for name in self.feature_names], dtype=np.float32)

    def _predict(self, row: np.ndarray) -> np.ndarray:
        proba = self.model.predict_proba(row.reshape(1, -1))[0]
        classes = getattr(self.model, "classes_", None)
        if classes is None or len(classes) == len(self.stage_names):
            return np.asarray(proba, dtype=np.float32)
        full = np.zeros(len(self.stage_names), dtype=np.float32)
        full[np.asarray(classes, dtype=int)] = proba
        return full
