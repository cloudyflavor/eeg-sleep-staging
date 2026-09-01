"""수면다원검사 파일에서 뇌파 신호와 판독 결과를 읽는다.

신호 파일과 판독 파일이 짝을 이룬다. 파일명 앞부분이 같고 뒤쪽 한 글자만 다르다::

    SC4001E0-PSG.edf          신호
    SC4001EC-Hypnogram.edf    판독 결과
    ^^^^^^^ 이 부분이 공통
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_PSG_RE = re.compile(r"^(?P<subject>S[CT]\d{3})(?P<night>\d)(?P<tail>[A-Z]\w?)-PSG$")


@dataclass(frozen=True)
class Recording:
    """PSG 신호 파일과 그 라벨 파일 한 쌍."""

    subject: str
    night: int
    psg: Path
    hyp: Path

    @property
    def key(self) -> str:
        """산출물 파일명이자 피험자-밤을 가리키는 키. 예: ``SC400N1``."""
        return f"{self.subject}N{self.night}"


def find_recordings(root: str | Path) -> list[Recording]:
    """디렉터리에서 신호와 판독 파일의 짝을 모두 찾는다.

    파일명으로 직접 맞춰본다. 목록을 정렬해 순서대로 짝지으면 파일이 하나
    빠졌을 때 그 뒤가 전부 밀리는데 아무 오류도 나지 않는다.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"데이터 디렉터리가 없습니다: {root}")

    out = []
    for psg in sorted(root.glob("*-PSG.edf")):
        m = _PSG_RE.match(psg.stem)
        if m is None:
            raise ValueError(f"파일명 형식을 모르겠습니다: {psg.name}")

        hyps = sorted(root.glob(f"{psg.stem[:7]}*-Hypnogram.edf"))
        if len(hyps) != 1:
            raise FileNotFoundError(
                f"{psg.name} 의 라벨 파일이 {len(hyps)}개입니다 (1개여야 함): "
                f"{[h.name for h in hyps]}"
            )
        out.append(Recording(m["subject"], int(m["night"]), psg, hyps[0]))

    if not out:
        raise FileNotFoundError(f"{root} 에 *-PSG.edf 가 없습니다")
    return out


def read_recording(
    rec: Recording,
    channels: list[str],
    sfreq: float,
    epoch_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """신호와 판독 결과를 함께 읽는다.

    신호는 (채널, 샘플) 마이크로볼트 배열, 판독 결과는 에포크마다 문자열 하나다.

    두 파일의 측정 시작 시각이 같은지 확인하고 다르면 오류를 낸다. 어긋난 채로
    진행하면 라벨이 통째로 밀린 상태에서 학습이 문제없이 끝나버린다.
    """
    import mne

    raw = mne.io.read_raw_edf(rec.psg, preload=False, verbose="ERROR")

    missing = [c for c in channels if c not in raw.ch_names]
    if missing:
        raise KeyError(f"{rec.psg.name} 에 채널이 없습니다: {missing}")
    if abs(raw.info["sfreq"] - sfreq) > 1e-6:
        # 조용히 리샘플링되는 것보다 시끄럽게 죽는 편이 낫다
        raise ValueError(f"{rec.psg.name} 샘플링레이트 {raw.info['sfreq']} Hz, 기대값 {sfreq} Hz")

    ann = mne.read_annotations(rec.hyp)
    if raw.info["meas_date"] is not None and ann.orig_time is not None:
        offset = (ann.orig_time - raw.info["meas_date"]).total_seconds()
        if abs(offset) > 1e-6:
            raise ValueError(f"{rec.key}: 신호와 라벨 시작시각이 {offset}초 어긋납니다")

    raw.pick(channels)
    signal = (raw.get_data() * 1e6).astype("float32")  # V → µV
    return signal, expand_annotations(ann.description, ann.duration, epoch_seconds)


def expand_annotations(descriptions, durations, epoch_seconds: float) -> np.ndarray:
    """구간 단위 판독 결과를 에포크 단위로 펼친다.

    판독 파일은 "몇 시부터 30분간 각성" 형태로 되어 있다. 이를 30초마다 하나씩
    값이 있는 형태로 바꾼다. 구간 길이가 30초의 배수가 아니면 격자가 어긋난
    것이므로 반올림하지 않고 오류를 낸다.
    """
    per_epoch = []
    for desc, duration in zip(descriptions, durations, strict=True):
        n = duration / epoch_seconds
        if abs(n - round(n)) > 1e-6:
            raise ValueError(f"주석 길이 {duration}초가 에포크 {epoch_seconds}초의 배수가 아닙니다")
        per_epoch += [str(desc)] * round(n)
    return np.array(per_epoch, dtype=object)
