"""1단계 산출물을 원본 EDF 와 대조한다.

**파이프라인 코드를 쓰지 않는다.** mne 로 직접 읽고 주석을 손으로 펼친다.
같은 코드로 두 번 계산하면 같은 버그가 두 번 나올 뿐이라 검증이 되지 않는다.
"""

import numpy as np

from sleepstage.io.edf import Recording, find_recordings
from sleepstage.preprocess.prepare import _load

#: 라벨 지도를 여기 다시 적는 것도 의도한 것이다. 설정에서 읽어오면
#: 설정이 틀렸을 때 검증도 같이 틀린다.
_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 1,
    "Sleep stage 3": 2,
    "Sleep stage 4": 2,
    "Sleep stage R": 3,
}


def check_recording(rec: Recording, npz_dir, n_signal_samples: int = 0, rng=None) -> list[str]:
    """녹음 하나를 대조하고 문제 목록을 돌려준다. 빈 목록이면 통과."""
    import mne

    mne.set_log_level("ERROR")
    path = npz_dir / f"{rec.key}.npz"
    if not path.exists():
        return [f"{rec.key}: npz 가 없습니다"]

    z = np.load(path, allow_pickle=False)
    labels, epoch_idx = z["labels"], z["epoch_idx"]
    problems = []

    ann = mne.read_annotations(rec.hyp)
    per_epoch = []
    for desc, duration in zip(ann.description, ann.duration, strict=True):
        per_epoch += [str(desc)] * int(round(duration / 30))

    try:
        expected = np.array([_MAP[per_epoch[i]] for i in epoch_idx], dtype=np.int8)
    except (KeyError, IndexError) as e:
        return [f"{rec.key}: 라벨 재구성 실패 ({e})"]
    if not np.array_equal(labels, expected):
        problems.append(f"{rec.key}: 라벨 {int((labels != expected).sum())}개 불일치")

    if n_signal_samples:
        raw = mne.io.read_raw_edf(rec.psg, preload=False, verbose="ERROR")
        raw.pick([str(c) for c in z["ch_names"]])
        signal = raw.get_data() * 1e6
        width = z["data"].shape[-1]
        rng = rng or np.random.default_rng(0)
        for k in rng.choice(len(epoch_idx), min(n_signal_samples, len(epoch_idx)), replace=False):
            start = int(epoch_idx[k]) * width
            reference = signal[:, start : start + width].astype("float32")
            if not np.allclose(z["data"][k], reference, atol=1e-3):
                problems.append(f"{rec.key}: 에포크 {k} 신호 불일치")

    return problems


def verify_all(cfg, root=None, npz_dir=None, signal_every: int = 15, samples: int = 3) -> dict:
    """전체 녹음의 라벨을 대조하고, 일부는 신호까지 대조한다.

    신호 대조는 EDF 를 다시 읽어야 해서 비싸다. ``signal_every`` 개마다 한 번만 한다.
    """
    root, npz_dir = _load(cfg, root, npz_dir)
    recordings = find_recordings(root)
    rng = np.random.default_rng(0)

    problems, n_signal_checked = [], 0
    for i, rec in enumerate(recordings):
        do_signal = signal_every and i % signal_every == 0
        n_signal_checked += bool(do_signal)
        found = check_recording(rec, npz_dir, samples if do_signal else 0, rng)
        problems.extend(found)
        print(f"  {rec.key:>9}  {'문제 ' + str(len(found)) if found else 'OK'}", flush=True)

    return {
        "n_recordings": len(recordings),
        "n_signal_checked": n_signal_checked,
        "problems": problems,
    }
