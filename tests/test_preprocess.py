"""1단계 전처리 단위 테스트 — 실제 EDF 없이 도는 것만."""

from __future__ import annotations

import numpy as np
import pytest

from sleepstage.io.edf import expand_annotations, find_recordings
from sleepstage.preprocess.epochs import DROP, crop_bounds, epoch_signal, quality_metrics, to_codes

LABEL_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 1,
    "Sleep stage 3": 2,
    "Sleep stage 4": 2,
    "Sleep stage R": 3,
}
DROP_LIST = ["Sleep stage ?", "Movement time"]


# ── 주석 펼치기 ────────────────────────────────────────────────────────────


def test_expand_splits_duration_into_epochs():
    out = expand_annotations(["Sleep stage W", "Sleep stage 2"], [90.0, 60.0], 30.0)
    assert list(out) == ["Sleep stage W"] * 3 + ["Sleep stage 2"] * 2


def test_expand_rejects_non_multiple_duration():
    """격자가 어긋나는 건 반올림하지 말고 실패해야 한다."""
    with pytest.raises(ValueError, match="배수가 아닙니다"):
        expand_annotations(["Sleep stage W"], [45.0], 30.0)


# ── 라벨 코드 ──────────────────────────────────────────────────────────────


def test_to_codes_merges_stages_and_counts_drops():
    per_epoch = np.array(
        ["Sleep stage 3", "Sleep stage 4", "Sleep stage 1", "Sleep stage ?", "Movement time"],
        dtype=object,
    )
    codes, counts = to_codes(per_epoch, LABEL_MAP, DROP_LIST)
    assert list(codes) == [2, 2, 1, DROP, DROP]  # 3·4 → DS, 1 → LS
    assert counts == {"Sleep stage ?": 1, "Movement time": 1}


def test_to_codes_raises_on_unknown_annotation():
    """모르는 라벨을 조용히 흘려보내면 나중에 못 찾는다."""
    with pytest.raises(KeyError, match="모르는 주석"):
        to_codes(np.array(["Sleep stage 9"], dtype=object), LABEL_MAP, DROP_LIST)


# ── 에포크화 ───────────────────────────────────────────────────────────────


def test_epoch_signal_shape_and_content():
    data = np.arange(2 * 350, dtype="float32").reshape(2, 350)
    ep = epoch_signal(data, 100)
    assert ep.shape == (3, 2, 100)  # 꼬리 50샘플은 버림
    assert np.array_equal(ep[1, 0], data[0, 100:200])
    assert np.array_equal(ep[2, 1], data[1, 200:300])


# ── 절단 경계 ──────────────────────────────────────────────────────────────


def test_crop_bounds_keeps_edge_around_sleep():
    labels = np.array([0] * 10 + [1, 2, 3] + [0] * 10)
    assert crop_bounds(labels, wake_code=0, edge_epochs=2) == (8, 15)


def test_crop_bounds_clamps_at_array_edges():
    labels = np.array([1, 0, 0, 1])
    assert crop_bounds(labels, wake_code=0, edge_epochs=60) == (0, 4)


def test_crop_bounds_on_all_wake_keeps_everything():
    """전부 각성이면 자를 근거가 없다."""
    labels = np.zeros(5, dtype=np.int8)
    assert crop_bounds(labels, wake_code=0, edge_epochs=60) == (0, 5)


# ── 품질 지표 ──────────────────────────────────────────────────────────────


def test_quality_metrics_flags_flat_channel():
    epochs = np.zeros((1, 2, 100), dtype="float32")
    epochs[0, 1] = np.random.default_rng(0).normal(size=100)
    qm = quality_metrics(epochs)
    assert qm["flat_ratio"][0, 0] == 1.0  # 전극 탈락
    assert qm["flat_ratio"][0, 1] < 0.1
    assert qm["ptp"][0, 0] == 0.0
    assert all(v.shape == (1, 2) for v in qm.values())


# ── 파일 짝짓기 ────────────────────────────────────────────────────────────


def _touch(d, *names):
    for n in names:
        (d / n).write_bytes(b"")


def test_find_recordings_pairs_by_name(tmp_path):
    _touch(
        tmp_path,
        "SC4001E0-PSG.edf",
        "SC4001EC-Hypnogram.edf",
        "SC4002E0-PSG.edf",
        "SC4002EV-Hypnogram.edf",
    )
    recs = find_recordings(tmp_path)
    assert [(r.subject, r.night, r.key) for r in recs] == [
        ("SC400", 1, "SC400N1"),
        ("SC400", 2, "SC400N2"),
    ]


def test_find_recordings_raises_when_label_missing(tmp_path):
    """DeepSleepNet 은 목록을 정렬해 인덱스로 짝지어서, 빠지면 조용히 어긋난다."""
    _touch(tmp_path, "SC4001E0-PSG.edf", "SC4002E0-PSG.edf", "SC4002EV-Hypnogram.edf")
    with pytest.raises(FileNotFoundError, match="라벨 파일이 0개"):
        find_recordings(tmp_path)


def test_find_recordings_rejects_unknown_filename(tmp_path):
    _touch(tmp_path, "WEIRD-PSG.edf")
    with pytest.raises(ValueError, match="파일명 형식"):
        find_recordings(tmp_path)
