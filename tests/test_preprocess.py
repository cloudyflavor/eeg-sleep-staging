"""1단계 전처리 단위 테스트 — 실제 EDF 없이 도는 것만."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sleepstage.features.context import shifted
from sleepstage.features.normalize import MIN_PERIODS, expanding_robust
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


# ── 특징 추출 ──────────────────────────────────────────────────────────────


def test_shift_marks_only_future_columns_with_lookahead():
    """shift(+1) 은 과거를 끌어온다. 부호를 반대로 적으면 미래를 보는 특징이
    lookahead 0 으로 기록돼 '실시간 가능' 이라고 잘못 보고하게 된다."""
    feats = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out, lookahead = shifted(feats, np.arange(5), shifts=(-1, 1))

    assert lookahead["x_ep-1"] == 0  # 과거 참조 — 실시간 가능
    assert lookahead["x_ep+1"] == 1  # 미래 참조 — 1 에포크 지연 필요
    assert out.loc[2, "x_ep-1"] == 2.0  # 한 칸 전 값
    assert out.loc[2, "x_ep+1"] == 4.0  # 한 칸 뒤 값


def test_shift_leaves_gaps_missing():
    """판독불가 제거로 생긴 구멍을 이웃인 척하면 안 된다."""
    feats = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    out, _ = shifted(feats, np.array([10, 11, 20]), shifts=(1,))  # 11 → 20 은 9칸 차이

    assert out.loc[1, "x_ep-1"] == 1.0  # 10 → 11 은 연속
    assert pd.isna(out.loc[2, "x_ep-1"])  # 11 → 20 은 구멍


def test_expanding_normalization_uses_only_the_past():
    """뒤쪽 값이 바뀌어도 앞쪽 정규화 결과는 그대로여야 한다."""
    a = pd.DataFrame({"x": np.arange(60, dtype=float)})
    b = a.copy()
    b.loc[50:, "x"] = 999.0

    na, nb = expanding_robust(a), expanding_robust(b)
    pd.testing.assert_series_equal(na["x"][:50], nb["x"][:50])


def test_expanding_normalization_is_missing_during_warmup():
    out = expanding_robust(pd.DataFrame({"x": np.arange(30, dtype=float)}))
    assert out["x"][: MIN_PERIODS - 1].isna().all()
    assert out["x"][MIN_PERIODS:].notna().all()


def test_learning_curve_subsamples_whole_subjects():
    """부분집합도 피험자 단위여야 한다. 에포크 단위로 줄이면 누수가 그대로다."""
    from sleepstage.evaluation.split import fold_masks

    table = pd.DataFrame({"subject": ["A"] * 3 + ["B"] * 3 + ["C"] * 3, "x": range(9)})
    train_mask, test_mask = fold_masks(table, {"train": ["A", "B"], "test": ["C"]})
    # 피험자는 통째로 들어가거나 통째로 빠진다
    assert table.loc[train_mask, "subject"].value_counts().eq(3).all()
    assert not (train_mask & test_mask).any()


def test_feature_subsets_select_expected_columns():
    """단일 채널 부분집합에 반대 채널·ratio 열이 새면 ablation 이 무의미해진다."""
    from sleepstage.evaluation.train import SUBSETS

    cols = [
        "EEG Fpz-Cz__fdelta",
        "EEG Pz-Oz__fdelta_norm_c7min",
        "ratio__alpha_back_front",
        "EEG Fpz-Cz__mmd_c7min",
        "EEG Pz-Oz__esis_norm",
        "time_hour",
    ]

    def pick(name):
        return [c for c in cols if SUBSETS[name](c)]

    assert pick("fpz_only") == ["EEG Fpz-Cz__fdelta", "EEG Fpz-Cz__mmd_c7min", "time_hour"]
    assert "ratio__alpha_back_front" not in pick("pz_only")
    assert pick("no_distance") == [
        "EEG Fpz-Cz__fdelta",
        "EEG Pz-Oz__fdelta_norm_c7min",
        "ratio__alpha_back_front",
        "time_hour",
    ]


def test_search_space_includes_default_first():
    """기본값을 반드시 후보에 넣어야 '튜닝이 기본값보다 나은가' 를 말할 수 있다."""
    from sleepstage.evaluation.tune import CATBOOST_SPACE, sample_space

    cand = sample_space(CATBOOST_SPACE, 8, seed=0)
    assert cand[0] == {"depth": 6, "iterations": 1000, "learning_rate": 0.1, "l2_leaf_reg": 3}
    assert len({json.dumps(c, sort_keys=True) for c in cand}) == 8  # 중복 없음
