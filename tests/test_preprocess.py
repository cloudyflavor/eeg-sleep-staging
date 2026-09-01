"""에포크 npz 를 만드는 부분의 단위 테스트. 실제 EDF 없이 도는 것만 담는다."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sleepstage.core.context import shifted
from sleepstage.core.edf import expand_annotations, find_recordings
from sleepstage.core.epochs import DROP, crop_bounds, epoch_signal, quality_metrics, to_codes
from sleepstage.core.normalize import MIN_PERIODS, expanding_robust

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


def test_fold_masks_keep_whole_subjects():
    """사람이 통째로 한쪽에 들어가야 한다. 에포크 단위로 갈리면 누수다."""
    from sleepstage.experiment.split import fold_masks

    table = pd.DataFrame({"subject": ["A"] * 3 + ["B"] * 3 + ["C"] * 3, "x": range(9)})
    train_mask, test_mask = fold_masks(table, {"train": ["A", "B"], "test": ["C"]})
    # 피험자는 통째로 들어가거나 통째로 빠진다
    assert table.loc[train_mask, "subject"].value_counts().eq(3).all()
    assert not (train_mask & test_mask).any()


def test_feature_subsets_select_expected_columns():
    """단일 채널 부분집합에 반대 채널·ratio 열이 새면 ablation 이 무의미해진다."""
    from sleepstage.experiment.train import SUBSETS

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


def test_every_setting_changes_the_output_name():
    """설정 하나라도 바뀌면 파일 이름이 달라져야 한다.

    이름에서 빠진 설정이 있으면 내용이 다른 표가 같은 이름을 쓰고, 이미 있다는
    이유로 조용히 재사용된다. 값이 틀리는 게 아니라 엉뚱한 파일을 읽는 사고다.
    """
    from sleepstage.experiment.extract import SETTING_KEYS, settings_hash

    base = {
        "filter": "zerophase",
        "normalize": "expanding",
        "context": "smooth",
        "distance_features": True,
        "welch_average": "median",
        "window_extremes": False,
        "spindle_events": False,
        "channel_correlation": False,
    }
    assert set(base) == set(SETTING_KEYS)  # 키가 늘면 이 테스트도 같이 늘려야 한다

    names = {settings_hash({"features": base})}
    for key, value in base.items():
        other = dict(base)
        other[key] = "mean" if key == "welch_average" else not_like(value)
        names.add(settings_hash({"features": other}))
    assert len(names) == len(base) + 1


def not_like(value):
    return not value if isinstance(value, bool) else f"{value}_다른값"


def test_added_features_get_their_own_folder():
    """더한 실험이 기준선과 같은 폴더에 들어가면 무엇을 돌린 결과인지 알 수 없다."""
    from sleepstage.experiment.naming import addition_slug, run_dir

    base = {"filter": "zerophase", "normalize": "expanding", "context": "smooth"}
    assert addition_slug(base) == ""
    assert addition_slug({**base, "welch_average": "median"}) == ""  # 기본값은 안 붙는다
    both = {**base, "welch_average": "mean", "spindle_events": True}
    assert addition_slug(both) == "mean+spindles"

    # 줄인 것은 개선 계열에서만 이름에 붙는다. 탐색 계열은 options 칸이 이미 쓴다.
    assert addition_slug(base, {"feature_subset": "no_p2min"}) == ""
    assert addition_slug(both, {"feature_select": "top:200"}) == "mean+spindles-top200"
    assert addition_slug(both, {"feature_subset": "no_p2min"}) == "mean+spindles-no_p2min"

    args = ("runs", "catboost", "zerophase-expanding-smooth", "cw-balanced", "abcdef123456")
    assert str(run_dir(*args)) == "runs/catboost/zerophase-expanding-smooth/cw-balanced/abcdef"
    assert str(run_dir(*args, "mean+spindles")) == "runs/improve/mean+spindles/abcdef"


def test_subject_info_must_cover_every_recording():
    """한 녹음이라도 나이가 비면 그 녹음만 다른 조건으로 학습되고 실행해도 안 보인다."""
    import pandas as pd
    import pytest

    from sleepstage.experiment import subjects

    table = pd.DataFrame({"subject": ["SC400", "SC401"], "night": [1, 1], "x": [0.0, 1.0]})
    info = pd.DataFrame({"subject": ["SC400"], "night": [1], "age": [33], "female": [1]})
    with pytest.MonkeyPatch.context() as m:
        m.setattr(subjects, "load", lambda _path: info)
        with pytest.raises(ValueError, match="피험자 정보가 없는"):
            subjects.attach(table, "x.xls", "age")
        got, cols = subjects.attach(table.iloc[:1], "x.xls", "age+sex")
    assert cols == ["age", "female"]
    assert got["age"].tolist() == [33.0]


def test_age_weight_evens_out_the_bands():
    """조각 수가 많은 나이대가 학습을 지배하지 않아야 한다."""
    import numpy as np
    import pandas as pd

    from sleepstage.experiment import subjects

    table = pd.DataFrame({"age": [30.0] * 90 + [85.0] * 10})
    w = subjects.age_band_weight(table, np.ones(100, dtype=bool))
    assert w[:90].sum() == pytest.approx(w[90:].sum())  # 두 구간의 무게 총합이 같다


def test_channel_correlation_tells_shape_from_size():
    """세기 비율만으로는 서파와 이마 사건이 안 갈린다. 닮았는지를 봐야 갈린다."""
    from sleepstage.core.temporal import band_correlation

    sfreq, t = 100.0, np.arange(3000) / 100.0
    slow = 40 * np.sin(2 * np.pi * 0.8 * t)
    # 머리 전체가 같이 출렁이는 서파. 뒤통수가 작을 뿐 파형은 같다.
    together = band_correlation(slow[None], 0.5 * slow[None], sfreq, {"eog": (0.3, 2.0)})
    # 이마에서만 나는 사건. 뒤통수는 무관한 신호다.
    apart = band_correlation(
        slow[None], (40 * np.sin(2 * np.pi * 1.3 * t + 1.0))[None], sfreq, {"eog": (0.3, 2.0)}
    )
    assert together["corr_eog"][0] > 0.9
    assert abs(apart["corr_eog"][0]) < 0.5
    # 세기 비율은 두 경우가 똑같아서 못 가른다
    assert together["corr_broad"][0] > apart["corr_broad"][0]


def test_budget_filters_columns_by_lookahead():
    """예산이 특징의 필요 미래보다 작으면 그 열은 빠져야 한다"""
    from sleepstage.experiment.budget import columns_within

    look = {"a": 0, "b": 2, "c7min": 7}
    assert columns_within(look, 0) == ["a"]
    assert columns_within(look, 1) == ["a", "b"]
    assert columns_within(look, 4) == ["a", "b", "c7min"]
    assert columns_within(look, None) == ["a", "b", "c7min"]


def test_sequence_eval_scores_every_epoch_once():
    """조각 하나가 두 번 채점되거나 빠지면 점수가 조용히 달라진다."""
    import numpy as np

    pytest.importorskip("torch")
    from sleepstage.experiment.deep.data import SequenceDataset

    length = 15
    for n in (length, 40, 137):
        signals = np.zeros((n, 2, 4), dtype="float32")
        labels = np.arange(n, dtype="int64")  # 조각 번호를 라벨 자리에 넣어 추적한다
        ds = SequenceDataset([(signals, labels)], length, stride=1, score_all=True)

        scored = np.concatenate([y[m].numpy() for _, y, m in (ds[i] for i in range(len(ds)))])
        assert sorted(scored) == list(range(n)), f"n={n} 에서 채점 대상이 어긋납니다"


def test_sequence_windows_do_not_cross_recordings():
    """다른 사람의 밤이 한 묶음에 섞이면 누수다."""
    import numpy as np

    pytest.importorskip("torch")
    from sleepstage.experiment.deep.data import SequenceDataset

    a = (np.zeros((20, 2, 4), dtype="float32"), np.zeros(20, dtype="int8"))
    b = (np.ones((20, 2, 4), dtype="float32"), np.ones(20, dtype="int8"))
    ds = SequenceDataset([a, b], length=15, stride=1, score_all=True)

    for i in range(len(ds)):
        x, _, _ = ds[i]
        assert len(x.unique()) == 1, "한 묶음 안에 두 녹음이 섞였습니다"


def test_reported_lag_includes_filter_padding():
    """필터 여유분을 빼고 세면 실제보다 짧은 지연을 보고하게 된다."""
    pytest.importorskip("antropy")
    from sleepstage.serving.stream import StreamingStager

    config = {
        "channels": ["a", "b"],
        "sfreq": 100.0,
        "stage_names": ["W", "LS", "DS", "R"],
        "preprocess": {"context": "smooth", "filter": "zerophase", "normalize": "expanding"},
    }
    stager = StreamingStager(model=None, config=config, feature_names=[])
    assert stager.lag == stager.delay + stager.filter_pad
    assert stager.lag > stager.delay


def test_relative_band_powers_sum_to_one():
    """합이 1이 아니면 상대 세기가 아니고, 부족분이 델타 우세도와 상관된다."""
    from sleepstage.core.spectral import BANDS, band_powers, welch_psd

    rng = np.random.default_rng(0)
    sfreq = 100.0
    t = np.arange(3000) / sfreq
    for freq, amp in ((1.0, 50.0), (10.0, 20.0), (20.0, 5.0)):
        sig = (amp * np.sin(2 * np.pi * freq * t) + rng.normal(0, 3, 3000))[None, :]
        got = band_powers(*welch_psd(sig, sfreq))
        total = sum(float(got[name][0]) for name in BANDS)
        assert abs(total - 1.0) < 1e-4, f"{freq}Hz 신호에서 합이 {total:.4f}"


def test_selection_keeps_one_of_two_identical_columns():
    """닮은 열을 안 걷어내면 가지치기가 이름만 가지치기다."""
    pytest.importorskip("lightgbm")
    from sleepstage.experiment.select import choose

    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 600)
    signal = y + rng.normal(0, 0.3, 600)
    X = np.column_stack([signal, signal, rng.normal(0, 1, 600)]).astype("float32")

    assert choose("none", X, y) is None
    kept = choose("decorr:0.95", X, y, threads=1)
    assert len(kept) == 2  # 똑같은 두 열 중 하나만 남는다
    assert len(choose("top:2", X, y, threads=1)) == 2
    with pytest.raises(ValueError):
        choose("top", X, y, threads=1)


def test_selection_never_sees_the_rows_it_is_scored_on(tmp_path, monkeypatch):
    """평가 행을 보고 열을 고르면 점수가 부풀고 실행해도 안 보인다."""
    import json

    from sleepstage.config import Config
    from sleepstage.experiment import select, train

    rng = np.random.default_rng(0)
    subjects = [f"SC4{i:02d}" for i in range(10)]
    rows = pd.DataFrame(
        {
            "subject": np.repeat(subjects, 40),
            "night": 1,
            "epoch_idx": np.tile(np.arange(40), 10),
            "stage": rng.integers(0, 4, 400),
            "a": rng.normal(size=400),
            "b": rng.normal(size=400),
        }
    )
    features = tmp_path / "t.parquet"
    rows.to_parquet(features, index=False)
    (tmp_path / "t.manifest.json").write_text(
        json.dumps({"settings": {"filter": "none", "normalize": "none", "context": "none"}})
    )
    splits = tmp_path / "s.json"
    splits.write_text(
        json.dumps(
            {
                "folds": [
                    {"train": subjects[5:], "test": subjects[:5]},
                    {"train": subjects[:5], "test": subjects[5:]},
                ]
            }
        )
    )

    seen = []

    def spy(name, X, y, sw=None, threads=8):
        seen.append(len(X))
        return np.array([0])

    monkeypatch.setattr(select, "choose", spy)
    train.run_cv(
        Config(
            tmp_path / "cfg.yaml",
            {
                "train": {
                    "features": str(features),
                    "splits": str(splits),
                    "model": "majority",
                    "class_weight": "none",
                    "seed": 0,
                    "drop_missing": True,
                    "feature_subset": "all",
                    "feature_select": "top:1",
                    "subject_info": "none",
                    "subject_table": "",
                    "out_dir": str(tmp_path / "runs"),
                    "mlflow": False,
                }
            },
            "testhash",
        )
    )

    assert seen, "가지치기가 아예 불리지 않았다"
    assert all(n < len(rows) for n in seen), f"평가 행까지 보고 골랐다: {seen}"
    assert set(seen) == {200}
