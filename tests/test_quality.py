"""녹음 단위 품질 검사.

여기 있는 것은 전부 틀려도 실행이 안 죽는 것들이다. 자르기를 빼먹으면 비율만
조용히 달라지고, 채널 조건을 뒤집으면 아무것도 안 걸리는데 오류도 안 난다.
"""

import json

import numpy as np
import pandas as pd
import pytest

from sleepstage.core.quality import assess, flat_fraction
from sleepstage.experiment.pipeline.quality import drop_excluded, load_excluded, scan, scan_one

CH = ["EEG Fpz-Cz", "EEG Pz-Oz"]
PARAMS = {"flat_uv": 20.0, "max_flat_frac": 0.20}


def _ptp(fpz, pz):
    return np.stack([np.asarray(fpz, float), np.asarray(pz, float)], axis=1)


def test_one_bad_channel_is_enough():
    """문헌은 모든 채널이 동시에 나빠야 배제한다. 그 규칙은 EOG 가 있을 때의 것이다.

    조건을 '모두' 로 되돌리면 한쪽만 죽은 녹음이 조용히 통과한다.
    """
    ptp = _ptp([80] * 100, [5] * 100)  # 이마는 정상, 뒤통수만 죽음
    out = assess(ptp, CH, PARAMS)
    assert out["exclude"]
    assert list(out["bad_channels"]) == ["EEG Pz-Oz"]


def test_healthy_recording_passes():
    out = assess(_ptp([80] * 100, [60] * 100), CH, PARAMS)
    assert not out["exclude"]
    assert out["bad_channels"] == {}


def test_threshold_is_a_fraction_not_a_count():
    """평탄 조각이 몇 개인지가 아니라 몇 퍼센트인지로 판단해야 한다.

    개수로 바꾸면 긴 녹음이 무조건 걸리고 짧은 녹음이 무조건 통과한다.
    """
    short = assess(_ptp([80] * 10, [5] * 3 + [80] * 7), CH, PARAMS)
    long = assess(_ptp([80] * 1000, [5] * 300 + [80] * 700), CH, PARAMS)
    assert short["exclude"] == long["exclude"]


def test_fraction_is_strictly_below_threshold():
    """경계값은 평탄이 아니다. 부등호를 뒤집으면 정상 조각이 평탄으로 세어진다."""
    assert flat_fraction(_ptp([20] * 10, [19.9] * 10), 20.0).tolist() == [0.0, 1.0]


def _write_npz(path, ptp, crop, config_hash="abc123", subject="SC499", night=1, idx=None):
    np.savez(
        path,
        epoch_idx=np.arange(len(ptp), dtype=np.int32) if idx is None else np.asarray(idx, np.int32),
        ptp=np.asarray(ptp, np.float32),
        ch_names=np.array(CH),
        subject=subject,
        night=np.int8(night),
        crop_lo=np.int32(crop[0]),
        crop_hi=np.int32(crop[1]),
        config_hash=config_hash,
    )
    return path


def test_scan_measures_inside_the_crop(tmp_path):
    """자른 구간에서 재야 한다.

    자르기 전에는 낮에 깨어 있던 구간이 절반을 넘는다. 그게 섞이면 밤이 통째로
    나쁜 녹음도 비율이 희석되어 멀쩡해 보인다. 실행은 멀쩡히 끝난다.
    """
    n = 200
    ptp = _ptp([80] * n, [80] * 100 + [5] * 100)  # 뒤쪽 절반만 평탄
    # 평탄한 뒤쪽 절반만 학습이 본다
    path = _write_npz(tmp_path / "SC499N1.npz", ptp, (100, 200))
    cropped = scan_one(path, PARAMS, use_crop=True)
    whole = scan_one(path, PARAMS, use_crop=False)
    assert cropped["flat_fraction"][1] == 1.0 and cropped["exclude"]
    assert whole["flat_fraction"][1] == 0.5
    assert cropped["n_epochs"] == 100


def test_drop_excluded_matches_on_key_not_position():
    """녹음을 키로 골라야 한다. 행 번호는 결측 제거를 거치며 다시 매겨진다."""
    table = (
        pd.DataFrame(
            {
                "subject": ["SC401", "SC401", "SC402", "SC402"],
                "night": [1, 2, 1, 2],
                "x": [0, 1, 2, 3],
            }
        )
        .drop(index=0)
        .reset_index(drop=True)
    )
    kept, info = drop_excluded(table, {("SC402", 1)})
    assert kept["x"].tolist() == [1, 3]
    assert info == {"n_recordings_excluded": 1, "n_rows_excluded": 1}


def test_no_exclusion_leaves_the_table_alone():
    table = pd.DataFrame({"subject": ["SC401"], "night": [1], "x": [0]})
    kept, info = drop_excluded(table, set())
    assert kept.equals(table) and info["n_rows_excluded"] == 0


def test_unknown_check_raises():
    with pytest.raises(KeyError):
        assess(_ptp([80] * 10, [80] * 10), CH, {**PARAMS, "checks": ["없는검사"]})


def test_attenuating_a_channel_makes_it_flagged():
    """진폭을 인위적으로 줄인 사본이 걸리는지.

    실제로 의심스러운 녹음을 보고 임계값을 고르면 순환이 된다. 이 시험은 그 녹음을
    쓰지 않으므로 규칙이 감쇠 자체를 잡는다는 것을 따로 보여준다.
    """
    healthy = np.full(200, 60.0)
    assert not assess(_ptp(healthy, healthy), CH, PARAMS)["exclude"]
    assert assess(_ptp(healthy, healthy * 0.3), CH, PARAMS)["exclude"]


def test_stale_report_is_refused(tmp_path):
    """에포크를 다시 만든 뒤 보고서를 안 고치면 옛 판정으로 학습이 돈다.

    실행은 멀쩡히 끝나고 배제 목록만 틀린다. 그래서 조용하다.
    """
    _write_npz(tmp_path / "SC499N1.npz", _ptp([80] * 20, [5] * 20), (0, 20), config_hash="old")
    report = tmp_path / "quality.json"
    report.write_text(json.dumps(scan(tmp_path, PARAMS)), encoding="utf-8")
    assert load_excluded(report) == {("SC499", 1)}

    _write_npz(tmp_path / "SC499N1.npz", _ptp([80] * 20, [5] * 20), (0, 20), config_hash="new")
    with pytest.raises(SystemExit):
        load_excluded(report)


def test_empty_crop_is_refused(tmp_path):
    """빈 구간은 비율이 nan 이 되고 nan > 임계값 은 False 라 조용히 통과한다."""
    _write_npz(tmp_path / "SC499N1.npz", _ptp([80] * 20, [5] * 20), (5, 5))
    with pytest.raises(ValueError, match="조각이 없습니다"):
        scan_one(tmp_path / "SC499N1.npz", PARAMS)


def test_mixed_epoch_configs_are_refused(tmp_path):
    """설정이 다른 npz 가 섞이면 한 보고서 안에서 기준이 갈린다."""
    _write_npz(tmp_path / "SC499N1.npz", _ptp([80] * 20, [80] * 20), (0, 20), config_hash="a")
    _write_npz(
        tmp_path / "SC499N2.npz", _ptp([80] * 20, [80] * 20), (0, 20), config_hash="b", night=2
    )
    with pytest.raises(ValueError, match="서로 다른 설정"):
        scan(tmp_path, PARAMS)


def test_crop_bounds_are_positions_not_epoch_numbers(tmp_path):
    """crop_lo/hi 는 걸러낸 배열 안의 위치다. features.py 도 data[lo:hi] 로 자른다.

    조각 번호와 비교하면 판독불가로 구멍이 난 녹음만 다른 조각을 보게 된다.
    구멍이 없는 녹음에서는 둘이 같은 답을 내서 시험 실행으로는 안 드러난다.
    """
    n = 10
    ptp = _ptp([80] * n, [80] * 5 + [5] * 5)  # 뒤쪽 다섯 개만 평탄
    # 번호가 0..4 다음 100..104 로 튄다. 판독불가로 구멍이 난 모양이다.
    idx = list(range(5)) + list(range(100, 105))
    path = _write_npz(tmp_path / "SC499N1.npz", ptp, (5, 10), idx=idx)

    out = scan_one(path, PARAMS)
    # 위치로 자르면 뒤쪽 다섯 개, 즉 평탄한 쪽만 남는다
    assert out["n_epochs"] == 5
    assert out["flat_fraction"][1] == 1.0
    # 번호로 잘랐다면 5 이상 10 미만인 번호가 없어 조각이 0개가 되고 nan 이 됐을 것이다
