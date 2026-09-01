"""설정 로딩과 해시 동작 검증."""

import textwrap

import pytest

from sleepstage.config import Config, config_hash


def test_load_keeps_data_path_and_hash_together(tmp_path):
    """설정과 해시가 짝을 이뤄 다니는 게 이 클래스의 존재 이유다."""
    p = tmp_path / "cfg.yaml"
    p.write_text(
        textwrap.dedent("""
        crop:
          wake_edge_epochs: 60
    """),
        encoding="utf-8",
    )

    cfg = Config.load(p)
    assert cfg["crop"]["wake_edge_epochs"] == 60
    assert cfg.path == p
    assert cfg.hash == config_hash(cfg.data)


def test_hash_is_order_independent():
    """키 순서가 달라도 같은 설정이면 같은 해시여야 한다."""
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})


def test_hash_changes_with_value():
    """값이 바뀌면 해시도 바뀌어야 재실행 판단이 가능하다."""
    assert config_hash({"edge_epochs": 60}) != config_hash({"edge_epochs": 0})


def test_override_applies_dotted_keys(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("features:\n  filter: causal\n  crop: true\n", encoding="utf-8")
    cfg = Config.load(p).override(["features.filter=none", "features.crop=false"])
    assert cfg["features"] == {"filter": "none", "crop": False}


def test_override_rejects_unknown_key(tmp_path):
    """오타가 조용히 새 키를 만들면 다른 실험이 된 줄도 모른다."""
    p = tmp_path / "cfg.yaml"
    p.write_text("features:\n  filter: causal\n", encoding="utf-8")
    with pytest.raises(KeyError):
        Config.load(p).override(["features.filtre=none"])


def test_override_leaves_original_untouched(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("features:\n  filter: causal\n", encoding="utf-8")
    cfg = Config.load(p)
    cfg.override(["features.filter=none"])
    assert cfg["features"]["filter"] == "causal"


def test_code_fingerprint_follows_values_not_text():
    """값이 같으면 코드를 어떻게 고쳤든 같은 지문이고, 값이 다르면 다른 지문이어야 한다."""
    pytest.importorskip("antropy")
    from unittest import mock

    import sleepstage.output_hash as fp
    from sleepstage.core import spectral

    before = fp.code_fingerprint()
    assert before == fp.code_fingerprint(), "같은 코드인데 지문이 달라졌습니다"
    assert len(before) == fp.LENGTH

    # 대역 하나를 살짝 옮기면 값이 바뀌므로 지문도 바뀌어야 한다
    moved = {**spectral.BANDS, "alpha": (8.0, 12.5)}
    with mock.patch.dict(spectral.BANDS, moved, clear=True):
        assert fp.code_fingerprint() != before, "값이 바뀌었는데 지문이 그대로입니다"


def test_fingerprint_mismatch_stops_and_explains():
    """조용히 건너뛰면 고친 줄 알고 옛 값으로 계속 실험하게 된다."""
    import pytest

    from sleepstage.output_hash import check_or_explain

    check_or_explain(None, "abc123", "x")  # 기록이 없으면 통과
    check_or_explain("abc123", "abc123", "x")  # 같으면 통과
    with pytest.raises(SystemExit) as e:
        check_or_explain("abc123", "def456", "features.parquet")
    assert "--overwrite" in str(e.value)
