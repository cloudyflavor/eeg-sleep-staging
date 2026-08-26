"""설정 로딩과 해시 동작 검증."""

import textwrap

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
