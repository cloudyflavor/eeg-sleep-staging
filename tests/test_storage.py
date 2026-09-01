"""판정과 세션 상태가 실제로 남는지 확인합니다."""

import numpy as np
import pytest


def decision(idx, stage="LS", conf=0.7):
    return {
        "epoch_idx": idx,
        "stage": stage,
        "confidence": conf,
        "proba": {"W": 0.1, "LS": conf, "DS": 0.1, "R": 0.1},
    }


def test_same_epoch_twice_does_not_duplicate(tmp_path):
    """같은 조각이 다시 오면 행이 늘어난다. 나중에 어느 판정이 맞는지 알 수 없게 된다."""
    from sleepstage.serving.storage import PredictionStore

    store = PredictionStore(tmp_path / "p.db")
    store.record("s1", decision(7, "LS", 0.7), "v1")
    store.record("s1", decision(7, "DS", 0.9), "v1")

    rows = store.history("s1")
    assert len(rows) == 1, "같은 조각이 두 행으로 남았습니다"
    assert rows[0]["stage"] == "DS", "나중 판정으로 갱신되지 않았습니다"


def test_audit_keeps_model_version(tmp_path):
    """어느 모델이 낸 판정인지 남지 않으면 모델을 바꾼 뒤 문제를 되짚을 수 없다."""
    from sleepstage.serving.storage import PredictionStore

    store = PredictionStore(tmp_path / "p.db")
    store.record("s1", decision(1), "model-v2")
    store.log("s1", "open", model_version="model-v2")

    assert store.history("s1")[0]["model_version"] == "model-v2"


def test_session_snapshot_restores_progress(tmp_path):
    """이어가지 못하면 서버가 죽을 때마다 그 기기는 다시 4.5분을 기다려야 한다."""
    pytest.importorskip("antropy")
    from sleepstage.serving.storage import SessionStore
    from sleepstage.serving.stream import StreamingStager

    config = {
        "channels": ["a", "b"],
        "sfreq": 100.0,
        "stage_names": ["W", "LS", "DS", "R"],
        "preprocess": {"context": "smooth", "filter": "zerophase", "normalize": "expanding"},
    }
    before = StreamingStager(model=None, config=config, feature_names=[])
    before.received, before.index = 37, 35
    before.stats.history = {"x": [1.0, 2.0, 3.0]}
    before.buffer.append({"x": 1.0})
    before.raw_buffer.append(np.ones((2, 3000), dtype="float32"))

    store = SessionStore(tmp_path)
    store.save("s1", before)

    after = StreamingStager(model=None, config=config, feature_names=[])
    assert store.restore("s1", after) is True
    assert after.received == 37
    assert after.index == 35
    assert after.stats.history == {"x": [1.0, 2.0, 3.0]}
    assert len(after.raw_buffer) == 1
    assert after.raw_buffer[0].shape == (2, 3000)

    store.drop("s1")
    assert store.restore("s1", after) is False


def test_unknown_model_type_is_rejected(tmp_path):
    """모르는 종류를 다른 것으로 읽으려 들면 엉뚱한 오류가 나거나 조용히 넘어간다."""
    import json

    from sleepstage.serving.export import MODEL_LOADERS, load_artifact

    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "torch", "model_file": "m.pt"}), encoding="utf-8"
    )
    (tmp_path / "feature_names.json").write_text(json.dumps({"names": []}), encoding="utf-8")

    with pytest.raises(ValueError) as e:
        load_artifact(tmp_path)
    assert "torch" in str(e.value)
    assert set(MODEL_LOADERS) == {"catboost", "xgboost", "lightgbm"}
