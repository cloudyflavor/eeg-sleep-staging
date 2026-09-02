"""단계 전이 후처리를 서빙에서 쓸 때 조용히 틀릴 수 있는 것들.

여기 있는 것은 전부 틀려도 실행이 안 죽는다. 스트리밍 판이 실험 판과 다른 답을 내도
오류가 없고, 세션을 이어받을 때 흐름을 잃어도 판정은 계속 나온다.
"""

import json

import numpy as np
import pytest

from sleepstage.core.transitions import PastOnlyDecoder
from sleepstage.experiment.modeling.stage_transitions import decode_past_only

N = 4


def _table(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.log(rng.dirichlet(np.ones(N), size=N))


def _proba(n: int, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).dirichlet(np.ones(N), size=n)


@pytest.mark.parametrize("weight", [0.0, 0.1, 0.3, 1.0])
def test_streaming_matches_batch(weight):
    """실험은 녹음을 겹쳐 놓고 한 번에 풀고 서빙은 조각을 하나씩 받는다.

    두 판이 갈라지면 보고한 성능과 기기가 내는 답이 달라지는데 오류는 안 난다.
    """
    trans, proba = _table(), _proba(400)
    decoder = PastOnlyDecoder(trans, weight)
    streamed = [decoder.step(p) for p in proba]
    batch = decode_past_only([np.log(np.clip(proba, 1e-12, None))], weight * trans)[0]
    assert streamed == batch.tolist()


def test_state_round_trip_keeps_the_stream_going():
    """세션을 이어받을 때 흐름을 안 넘기면 그 시점부터 과거가 없는 것처럼 푼다."""
    trans, proba = _table(), _proba(60)
    whole = PastOnlyDecoder(trans, 0.3)
    expected = [whole.step(p) for p in proba]

    first = PastOnlyDecoder(trans, 0.3)
    got = [first.step(p) for p in proba[:25]]
    resumed = PastOnlyDecoder(trans, 0.3)
    resumed.load(json.loads(json.dumps(first.state())))  # 저장은 json 을 거친다
    got += [resumed.step(p) for p in proba[25:]]
    assert got == expected


def test_fresh_decoder_state_is_none():
    d = PastOnlyDecoder(_table(), 0.3)
    assert d.state() is None
    d.step(_proba(1)[0])
    assert d.state() is not None
    d.reset()
    assert d.state() is None


def test_weight_zero_is_the_same_as_no_smoothing():
    """세기 0 은 표를 안 믿는 것이므로 모델 답을 그대로 내야 한다."""
    trans, proba = _table(), _proba(120)
    d = PastOnlyDecoder(trans, 0.0)
    assert [d.step(p) for p in proba] == proba.argmax(axis=1).tolist()


def test_smoothing_removes_a_single_flicker():
    """후처리가 실제로 무언가를 바꾸는지. 안 바뀌면 붙여 놓고도 효과가 없다."""
    # 한 단계가 이어질 확률이 높은 표
    trans = np.log(np.eye(N) * 0.97 + (1 - np.eye(N)) * 0.01)
    proba = np.tile([0.9, 0.04, 0.03, 0.03], (9, 1))
    proba[4] = [0.4, 0.45, 0.1, 0.05]  # 한 칸만 살짝 다른 쪽으로 튐
    assert proba.argmax(axis=1)[4] == 1
    d = PastOnlyDecoder(trans, 1.0)
    assert [d.step(p) for p in proba][4] == 0


def test_wrong_shape_is_refused():
    d = PastOnlyDecoder(_table(), 0.3)
    with pytest.raises(ValueError):
        d.step(np.array([0.5, 0.5]))
    with pytest.raises(ValueError):
        PastOnlyDecoder(np.zeros((3, 4)))


def _config(with_transitions: bool) -> dict:
    config = {
        "channels": ["a", "b"],
        "sfreq": 100.0,
        "stage_names": ["W", "LS", "DS", "R"],
        "preprocess": {"context": "smooth", "filter": "zerophase", "normalize": "expanding"},
    }
    if with_transitions:
        config["transitions"] = {"weight": 0.3, "log_prob": _table().tolist()}
    return config


def test_stager_uses_the_table_when_the_artifact_has_one():
    """배포 파일에 표가 없으면 후처리 없이 돌아가야 하고, 있으면 걸려야 한다.

    조건을 뒤집어도 오류가 안 나고 성능만 달라진다.
    """
    pytest.importorskip("antropy")
    from sleepstage.serving.stream import StreamingStager

    plain = StreamingStager(model=None, config=_config(False), feature_names=[])
    smoothed = StreamingStager(model=None, config=_config(True), feature_names=[])
    assert plain.decoder is None
    assert smoothed.decoder is not None and smoothed.decoder.weight == 0.3


def test_snapshot_carries_the_decoder(tmp_path):
    """세션을 이어받을 때 흐름을 안 넘기면 그 시점부터 판정이 흔들린다."""
    pytest.importorskip("antropy")
    from sleepstage.serving.storage import SessionStore
    from sleepstage.serving.stream import StreamingStager

    config = _config(True)
    before = StreamingStager(model=None, config=config, feature_names=[])
    for p in _proba(20):
        before.decoder.step(p)

    store = SessionStore(tmp_path)
    store.save("s1", before)
    after = StreamingStager(model=None, config=config, feature_names=[])
    assert store.restore("s1", after) is True
    assert after.decoder.state() == pytest.approx(before.decoder.state())

    nxt = _proba(5, seed=9)
    assert [after.decoder.step(p) for p in nxt] == [before.decoder.step(p) for p in nxt]
