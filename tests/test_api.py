"""추론 API 가 실제로 판정을 돌려주는지 확인합니다.

모델 파일이 있어야 도는 시험이라 없으면 건너뜁니다.
"""

from pathlib import Path

import numpy as np
import pytest

ARTIFACT = Path("artifacts/model-v1")


@pytest.fixture
def client():
    pytest.importorskip("fastapi")
    if not (ARTIFACT / "config.json").exists():
        pytest.skip("모델 파일이 없습니다")
    from fastapi.testclient import TestClient

    from sleepstage.serving import api

    with TestClient(api.app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_unknown_session_is_rejected(client):
    """세션을 열지 않고 보낸 조각을 조용히 받아들이면 판정이 뒤섞인다."""
    body = {"samples": np.zeros((2, 3000)).tolist()}
    assert client.post("/sessions/nope/epochs", json=body).status_code == 404


def test_wrong_shape_is_rejected(client):
    """모양이 다른 입력을 그대로 넣으면 엉뚱한 값이 특징으로 들어간다."""
    client.post("/sessions/s1")
    body = {"samples": np.zeros((2, 100)).tolist()}
    assert client.post("/sessions/s1/epochs", json=body).status_code == 422


def test_answers_only_after_enough_epochs(client):
    """앞뒤 조각이 모이기 전에 답하면 그 판정은 근거가 없다."""
    client.post("/sessions/s2")
    rng = np.random.default_rng(0)
    answers = []
    for _ in range(40):
        body = {"samples": (rng.normal(0, 20, (2, 3000))).tolist()}
        answers.append(client.post("/sessions/s2/epochs", json=body).json())

    ready = [a for a in answers if a["ready"]]
    assert not any(a["ready"] for a in answers[:10]), "너무 일찍 답했습니다"
    assert ready, "끝까지 한 번도 답하지 않았습니다"
    assert ready[0]["stage"] in ("W", "LS", "DS", "R")
    assert 0.0 <= ready[0]["confidence"] <= 1.0

    idx = [a["epoch_index"] for a in ready]
    assert idx == sorted(idx), "판정 순서가 뒤섞였습니다"


def test_predictions_are_kept_after_the_session_closes(client, tmp_path):
    """판정이 남지 않으면 컨테이너가 죽는 순간 그 밤의 결과가 사라진다."""
    rng = np.random.default_rng(1)
    client.post("/sessions/keep")
    for _ in range(30):
        client.post(
            "/sessions/keep/epochs", json={"samples": rng.normal(0, 20, (2, 3000)).tolist()}
        )

    closed = client.delete("/sessions/keep").json()
    assert closed["stored"] > 0, "판정이 하나도 저장되지 않았습니다"

    got = client.get("/sessions/keep/predictions").json()
    assert got["count"] == closed["stored"]
    idx = [r["epoch_index"] for r in got["predictions"]]
    assert idx == sorted(set(idx)), "조각 번호가 겹치거나 뒤섞였습니다"
    assert got["predictions"][0]["model_version"]


def test_resume_continues_instead_of_starting_over(client):
    """이어가지 못하면 서버가 죽을 때마다 그 기기는 처음부터 다시 쌓아야 한다."""
    rng = np.random.default_rng(2)
    client.post("/sessions/again")
    for _ in range(25):
        client.post(
            "/sessions/again/epochs", json={"samples": rng.normal(0, 20, (2, 3000)).tolist()}
        )

    fresh = client.post("/sessions/again").json()
    assert fresh["received"] == 0

    resumed = client.post("/sessions/again?resume=true").json()
    assert resumed["resumed"] is True
    assert resumed["received"] > 0, "저장해 둔 진행 상태를 이어받지 못했습니다"
