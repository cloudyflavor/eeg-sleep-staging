"""판정 기능을 HTTP 로 여는 창구입니다.

기기는 30초마다 조각 하나를 보내고, 답할 준비가 되면 단계와 확신도를 받습니다.
아직 답할 수 없는 동안에는 결과 없이 몇 조각이 더 필요한지만 알려줍니다.

기기마다 진행 상태가 다르므로 세션을 따로 둡니다. 세션은 지금 프로세스 메모리에
있습니다. 서버를 여러 대로 늘리려면 같은 기기의 요청이 같은 서버로 가야 하고,
서버가 죽으면 그 세션은 처음부터 다시 시작해야 합니다.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sleepstage.serving.export import artifact_version, load_artifact
from sleepstage.serving.storage import BlobStore, PredictionStore, SessionStore
from sleepstage.serving.stream import StreamingStager

#: 모델 파일이 있는 곳. 컨테이너에서는 환경변수로 바꿉니다.
ARTIFACT_DIR = os.environ.get("SLEEPSTAGE_ARTIFACT", "artifacts/model-v1")

#: 판정과 기록을 남길 곳
DATA_DIR = Path(os.environ.get("SLEEPSTAGE_DATA", "data/serving"))

#: 원신호는 하룻밤에 29MB 라 기본으로 저장하지 않습니다
KEEP_RAW = os.environ.get("SLEEPSTAGE_KEEP_RAW", "0") == "1"

_model: Any = None
_config: dict = {}
_names: list[str] = []
_version: str = "unknown"
_sessions: dict[str, StreamingStager] = {}
_feature_rows: dict[str, list] = {}
_predictions: PredictionStore | None = None
_blobs: BlobStore | None = None
_snapshots: SessionStore | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    """모델을 시작할 때 한 번만 읽습니다. 요청마다 읽으면 매번 수백 밀리초를 버립니다."""
    global _model, _config, _names, _version, _predictions, _blobs, _snapshots
    _model, _config, _names = load_artifact(ARTIFACT_DIR)
    _version = artifact_version(ARTIFACT_DIR)
    _predictions = PredictionStore(DATA_DIR / "predictions.db")
    _blobs = BlobStore(DATA_DIR / "blobs")
    _snapshots = SessionStore(DATA_DIR / "sessions")
    yield
    for name in list(_sessions):
        _flush(name)
    _sessions.clear()


app = FastAPI(title="Sleep Staging", version="1.0", lifespan=lifespan)


class Epoch(BaseModel):
    """조각 하나. 채널마다 30초치 표본을 담습니다."""

    samples: list[list[float]] = Field(description="채널 수 x 3000, 단위는 마이크로볼트")


class Decision(BaseModel):
    session: str
    received: int
    ready: bool
    epoch_index: int | None = None
    stage: str | None = None
    confidence: float | None = None
    probabilities: dict[str, float] | None = None


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if _model is not None else "loading",
        "artifact": ARTIFACT_DIR,
        "model_version": _version,
        "model_type": _config.get("model_type"),
        "sessions": len(_sessions),
        "data_dir": str(DATA_DIR),
        "keep_raw": KEEP_RAW,
        "lag_epochs": StreamingStager(_model, _config, _names).lag if _model else None,
    }


@app.post("/sessions/{session}")
def open_session(session: str, resume: bool = False) -> dict[str, Any]:
    """기기 하나의 측정을 시작합니다.

    ``resume`` 을 주면 저장해 둔 상태에서 이어갑니다. 서버가 죽었다 다시 떴을 때
    처음부터 다시 쌓으면 답이 나오기까지 또 4.5분이 걸립니다.
    """
    stager = StreamingStager(_model, _config, _names)
    resumed = bool(resume and _snapshots.restore(session, stager))
    _sessions[session] = stager
    _feature_rows[session] = []
    _predictions.log(session, "resume" if resumed else "open", model_version=_version)
    return {"session": session, "status": "open", "resumed": resumed, "received": stager.received}


@app.delete("/sessions/{session}")
def close_session(session: str) -> dict[str, Any]:
    """측정을 끝냅니다. 창이 다 차지 않아 미뤄뒀던 조각들을 마저 판정해 돌려줍니다."""
    if session not in _sessions:
        raise HTTPException(404, f"열려 있지 않은 세션입니다: {session}")
    pending = _flush(session)
    _snapshots.drop(session)
    return {
        "session": session,
        "status": "closed",
        "pending": pending,
        "stored": _predictions.count(session),
    }


@app.get("/sessions/{session}/predictions")
def read_predictions(session: str) -> dict[str, Any]:
    """남겨둔 판정을 돌려줍니다. 측정이 끝난 뒤에도 조회할 수 있습니다."""
    rows = _predictions.history(session)
    if not rows:
        raise HTTPException(404, f"남아 있는 판정이 없습니다: {session}")
    return {"session": session, "count": len(rows), "predictions": rows}


def _flush(session: str) -> list[dict[str, Any]]:
    """미뤄둔 판정을 마저 내고 특징 표를 파일로 씁니다."""
    stager = _sessions.pop(session, None)
    if stager is None:
        return []
    pending = stager.flush()
    for answer in pending:
        _predictions.record(session, answer, _version)
    rows = _feature_rows.pop(session, [])
    written = _blobs.write_table(session, "features.parquet", rows, ["epoch_index", *_names])
    _predictions.log(session, "close", f"features={written}", model_version=_version)
    return pending


@app.post("/sessions/{session}/epochs", response_model=Decision)
def push_epoch(session: str, epoch: Epoch) -> Decision:
    """조각 하나를 넣습니다. 준비가 됐으면 몇 조각 전의 판정이 함께 옵니다."""
    stager = _sessions.get(session)
    if stager is None:
        raise HTTPException(404, f"열려 있지 않은 세션입니다: {session}")

    array = np.asarray(epoch.samples, dtype=np.float32)
    try:
        answer = stager.push(array)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e

    if stager.received % _snapshots.every == 0:
        _snapshots.save(session, stager)

    if answer is None:
        return Decision(session=session, received=stager.received, ready=False)

    _predictions.record(session, answer, _version)
    if stager.last_row is not None:
        _feature_rows[session].append([answer["epoch_idx"], *stager.last_row.tolist()])
    return Decision(
        session=session,
        received=stager.received,
        ready=True,
        epoch_index=answer["epoch_idx"],
        stage=answer["stage"],
        confidence=round(answer["confidence"], 4),
        probabilities={k: round(v, 4) for k, v in answer["proba"].items()},
    )
