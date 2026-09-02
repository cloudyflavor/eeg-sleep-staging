"""판정 결과와 특징과 세션 상태를 남깁니다.

지금은 판정이 나와도 아무 데도 남지 않아서, 컨테이너가 죽으면 그 밤의 결과가
사라집니다. 종류마다 크기가 크게 달라 저장하는 곳을 나눕니다.

    판정 결과   하룻밤 0.12MB   조회가 잦고 작다        SQLite
    특징 표     하룻밤 1.4MB    통째로 읽고 쓴다        parquet 파일
    원신호      하룻밤 28.8MB   기본으로 저장하지 않는다  parquet 파일
    세션 상태   수백 KB         다시 떴을 때 이어가려고   npz 와 json

DB 를 직접 부르는 자리를 이 파일 하나로 모읍니다. 서버를 여러 대로 늘릴 때
PostgreSQL 로 바꾸더라도 고칠 곳이 여기뿐입니다.
"""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    session       TEXT NOT NULL,
    epoch_index   INTEGER NOT NULL,
    stage         TEXT NOT NULL,
    confidence    REAL NOT NULL,
    p_w REAL, p_ls REAL, p_ds REAL, p_r REAL,
    model_version TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    PRIMARY KEY (session, epoch_index)
);
CREATE TABLE IF NOT EXISTS audit (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session       TEXT NOT NULL,
    event         TEXT NOT NULL,
    detail        TEXT,
    model_version TEXT,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_session ON audit(session);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class PredictionStore:
    """판정과 감사 기록을 남깁니다.

    같은 조각을 두 번 보내도 행이 늘지 않도록 세션과 조각 번호를 기본키로 둡니다.
    재전송 때문에 같은 시점의 판정이 여러 개 남으면 나중에 어느 쪽이 맞는지 알 수 없습니다.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path)
        con.execute("PRAGMA journal_mode=WAL")  # 쓰는 도중 죽어도 반쪽 행이 남지 않는다
        return con

    def record(self, session: str, decision: dict[str, Any], model_version: str) -> None:
        proba = decision["proba"]
        with self._connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    session,
                    decision["epoch_idx"],
                    decision["stage"],
                    decision["confidence"],
                    proba.get("W"),
                    proba.get("LS"),
                    proba.get("DS"),
                    proba.get("R"),
                    model_version,
                    _now(),
                ),
            )

    def log(self, session: str, event: str, detail: str = "", model_version: str = "") -> None:
        """무엇이 언제 일어났는지 남깁니다. 모델을 바꾼 뒤 문제가 생기면 이 기록으로 되짚습니다."""
        with self._connect() as con:
            con.execute(
                "INSERT INTO audit (session, event, detail, model_version, created_at) "
                "VALUES (?,?,?,?,?)",
                (session, event, detail, model_version, _now()),
            )

    def history(self, session: str) -> list[dict[str, Any]]:
        with self._connect() as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT * FROM predictions WHERE session = ? ORDER BY epoch_index", (session,)
            ).fetchall()
        return [dict(r) for r in rows]

    def count(self, session: str) -> int:
        with self._connect() as con:
            return con.execute(
                "SELECT COUNT(*) FROM predictions WHERE session = ?", (session,)
            ).fetchone()[0]


class BlobStore:
    """특징 표와 원신호를 파일로 남깁니다.

    통째로 읽고 통째로 쓰는 덩어리라 DB 에 넣을 이유가 없습니다. 읽고 쓰는 자리를
    여기로 모아두면 나중에 클라우드 저장소로 바꿀 때 이 파일만 고치면 됩니다.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def path_for(self, session: str, name: str) -> Path:
        return self.root / session / name

    def write_table(self, session: str, name: str, rows: list, columns: list[str]) -> Path | None:
        if not rows:
            return None
        import pandas as pd

        out = self.path_for(session, name)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=columns).to_parquet(out, index=False)
        return out


class SessionStore:
    """진행 중인 세션 상태를 저장해 서버가 다시 떴을 때 이어갈 수 있게 합니다.

    저장하지 않으면 서버가 죽는 순간 그 기기는 처음부터 다시 시작해야 하고,
    다시 답하기까지 또 4.5분이 걸립니다.
    """

    def __init__(self, root: str | Path, every: int = 20):
        self.root = Path(root)
        self.every = every

    def _paths(self, session: str) -> tuple[Path, Path]:
        base = self.root / session
        return base.with_suffix(".npz"), base.with_suffix(".json")

    def save(self, session: str, stager) -> None:
        arrays, meta = self._paths(session)
        arrays.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            arrays,
            raw=np.asarray(list(stager.raw_buffer), dtype="float32")
            if stager.raw_buffer
            else np.zeros((0, 0, 0), dtype="float32"),
        )
        meta.write_text(
            json.dumps(
                {
                    "received": stager.received,
                    "index": stager.index,
                    "buffer": list(stager.buffer),
                    "stats": stager.stats.history,
                    # 이것을 빼먹으면 세션을 이어받은 뒤 흐름이 처음부터 다시 시작한다
                    "decoder": stager.decoder.state() if stager.decoder else None,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def restore(self, session: str, stager) -> bool:
        arrays, meta = self._paths(session)
        if not meta.exists():
            return False
        state = json.loads(meta.read_text(encoding="utf-8"))
        stager.received = state["received"]
        stager.index = state["index"]
        stager.buffer.clear()
        stager.buffer.extend(state["buffer"])
        stager.stats.history = state["stats"]
        if stager.decoder is not None:
            stager.decoder.load(state.get("decoder"))
        if arrays.exists():
            raw = np.load(arrays)["raw"]
            stager.raw_buffer.clear()
            stager.raw_buffer.extend(raw[i] for i in range(len(raw)))
        return True

    def drop(self, session: str) -> None:
        for path in self._paths(session):
            path.unlink(missing_ok=True)
