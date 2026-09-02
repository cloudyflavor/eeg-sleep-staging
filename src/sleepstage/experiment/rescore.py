"""저장된 예측에서 품질 배제 녹음을 빼고 다시 채점한다.

학습을 다시 하지 않는다. 배제 대상이 학습에도 들어간 채로 나온 예측이므로
여기 숫자는 평가에서만 뺀 값이다. 학습에서까지 빼면 값이 달라진다.
그래서 배제 전후를 나란히 남긴다. 뺀 값만 적으면 어려운 것을 치운 게 보이지 않는다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from sleepstage.experiment.modeling.cross_validate import _metrics

#: 채점할 열. 후처리를 건 예측이 있으면 그것도 같이 잰다.
COLUMNS = ("pred", "pred_smoothed")


def _score(table: pd.DataFrame, column: str) -> dict[str, float]:
    return _metrics(table["stage"].to_numpy(), table[column].to_numpy())


def _excluded_mask(table: pd.DataFrame, excluded: set[tuple[str, int]]):
    """배제 대상 행을 고르는 조건. 키로 고른다 — 인덱스는 표를 거치며 다시 매겨진다."""
    keys = zip(table["subject"], table["night"], strict=True)
    return pd.Series([(str(s), int(n)) in excluded for s, n in keys], index=table.index)


def rescore(run_dir: str | Path, excluded: set[tuple[str, int]]) -> dict[str, Any]:
    """배제 전과 후, 그리고 배제된 것만 각각 채점한다."""
    run = Path(run_dir)
    path = run / "predictions_smoothed.parquet"
    if not path.exists():
        path = run / "predictions.parquet"
    if not path.exists():
        raise SystemExit(f"예측 파일이 없습니다: {run}")

    table = pd.read_parquet(path)
    out = _excluded_mask(table, excluded)
    kept, dropped = table[~out], table[out]

    result: dict[str, Any] = {
        "source": str(path),
        "n_rows": len(table),
        "n_recordings_excluded": int(dropped[["subject", "night"]].drop_duplicates().shape[0]),
        "n_rows_excluded": int(out.sum()),
        "columns": {},
    }
    for column in COLUMNS:
        if column not in table.columns:
            continue
        entry = {"before": _score(table, column), "after": _score(kept, column)}
        if len(dropped):
            entry["excluded_only"] = _score(dropped, column)
        entry["gain_pp"] = (entry["after"]["macro_f1"] - entry["before"]["macro_f1"]) * 100
        result["columns"][column] = entry
    return result


def write(result: dict[str, Any], run_dir: str | Path) -> Path:
    path = Path(run_dir) / "rescored.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def format_result(result: dict[str, Any]) -> str:
    lines = [
        f"녹음 {result['n_recordings_excluded']}건 배제  "
        f"조각 {result['n_rows_excluded']} / {result['n_rows']}"
    ]
    for column, e in result["columns"].items():
        lines.append(f"\n  {column}")
        for label, key in (
            ("전체", "before"),
            ("배제 후", "after"),
            ("배제된 것만", "excluded_only"),
        ):
            m = e.get(key)
            if not m:
                continue
            per = "  ".join(f"{k[3:]} {v:.3f}" for k, v in m.items() if k.startswith("f1_"))
            lines.append(
                f"    {label:<12} macro-F1 {m['macro_f1']:.4f}  kappa {m['kappa']:.4f}   {per}"
            )
        lines.append(f"    {'차이':<12} {e['gain_pp']:+.2f}%p")
    return "\n".join(lines)
