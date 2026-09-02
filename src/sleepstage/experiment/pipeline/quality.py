"""에포크 npz 를 훑어 녹음별 품질 보고서를 만든다.

특징을 뽑기 전에 돌린다. 배제 목록을 파일 하나로 고정해두면 학습과 평가와
서빙이 같은 목록을 본다. 실행할 때마다 다시 판정하면 조건이 갈린다.

보고서는 만들기만 하고 아무것도 지우지 않는다. 무엇을 뺄지는 학습 쪽이 정한다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from sleepstage.core.quality import assess

#: 보고서 파일 이름. 에포크 npz 와 같은 폴더에 둔다.
REPORT_NAME = "quality.json"


def scan_one(path: Path, params: dict[str, Any], use_crop: bool = True) -> dict[str, Any]:
    """녹음 하나를 검사한다. 신호는 안 읽고 저장된 지표만 읽는다."""
    with np.load(path, allow_pickle=False) as z:
        ptp = z["ptp"]
        channels = [str(c) for c in z["ch_names"]]
        subject, night = str(z["subject"]), int(z["night"])
        lo, hi = int(z["crop_lo"]), int(z["crop_hi"])
        config_hash = str(z["config_hash"])
    # 학습이 보는 구간에서 재야 한다. 자르기 전에는 낮에 깨어 있던 절반이 섞여
    # 비율이 희석되고, 밤이 통째로 나쁜 녹음이 멀쩡해 보인다.
    #
    # crop_lo/hi 는 조각 번호가 아니라 걸러낸 배열 안의 위치다. features.py 도
    # data[crop_lo:crop_hi] 로 위치를 자른다. epoch_idx 와 비교하면 판독불가로
    # 구멍이 난 녹음 48건만 조용히 다른 조각을 보게 된다.
    keep = np.zeros(len(ptp), bool)
    keep[lo:hi] = True
    if not use_crop:
        keep[:] = True
    # 비면 비율이 nan 이 되고 nan > 임계값 은 False 라 불량이 조용히 통과한다.
    if not keep.any():
        raise ValueError(f"{path.stem}: 잘린 구간에 조각이 없습니다")
    result = assess(ptp[keep], channels, params)
    return {
        "key": path.stem,
        "subject": subject,
        "night": night,
        "n_epochs": int(keep.sum()),
        "cropped": bool(use_crop),
        "epochs_config_hash": config_hash,
        **result,
    }


def scan(
    epochs_dir: str | Path,
    params: dict[str, Any],
    use_crop: bool = True,
    limit: int | None = None,
) -> dict[str, Any]:
    """폴더 전체를 검사하고 보고서 형태로 돌려준다."""
    files = sorted(Path(epochs_dir).glob("*.npz"))[: limit or None]
    if not files:
        raise SystemExit(f"에포크 npz 가 없습니다: {epochs_dir}")
    records = [scan_one(f, params, use_crop) for f in files]
    excluded = [r for r in records if r["exclude"]]
    hashes = {r["epochs_config_hash"] for r in records}
    if len(hashes) > 1:
        raise ValueError(f"에포크 npz 가 서로 다른 설정으로 만들어졌습니다: {sorted(hashes)}")
    return {
        "params": dict(params),
        "epochs_dir": str(epochs_dir),
        "epochs_config_hash": hashes.pop(),
        "cropped": bool(use_crop),
        "n_recordings": len(records),
        "n_excluded": len(excluded),
        "n_epochs_excluded": sum(r["n_epochs"] for r in excluded),
        "excluded": [[r["subject"], r["night"]] for r in excluded],
        "recordings": records,
    }


def write_report(report: dict[str, Any], epochs_dir: str | Path) -> Path:
    path = Path(epochs_dir) / REPORT_NAME
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_excluded(path: str | Path | None) -> set[tuple[str, int]]:
    """보고서에서 배제 대상만 꺼낸다. 경로가 없으면 빈 집합이라 아무것도 안 빠진다."""
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"품질 보고서가 없습니다: {p}\nsleepstage quality 를 먼저 돌리세요.")
    data = json.loads(p.read_text(encoding="utf-8"))
    _check_fresh(data)
    return {(str(s), int(n)) for s, n in data["excluded"]}


def _check_fresh(report: dict[str, Any]) -> None:
    """보고서가 지금 있는 npz 를 보고 만든 것인지 확인한다.

    에포크를 다시 만든 뒤 보고서를 안 고치면 옛 판정으로 학습이 돈다. 실행은 멀쩡히
    끝나고 배제 목록만 틀린다.
    """
    files = sorted(Path(report["epochs_dir"]).glob("*.npz"))
    if not files:
        raise SystemExit(f"보고서가 가리키는 에포크가 없습니다: {report['epochs_dir']}")
    with np.load(files[0], allow_pickle=False) as z:
        now = str(z["config_hash"])
    if now != report["epochs_config_hash"]:
        raise SystemExit(
            f"품질 보고서가 낡았습니다. 에포크 설정 {report['epochs_config_hash']} -> {now}\n"
            "sleepstage quality 를 다시 돌리세요."
        )


def drop_excluded(table, excluded: set[tuple[str, int]]):
    """표에서 배제 대상 녹음의 행을 지운다. 몇 개가 빠졌는지 함께 돌려준다."""
    if not excluded:
        return table, {"n_recordings_excluded": 0, "n_rows_excluded": 0}
    keys = list(zip(table["subject"], table["night"], strict=True))
    keep = np.array([(str(s), int(n)) not in excluded for s, n in keys])
    info = {
        "n_recordings_excluded": len({k for k in keys if (str(k[0]), int(k[1])) in excluded}),
        "n_rows_excluded": int((~keep).sum()),
    }
    return table[keep].reset_index(drop=True), info


def format_report(report: dict[str, Any]) -> str:
    """사람이 읽을 요약. 걸린 것과 아슬아슬한 것을 같이 보여준다."""
    p = report["params"]
    lines = [
        f"녹음 {report['n_recordings']}건  "
        f"진폭 {p['flat_uv']} µV 미만이 {p['max_flat_frac']:.0%} 넘는 채널이 있으면 배제",
        f"배제 {report['n_excluded']}건  조각 {report['n_epochs_excluded']}개",
        "",
    ]
    rows = sorted(report["recordings"], key=lambda r: -max(r["flat_fraction"]))
    for r in rows[:12]:
        mark = "배제" if r["exclude"] else "  "
        frac = "  ".join(f"{v:.3f}" for v in r["flat_fraction"])
        why = ",".join(sorted(r["bad_channels"])) if r["bad_channels"] else ""
        lines.append(f"  {mark} {r['key']:>9}  평탄비율 {frac}  {why}")
    return "\n".join(lines)
