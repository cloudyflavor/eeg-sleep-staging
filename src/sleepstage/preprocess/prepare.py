"""1단계 — EDF 한 쌍을 에포크 npz 하나로.

녹음당 파일 하나를 만든다. 파일명이 곧 피험자 키라서 피험자 단위 교차검증의
분할이 **파일 목록 그 자체**가 된다. 인덱스 범위 계산이 끼어들 자리가 없다.
(단일 HDF5 라면 fold 마다 행 범위를 계산해야 하고, 데이터 누수는 늘 그런 곳에 숨는다.)

녹음별 통계는 평범한 딕셔너리로 다룬다. 그대로 ``manifest.json`` 이 되기 때문에
중간에 클래스를 두면 직렬화 코드만 늘어난다.
"""

import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from sleepstage.config import Config
from sleepstage.io.edf import Recording, find_recordings, read_recording
from sleepstage.preprocess.epochs import DROP, crop_bounds, epoch_signal, quality_metrics, to_codes


def prepare_one(
    rec: Recording,
    cfg: Config,
    out_dir: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """녹음 하나를 처리하고 감사용 통계를 돌려준다."""
    out_path = out_dir / f"{rec.key}.npz"
    if out_path.exists() and not overwrite and _stored_hash(out_path) == cfg.hash:
        return {"key": rec.key, "status": "skipped"}

    epoch_sec = cfg["epoch"]["seconds"]
    sfreq = cfg["epoch"]["sfreq"]
    channels = cfg["dataset"]["channels"]
    names = cfg["labels"]["names"]

    signal, per_epoch = read_recording(rec, channels, sfreq, epoch_sec)
    epochs = epoch_signal(signal, round(epoch_sec * sfreq))

    # EDFX 는 녹음이 끝난 뒤에도 주석이 이어진다 (채점자가 신호 없는 구간을
    # "Sleep stage ?" 로 채워둔 것). 신호가 없으니 애초에 채점 대상이 아니다.
    # **길이를 맞추고 나서** 제거 통계를 낸다 — 아니면 존재한 적 없는 에포크를
    # "제거했다" 고 세게 된다.
    n_annot, n_signal = len(per_epoch), len(epochs)
    n_scored = min(n_annot, n_signal)
    epochs, per_epoch = epochs[:n_scored], per_epoch[:n_scored]

    codes, dropped = to_codes(per_epoch, cfg["labels"]["map"], cfg["labels"]["drop"])
    keep = codes != DROP
    if not keep.any():
        raise ValueError(f"{rec.key}: 남은 에포크가 없습니다")

    # 원본 에포크 번호. 번호가 튀는 자리가 곧 판독불가로 생긴 시간축 구멍이다.
    epoch_idx = np.flatnonzero(keep).astype(np.int32)
    epochs, labels = epochs[keep], codes[keep]
    lo, hi = crop_bounds(labels, names.index("W"), cfg["crop"]["wake_edge_epochs"])

    out_dir.mkdir(parents=True, exist_ok=True)
    save = np.savez_compressed if cfg["output"].get("compress") else np.savez
    save(
        out_path,
        data=epochs,  # (n_ep, n_ch, spe) float32, µV
        labels=labels,  # (n_ep,) int8  0=W 1=LS 2=DS 3=R
        epoch_idx=epoch_idx,
        crop_lo=np.int32(lo),
        crop_hi=np.int32(hi),  # 자르지 않고 경계만
        subject=rec.subject,
        night=np.int8(rec.night),
        sfreq=np.float32(sfreq),
        ch_names=np.array(channels),
        unit="uV",
        config_hash=cfg.hash,
        **quality_metrics(epochs),
    )

    counts = Counter(labels.tolist())
    return {
        "key": rec.key,
        "status": "written",
        "subject": rec.subject,
        "night": rec.night,
        "n_epochs_annotation": n_annot,
        "n_epochs_signal": n_signal,
        "n_epochs_scored": n_scored,
        "n_epochs_kept": int(len(labels)),
        "dropped": dropped,
        "crop": [lo, hi],
        "n_after_crop": hi - lo,
        "class_counts": {names[k]: int(v) for k, v in sorted(counts.items())},
        "bytes": out_path.stat().st_size,
    }


def prepare_all(
    cfg: Config,
    root: str | Path | None = None,
    out_dir: str | Path | None = None,
    limit: int | None = None,
    overwrite: bool = False,
    jobs: int = 1,
) -> dict[str, Any]:
    """전체 녹음을 처리하고 매니페스트를 남긴다."""
    root = Path(root or cfg["dataset"]["root"])
    out_dir = Path(out_dir or cfg["output"]["dir"])

    recs = find_recordings(root)[: limit or None]
    n_sub = len({r.subject for r in recs})
    print(f"녹음 {len(recs)}개  피험자 {n_sub}명  →  {out_dir}", flush=True)

    run = partial(prepare_one, cfg=cfg, out_dir=out_dir, overwrite=overwrite)
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            results = list(ex.map(run, recs))
    else:
        results = [run(r) for r in recs]

    for st in results:
        print(format_audit(st), flush=True)

    manifest = _summarise(results, cfg, root, out_dir)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def format_audit(st: dict[str, Any]) -> str:
    """녹음 하나에 무슨 일이 있었는지 한 덩어리로.

    "왜 이 파일만 에포크가 적지?" 를 나중에 추적할 수 있는 유일한 수단이다.
    DeepSleepNet 은 ``print`` 가 로깅의 전부라 터미널을 닫으면 사라진다.
    """
    if st["status"] == "skipped":
        return f"{st['key']:>9}  건너뜀 (설정 동일)"

    lines = [f"{st['key']:>9}  주석 {st['n_epochs_annotation']} 에포크"]
    extra = st["n_epochs_annotation"] - st["n_epochs_signal"]
    if extra:
        what = "신호 없는 꼬리 주석" if extra > 0 else "주석 없는 꼬리 신호"
        lines.append(f"├ {what} {abs(extra)} 제외 → 채점 대상 {st['n_epochs_scored']}")
    lines += [f"├ {k} 제거: {v}" for k, v in sorted(st["dropped"].items())]
    cls = " / ".join(f"{k} {v}" for k, v in st["class_counts"].items())
    lines.append(f"├ 유지 {st['n_epochs_kept']}  ({cls})")
    lo, hi = st["crop"]
    lines.append(f"└ 절단 경계 [{lo}:{hi}] → {st['n_after_crop']} (적용은 2단계)")
    return "\n            ".join(lines)


def _stored_hash(path: Path) -> str | None:
    """이미 만들어둔 npz 가 어떤 설정으로 만들어졌는지. 못 읽으면 다시 만든다."""
    try:
        with np.load(path, allow_pickle=False) as z:
            return str(z["config_hash"])
    except Exception:  # noqa: BLE001 — 깨진 파일은 그냥 다시 만든다
        return None


def _summarise(
    results: list[dict[str, Any]], cfg: Config, root: Path, out_dir: Path
) -> dict[str, Any]:
    """건너뛴 녹음의 통계는 이전 매니페스트에서 되살린다.

    아니면 재실행할 때마다 집계가 0 이 되어 기록이 통째로 사라진다.
    """
    previous: dict[str, dict[str, Any]] = {}
    path = out_dir / "manifest.json"
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            if old.get("config_hash") == cfg.hash:
                previous = {r["key"]: r for r in old["recordings"] if r["status"] == "written"}
        except (OSError, KeyError, json.JSONDecodeError):
            pass

    n_skipped = sum(st["status"] == "skipped" for st in results)
    records = sorted((previous.get(st["key"], st) for st in results), key=lambda st: st["key"])
    written = [st for st in records if st["status"] == "written"]

    classes: Counter = Counter()
    dropped: Counter = Counter()
    for st in written:
        classes.update(st["class_counts"])
        dropped.update(st["dropped"])

    return {
        "config_hash": cfg.hash,
        "root": str(root),
        "n_recordings": len(records),
        "n_written": len(written),
        "n_skipped": n_skipped,
        "n_subjects": len({st["subject"] for st in written}),
        "n_epochs_kept": sum(st["n_epochs_kept"] for st in written),
        "n_epochs_after_crop": sum(st["n_after_crop"] for st in written),
        "class_counts": dict(classes),
        "dropped": dict(dropped),
        "bytes": sum(st["bytes"] for st in written),
        "recordings": records,
    }
