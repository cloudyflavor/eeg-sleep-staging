"""원본 파일을 30초 조각 단위의 배열 파일로 바꾼다.

녹음 하나가 파일 하나가 된다. 파일 이름이 곧 피험자 번호라서, 나중에 사람 단위로
데이터를 나눌 때 파일 목록만 나누면 된다.

여기서는 정보를 잃는 처리를 하지 않는다. 필터나 정규화처럼 실험마다 바꿔볼
처리는 다음 단계로 미룬다. 이 파일을 다시 만드는 데 시간이 오래 걸리기 때문이다.
"""

import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from sleepstage.config import Config
from sleepstage.core.edf import Recording, find_recordings, read_recording
from sleepstage.core.epochs import DROP, crop_bounds, epoch_signal, quality_metrics, to_codes


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

    # 판독 파일이 신호보다 길게 이어지는 경우가 있다. 길이를 맞춘 뒤에 제거
    # 개수를 세야 실제로 없던 구간을 지웠다고 기록하지 않는다.
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
    root, out_dir = _load(cfg, root, out_dir)

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
    """녹음 하나가 어떻게 처리됐는지 사람이 읽을 형태로 만든다.

    나중에 특정 파일만 에포크 수가 적을 때 원인을 찾을 수 있다.
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
    lines.append(f"└ 절단 경계 [{lo}:{hi}] → {st['n_after_crop']} (적용은 특징 추출에서)")
    return "\n            ".join(lines)


def _load(cfg: Config, root, out_dir) -> tuple[Path, Path]:
    """설정의 경로와 CLI 덮어쓰기를 합친다."""
    return Path(root or cfg["dataset"]["root"]), Path(out_dir or cfg["output"]["dir"])


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
    """전체 통계를 모은다.

    이미 처리해 건너뛴 녹음은 이전 기록에서 값을 가져온다. 그러지 않으면 다시
    실행할 때마다 집계가 0 이 되어 기록이 사라진다.
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
