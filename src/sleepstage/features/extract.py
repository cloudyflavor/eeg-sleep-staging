"""2단계 — 에포크 npz → 특징 parquet. 근거: docs/04-stage2.md.

처리 순서 고정: 필터 → 특징 → 정규화 → 문맥 → 절단. 전부 warmup 포함 창에서 계산한다.
"""

import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sleepstage.config import Config, config_hash
from sleepstage.features import context, filters, normalize, spectral, temporal

STAGE_NAMES = ("W", "LS", "DS", "R")

#: 산출물의 정체성을 정하는 설정. 경로는 데이터를 바꾸지 않으므로 제외.
SETTING_KEYS = ("filter", "normalize", "context", "distance_features")

#: 절단 구간 앞의 예열 길이(에포크). 누적 통계·인과 필터가 자리 잡는 구간이며
#: "잠들기 40분 전 착용" 을 흉내 낸다. 학습에는 넣지 않는다.
WARMUP_EPOCHS = 20


def settings_hash(cfg: Config) -> str:
    """이 설정이 만들 특징 표의 식별자."""
    return config_hash({k: cfg["features"][k] for k in SETTING_KEYS})


def output_path(cfg: Config) -> Path:
    """<out_dir>/<설정해시>.parquet — 설정과 산출물이 1:1."""
    return Path(cfg["features"]["out_dir"]) / f"{settings_hash(cfg)}.parquet"


def epoch_features(
    epochs: np.ndarray,
    sfreq: float,
    channels: list[str],
    with_distance: bool,
) -> pd.DataFrame:
    """(n_ep, n_ch, n_samp) → 열이 ``채널__특징`` 인 표. 채널별 분석·선택이 열 이름으로 된다."""
    frames = []
    for i, ch in enumerate(channels):
        sig = epochs[:, i, :]
        freqs, psd = spectral.welch_psd(sig, sfreq)

        feats: dict[str, np.ndarray] = {}
        feats.update(spectral.band_powers(freqs, psd))
        feats.update(temporal.basic_stats(sig))
        feats.update(temporal.hjorth(sig))
        feats.update(_complexity(sig))
        if with_distance:
            feats.update(temporal.distance_energy(sig))

        frames.append(pd.DataFrame(feats).add_prefix(f"{ch}__"))

    out = pd.concat(frames, axis=1)
    if len(channels) == 2:
        out = pd.concat([out, _channel_ratios(out, channels)], axis=1)
    return out


def _complexity(sig: np.ndarray) -> dict[str, np.ndarray]:
    import antropy as ant

    return {
        "perm": np.apply_along_axis(ant.perm_entropy, 1, sig, normalize=True).astype("float32"),
        "higuchi": np.apply_along_axis(ant.higuchi_fd, 1, sig).astype("float32"),
        "petrosian": ant.petrosian_fd(sig, axis=1).astype("float32"),
    }


def _channel_ratios(feats: pd.DataFrame, channels: list[str]) -> pd.DataFrame:
    """두 채널의 비율 — 전극 위치 정보."""
    front, back = channels
    return pd.DataFrame(
        {
            "ratio__alpha_back_front": feats[f"{back}__alpha"] / feats[f"{front}__alpha"],
            "ratio__delta_front_back": feats[f"{front}__fdelta"] / feats[f"{back}__fdelta"],
        }
    ).astype("float32")


def build_recording(path: Path, cfg: Config) -> tuple[pd.DataFrame, dict[str, int], dict]:
    """npz 하나 → (표, lookahead 장부, 감사 기록)."""
    z = np.load(path, allow_pickle=False)
    sfreq = float(z["sfreq"])
    channels = [str(c) for c in z["ch_names"]]

    # 계산 창 = 절단 구간 + 앞쪽 warmup. 나머지는 읽지 않는다.
    crop_lo, crop_hi = int(z["crop_lo"]), int(z["crop_hi"])
    start = max(0, crop_lo - WARMUP_EPOCHS)
    keep_from = crop_lo - start

    epochs = z["data"][start:crop_hi]
    labels = z["labels"][start:crop_hi]
    epoch_idx = z["epoch_idx"][start:crop_hi]
    epoch_sec = epochs.shape[-1] / sfreq

    filter_mode = cfg["features"]["filter"]
    signal = filters.apply_filter(
        epochs.transpose(1, 0, 2).reshape(len(channels), -1), sfreq, filter_mode
    )
    epochs = signal.reshape(len(channels), len(labels), -1).transpose(1, 0, 2)

    raw = epoch_features(epochs, sfreq, channels, cfg["features"]["distance_features"])
    lookahead = dict.fromkeys(raw.columns, 0)
    blocks = [raw]

    norm_mode = cfg["features"]["normalize"]
    audit: dict[str, Any] = {"filter": filter_mode, "normalize": norm_mode}
    if norm_mode != "none":
        if norm_mode not in normalize.NORMALIZERS:
            raise ValueError(
                f"모르는 정규화 방식입니다: {norm_mode!r} (none/{'/'.join(normalize.NORMALIZERS)})"
            )
        fn, cost = normalize.NORMALIZERS[norm_mode]
        normed = fn(raw).add_suffix("_norm")
        audit["missing"] = normalize.missing_report(normed, keep_from)
        lookahead.update(dict.fromkeys(normed.columns, len(raw) if cost is None else cost))
        blocks.append(normed)

    base = pd.concat(blocks, axis=1)

    ctx_mode = cfg["features"]["context"]
    audit["context"] = ctx_mode
    if ctx_mode not in context.CONTEXTS:
        choices = "/".join(context.CONTEXTS)
        raise ValueError(f"모르는 시간 문맥 방식입니다: {ctx_mode!r} ({choices})")
    builder = context.CONTEXTS[ctx_mode]
    if builder is not None:
        extra, la = builder(base, epoch_idx)
        blocks.append(extra)
        lookahead.update(la)

    hours = context.elapsed_hours(epoch_idx - epoch_idx[0], epoch_sec)
    blocks.append(hours.to_frame())
    lookahead[hours.name] = 0

    table = normalize.as_float32(pd.concat(blocks, axis=1))

    # warmup 은 통계를 세우는 데만 쓰고 학습에는 넣지 않는다.
    table = table.iloc[keep_from:].reset_index(drop=True)

    table.insert(0, "stage", labels[keep_from:])
    table.insert(0, "epoch_idx", epoch_idx[keep_from:])
    table.insert(0, "night", np.int8(z["night"]))
    table.insert(0, "subject", str(z["subject"]))

    audit.update(
        {
            "key": path.stem,
            "n_epochs": len(table),
            "n_features": len(lookahead),
            "crop": [crop_lo, crop_hi],
            "warmup_epochs": keep_from,
            "class_counts": {
                STAGE_NAMES[k]: int(v)
                for k, v in zip(*np.unique(table["stage"], return_counts=True), strict=True)
            },
        }
    )
    return table, lookahead, audit


def extract_all(
    cfg: Config,
    epochs_dir: str | Path | None = None,
    out_path: str | Path | None = None,
    limit: int | None = None,
    jobs: int = 1,
    overwrite: bool = False,
) -> dict[str, Any]:
    """전체 녹음 → parquet 하나. 같은 설정의 완성 산출물이 있으면 건너뛴다."""
    epochs_dir = Path(epochs_dir or cfg["features"]["epochs_dir"])
    out_path = Path(out_path) if out_path else output_path(cfg)
    manifest_path = out_path.with_suffix(".manifest.json")

    files = sorted(epochs_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"{epochs_dir} 에 *.npz 가 없습니다")
    n_total = len(files)
    files = files[: limit or None]

    if not overwrite and _is_complete(manifest_path, settings_hash(cfg), n_total):
        print(f"건너뜀 — 같은 설정의 산출물이 이미 있습니다: {out_path}", flush=True)
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    print(f"녹음 {len(files)}개  →  {out_path}", flush=True)

    run = partial(build_recording, cfg=cfg)
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            results = list(ex.map(run, files))
    else:
        results = [run(f) for f in files]

    tables, audits, lookahead = [], [], None
    for table, la, audit in results:
        if lookahead is not None and la.keys() != lookahead.keys():
            # 장부가 마지막 녹음 것으로 조용히 어긋나는 것 방지
            raise ValueError(f"{audit['key']} 의 열 구성이 다릅니다")
        tables.append(table)
        audits.append(audit)
        lookahead = la
        n_feat = audit["n_features"]
        print(f"  {audit['key']:>9}  {audit['n_epochs']:>5} 에포크  특징 {n_feat}", flush=True)

    full = pd.concat(tables, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(out_path, index=False)

    manifest = {
        "settings_hash": settings_hash(cfg),
        "config_hash": cfg.hash,
        "n_recordings": len(files),
        "n_subjects": int(full["subject"].nunique()),
        "n_epochs": len(full),
        "n_features": len(lookahead),
        "class_counts": {STAGE_NAMES[k]: int(v) for k, v in full["stage"].value_counts().items()},
        "settings": {k: cfg["features"][k] for k in SETTING_KEYS},
        "lookahead_epochs": lookahead,
        "bytes": out_path.stat().st_size,
        "recordings": audits,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _is_complete(manifest_path: Path, want_hash: str, n_total: int) -> bool:
    """전체 실행 산출물인지. --limit 부분 실행은 완성으로 치지 않는다."""
    if not manifest_path.exists():
        return False
    try:
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return m.get("settings_hash") == want_hash and m.get("n_recordings") == n_total
