"""2단계 — 에포크 npz 를 특징 parquet 으로. 설계 근거는 `docs/04-stage2.md`.

처리 순서가 설계의 핵심이다::

    필터 → 특징 → 정규화 → 시간 문맥 → **마지막에 절단**
    └───────── 전부 절단 전 전체 배열에서 ─────────┘

절단을 마지막에 하는 건 실제 기기와 맞추기 위해서다. 기기는 착용한 순간부터 쌓지
잠든 순간부터 쌓지 않는다. 순서를 바꾸면 조용히 틀린다.
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

#: 클래스 이름. 라벨 정수의 순서와 같다.
STAGE_NAMES = ("W", "LS", "DS", "R")

#: 산출물의 정체성을 정하는 설정. **경로는 넣지 않는다** — 같은 처리를 다른 폴더에
#: 내보냈다고 해서 다른 데이터가 되는 게 아니다.
SETTING_KEYS = ("filter", "normalize", "context", "distance_features")


def settings_hash(cfg: Config) -> str:
    """이 설정이 만들 특징 표의 식별자."""
    return config_hash({k: cfg["features"][k] for k in SETTING_KEYS})


def output_path(cfg: Config) -> Path:
    """``<out_dir>/<설정해시>.parquet``. 설정과 산출물이 1:1 로 묶인다."""
    return Path(cfg["features"]["out_dir"]) / f"{settings_hash(cfg)}.parquet"


def epoch_features(
    epochs: np.ndarray,
    sfreq: float,
    channels: list[str],
    with_distance: bool,
) -> pd.DataFrame:
    """``(n_epochs, n_channels, n_samples)`` → 열이 ``채널__특징`` 인 표.

    이름에 채널을 붙여두면 채널별·갈래별 중요도 분석과 1채널 vs 2채널 비교가
    열 선택만으로 끝난다.
    """
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
    """알파는 후두엽, 델타는 전두엽에서 강하므로 두 채널의 비율이 위치 정보를 담는다."""
    front, back = channels
    return pd.DataFrame(
        {
            "ratio__alpha_back_front": feats[f"{back}__alpha"] / feats[f"{front}__alpha"],
            "ratio__delta_front_back": feats[f"{front}__fdelta"] / feats[f"{back}__fdelta"],
        }
    ).astype("float32")


def build_recording(path: Path, cfg: Config) -> tuple[pd.DataFrame, dict[str, int], dict]:
    """npz 하나를 특징 표로. ``(표, lookahead 장부, 감사 통계)``.

    lookahead 장부는 열마다 필요한 미래 에포크 수를 적는다. 나중에 지연 예산을 넣으면
    자동으로 걸러진다.
    """
    z = np.load(path, allow_pickle=False)
    epochs = z["data"]
    labels = z["labels"]
    epoch_idx = z["epoch_idx"]
    sfreq = float(z["sfreq"])
    channels = [str(c) for c in z["ch_names"]]
    epoch_sec = epochs.shape[-1] / sfreq

    filter_mode = cfg["features"]["filter"]
    signal = filters.apply_filter(
        epochs.transpose(1, 0, 2).reshape(len(channels), -1), sfreq, filter_mode
    )
    epochs = signal.reshape(len(channels), len(labels), -1).transpose(1, 0, 2)

    raw = epoch_features(epochs, sfreq, channels, cfg["features"]["distance_features"])
    lookahead = dict.fromkeys(raw.columns, 0)
    blocks = [raw]

    crop = (int(z["crop_lo"]), int(z["crop_hi"]))

    norm_mode = cfg["features"]["normalize"]
    audit: dict[str, Any] = {"filter": filter_mode, "normalize": norm_mode}
    if norm_mode != "none":
        fn = normalize.expanding_robust if norm_mode == "expanding" else normalize.posthoc_robust
        normed = fn(raw).add_suffix("_norm")
        audit["missing"] = normalize.missing_report(normed, crop)
        # 사후 정규화는 밤 전체를 봐야 한다 → 실시간 불가. 장부에 크게 적어 걸러지게 한다.
        cost = 0 if norm_mode == "expanding" else len(raw)
        lookahead.update(dict.fromkeys(normed.columns, cost))
        blocks.append(normed)

    base = pd.concat(blocks, axis=1)

    ctx_mode = cfg["features"]["context"]
    audit["context"] = ctx_mode
    if ctx_mode == "smooth":
        extra, la = context.smoothed(base)
        blocks.append(extra)
        lookahead.update(la)
    elif ctx_mode == "shift":
        extra, la = context.shifted(base, epoch_idx)
        blocks.append(extra)
        lookahead.update(la)
    elif ctx_mode != "none":
        raise ValueError(f"모르는 시간 문맥 방식입니다: {ctx_mode!r} (none / smooth / shift)")

    hours = context.elapsed_hours(epoch_idx, epoch_sec)
    blocks.append(hours.to_frame())
    lookahead[hours.name] = 0

    table = normalize.as_float32(pd.concat(blocks, axis=1))

    # ── 여기서 처음으로 자른다 ────────────────────────────────────────────
    # 앞뒤 30분만 남긴다. 이 데이터는 집에서 24시간 연속 녹음한 것이라 절반 이상이
    # 낮에 깨어 있는 구간인데, 실제 기기는 자기 전에 쓰고 아침에 벗는다.
    lo, hi = crop
    table = table.iloc[lo:hi].reset_index(drop=True)

    table.insert(0, "stage", labels[lo:hi])
    table.insert(0, "epoch_idx", epoch_idx[lo:hi])
    table.insert(0, "night", np.int8(z["night"]))
    table.insert(0, "subject", str(z["subject"]))

    audit.update(
        {
            "key": path.stem,
            "n_epochs": len(table),
            "n_features": len(lookahead),
            "crop": [lo, hi],
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
    """전체 녹음의 특징을 parquet 하나로. 피험자 분할은 ``subject`` 열로 하므로 나눌 필요가 없다."""
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
            # 채널 수가 다른 녹음이 섞이면 장부가 조용히 어긋난다
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
    """같은 설정으로 **전체**를 이미 돌렸는지. 부분 실행(``--limit``)은 완성으로 치지 않는다."""
    if not manifest_path.exists():
        return False
    try:
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return m.get("settings_hash") == want_hash and m.get("n_recordings") == n_total
