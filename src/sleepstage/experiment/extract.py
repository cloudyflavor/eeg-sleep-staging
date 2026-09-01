"""30초 조각 신호를 모델이 먹을 숫자 표로 바꾼다.

처리 순서가 정해져 있다. 필터를 걸고, 특징을 뽑고, 사람별 기준으로 맞추고,
주변 흐름을 붙인 다음, 마지막에 필요 없는 구간을 잘라낸다.

자르기를 마지막에 하는 이유는 실제 기기와 조건을 맞추기 위해서다. 기기는 사용자가
착용한 순간부터 데이터를 쌓기 시작하지, 잠든 순간부터 쌓지 않는다.
"""

import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sleepstage.config import Config, config_hash
from sleepstage.core import context, events, filters, normalize, spectral, temporal
from sleepstage.fingerprint import check_or_explain, code_fingerprint

STAGE_NAMES = ("W", "LS", "DS", "R")

#: 결과 파일의 내용을 결정하는 설정들. 저장 경로는 내용을 바꾸지 않으므로 뺀다.
SETTING_KEYS = (
    "filter",
    "normalize",
    "context",
    "distance_features",
    "welch_average",
    "window_extremes",
    "spindle_events",
    "channel_correlation",
)

#: 본 구간 앞에 두는 준비 구간의 길이. 사람별 기준값을 잡고 필터가 안정되는 데
#: 필요하다. 실제로 잠들기 조금 전에 기기를 착용하는 상황과 같다.
#: 이 구간은 계산에만 쓰고 학습 데이터에는 넣지 않는다.
WARMUP_EPOCHS = 20


def settings_hash(cfg: Config) -> str:
    """설정 내용을 짧은 문자열로 요약한다. 결과 파일 이름이 된다."""
    return config_hash({k: cfg["features"][k] for k in SETTING_KEYS})


def output_path(cfg: Config) -> Path:
    """결과 파일 경로. 설정마다 다른 이름이 되므로 덮어쓸 일이 없다."""
    return Path(cfg["features"]["out_dir"]) / f"{settings_hash(cfg)}.parquet"


def epoch_features(
    epochs: np.ndarray,
    sfreq: float,
    channels: list[str],
    with_distance: bool,
    welch_average: str = "median",
    window_extremes: bool = False,
    spindle_events: bool = False,
    channel_correlation: bool = False,
) -> pd.DataFrame:
    """에포크마다 특징을 계산해 표로 만든다.

    열 이름에 채널을 붙여둔다. 나중에 채널별로 나눠 보거나 한쪽 채널만 쓰고 싶을 때
    이름만으로 골라낼 수 있다.
    """
    frames = []
    for i, ch in enumerate(channels):
        sig = epochs[:, i, :]
        freqs, psd = spectral.welch_psd(sig, sfreq, average=welch_average)

        feats: dict[str, np.ndarray] = {}
        feats.update(spectral.band_powers(freqs, psd))
        feats.update(temporal.basic_stats(sig))
        feats.update(temporal.hjorth(sig))
        feats.update(_complexity(sig))
        if with_distance:
            feats.update(temporal.distance_energy(sig))
        if window_extremes:
            feats.update(spectral.band_extremes(sig, sfreq))
        if spindle_events:
            counted = events.spindles_for_recording(sig[:, None, :], sfreq)[:, 0, :]
            feats["spindle_n"] = counted[:, 0]
            feats["spindle_sec"] = counted[:, 1]

        frames.append(pd.DataFrame(feats).add_prefix(f"{ch}__"))

    out = pd.concat(frames, axis=1)
    if len(channels) == 2:
        out = pd.concat([out, _channel_ratios(out, channels)], axis=1)
        if channel_correlation:
            bands = {"eog": spectral.EOG_BAND, **spectral.BANDS}
            corr = temporal.band_correlation(epochs[:, 0], epochs[:, 1], sfreq, bands)
            out = pd.concat([out, pd.DataFrame(corr).add_prefix("ratio__")], axis=1)
    return out


def _complexity(sig: np.ndarray) -> dict[str, np.ndarray]:
    import antropy as ant

    return {
        "perm": np.apply_along_axis(ant.perm_entropy, 1, sig, normalize=True).astype("float32"),
        "higuchi": np.apply_along_axis(ant.higuchi_fd, 1, sig).astype("float32"),
        "petrosian": ant.petrosian_fd(sig, axis=1).astype("float32"),
    }


def _channel_ratios(feats: pd.DataFrame, channels: list[str]) -> pd.DataFrame:
    """두 채널의 세기 비율. 뇌파는 부위마다 다르게 나타나므로 위치 정보가 된다."""
    front, back = channels
    return pd.DataFrame(
        {
            "ratio__alpha_back_front": feats[f"{back}__alpha"] / feats[f"{front}__alpha"],
            "ratio__delta_front_back": feats[f"{front}__fdelta"] / feats[f"{back}__fdelta"],
        }
    ).astype("float32")


def build_recording(path: Path, cfg: Config) -> tuple[pd.DataFrame, dict[str, int], dict]:
    """녹음 하나를 특징 표로 바꾼다.

    표와 함께, 각 열이 미래를 몇 에포크나 봐야 하는지 기록한 목록과 처리 내역을
    돌려준다.
    """
    z = np.load(path, allow_pickle=False)
    sfreq = float(z["sfreq"])
    channels = [str(c) for c in z["ch_names"]]

    # 본 구간과 준비 구간만 읽는다. 나머지는 쓰지 않으므로 메모리에 올리지 않는다.
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

    f = cfg["features"]
    raw = epoch_features(
        epochs,
        sfreq,
        channels,
        f["distance_features"],
        welch_average=f["welch_average"],
        window_extremes=f["window_extremes"],
        spindle_events=f["spindle_events"],
        channel_correlation=f["channel_correlation"],
    )
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

    # 준비 구간을 잘라낸다. 기준값을 잡는 데만 썼고 학습에는 넣지 않는다.
    table = table.iloc[keep_from:].reset_index(drop=True)

    table.insert(0, "stage", labels[keep_from:])
    table.insert(0, "epoch_idx", epoch_idx[keep_from:])
    table.insert(0, "night", np.int8(z["night"]))
    table.insert(0, "subject", str(z["subject"]))

    audit.update(
        {
            "key": path.stem,
            "sfreq": sfreq,
            "epoch_seconds": epoch_sec,
            "channels": channels,
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
    """모든 녹음을 처리해 하나의 표로 합친다.

    같은 설정으로 만든 완성된 파일이 이미 있으면 다시 계산하지 않는다.
    """
    epochs_dir = Path(epochs_dir or cfg["features"]["epochs_dir"])
    out_path = Path(out_path) if out_path else output_path(cfg)
    manifest_path = out_path.with_suffix(".manifest.json")

    files = sorted(epochs_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"{epochs_dir} 에 *.npz 가 없습니다")
    n_total = len(files)
    files = files[: limit or None]

    if not overwrite and _is_complete(manifest_path, settings_hash(cfg), n_total):
        settings = {k: cfg["features"][k] for k in SETTING_KEYS}
        check_or_explain(_recorded_code(manifest_path), code_fingerprint(settings), manifest_path)
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
            # 녹음마다 열 구성이 다르면 미래 참조 기록이 마지막 것으로 덮여
            # 잘못된 값이 남는다
            raise ValueError(f"{audit['key']} 의 열 구성이 다릅니다")
        tables.append(table)
        audits.append(audit)
        lookahead = la
        n_feat = audit["n_features"]
        print(f"  {audit['key']:>9}  {audit['n_epochs']:>5} 에포크  특징 {n_feat}", flush=True)

    full = pd.concat(tables, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(out_path, index=False)

    # 신호 조건은 녹음마다 같아야 하므로 첫 녹음 것을 표 전체의 것으로 적는다.
    # 배포 파일이 이 값을 읽어 간다. 여기 없으면 배포 쪽에 손으로 적게 된다.
    signal = {k: audits[0][k] for k in ("sfreq", "epoch_seconds", "channels")}
    if any({k: a[k] for k in signal} != signal for a in audits):
        raise ValueError("녹음마다 샘플링레이트나 채널이 다릅니다")

    manifest = {
        "settings_hash": settings_hash(cfg),
        "code_hash": code_fingerprint({k: cfg["features"][k] for k in SETTING_KEYS}),
        "config_hash": cfg.hash,
        **signal,
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


def _recorded_code(manifest_path: Path) -> str | None:
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8")).get("code_hash")
    except (OSError, json.JSONDecodeError):
        return None


def _is_complete(manifest_path: Path, want_hash: str, n_total: int) -> bool:
    """이미 만들어진 결과가 전체 데이터로 만든 것인지 확인한다.

    일부만 처리한 결과는 완성으로 치지 않는다.
    """
    if not manifest_path.exists():
        return False
    try:
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return m.get("settings_hash") == want_hash and m.get("n_recordings") == n_total
