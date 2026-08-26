"""2단계 — 에포크 npz 를 특징 parquet 으로.

**비교 실험의 축이 전부 여기 있다.** 1단계가 정보를 잃지 않는 처리만 했기 때문에,
npz 를 다시 만들지 않고 설정만 바꿔 몇 번이고 다시 돌릴 수 있다.

처리 순서가 중요하다::

    필터  →  특징  →  정규화  →  시간 문맥  →  **마지막에 절단**
    └────────── 전부 절단 전 전체 배열에서 ──────────┘

절단을 마지막에 하는 이유는 **실제 기기와 맞추기 위해서다.** 기기는 착용한 순간부터 데이터를
쌓지, 잠든 순간부터 쌓지 않는다. 절단 후에 누적 정규화를 하면 절단 구간의 첫 에포크가
"과거가 없는" 것처럼 취급되는데, 그 시점에 기기는 이미 몇 시간치를 갖고 있다.
(실측: 녹음 153개의 ``crop_lo`` 중앙값 839 = 약 7시간)

같은 이유로 인과 필터도 절단 전에 걸어야 한다. 절단 후에 걸면 첫 몇 초가 과거 없이
시작해 경계 왜곡이 생긴다.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sleepstage.config import Config
from sleepstage.features import context, filters, normalize, spectral, temporal

#: 클래스 이름. 라벨 정수의 순서와 같다.
STAGE_NAMES = ("W", "LS", "DS", "R")


def epoch_features(
    epochs: np.ndarray,
    sfreq: float,
    channels: list[str],
    with_distance: bool,
) -> pd.DataFrame:
    """에포크별 특징. ``(n_epochs, n_channels, n_samples)`` → 열이 ``채널__특징`` 인 표.

    **채널마다 따로 계산하고 이름에 채널을 붙인다.** 이름을 파싱하면 나중에 채널별·갈래별
    중요도를 묶어 분석할 수 있다 (sleep-linear 의 패턴).
    1채널 vs 2채널 비교도 열 선택만으로 끝난다.
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
    """엔트로피·프랙탈. 신호가 얼마나 예측 불가능한가.

    실측(SC400N1)상 **단변량으로는 어떤 대역파워보다 잘 가른다** —
    4클래스 분산분석 F값이 perm 920, petrosian 905, higuchi 735 인데 beta 가 198 이다.
    "비싼 특징이 값어치를 하나" 라는 의심이 적어도 단변량 기준으로는 틀렸다.
    (단서: 다른 특징과의 중복은 반영되지 않은 값이다. 최종 판단은 ablation 으로.)
    """
    import antropy as ant

    return {
        "perm": np.apply_along_axis(ant.perm_entropy, 1, sig, normalize=True).astype("float32"),
        "higuchi": np.apply_along_axis(ant.higuchi_fd, 1, sig).astype("float32"),
        "petrosian": ant.petrosian_fd(sig, axis=1).astype("float32"),
    }


def _channel_ratios(feats: pd.DataFrame, channels: list[str]) -> pd.DataFrame:
    """두 채널 사이의 비율. 전극 위치 정보를 담는다.

    알파파는 후두엽(Pz-Oz)에서, 델타파는 전두엽(Fpz-Cz)에서 강하다.
    비율이 "이 알파가 진짜 알파인가" 를 검증한다.

    **레퍼런스에 없다** — YASA 도 sleep-linear 도 단일 채널 기준이다.
    우리가 2채널을 쓰기로 했으니 시도해볼 지점이고, 안 되면 안 된다는 것도 결과다.
    """
    front, back = channels
    return pd.DataFrame(
        {
            "ratio__alpha_back_front": feats[f"{back}__alpha"] / feats[f"{front}__alpha"],
            "ratio__delta_front_back": feats[f"{front}__fdelta"] / feats[f"{back}__fdelta"],
        }
    ).astype("float32")


def build_recording(path: Path, cfg: Config) -> tuple[pd.DataFrame, dict[str, int], dict]:
    """npz 하나를 특징 표로. ``(표, lookahead 장부, 감사 통계)``.

    lookahead 장부는 열마다 "필요한 미래 에포크 수" 를 적는다. 스트리밍을 만드는 게 아니라
    **장부를 적어두는 것**이다. 나중에 예산을 넣으면 자동으로 걸러진다.
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

    crop = (int(z["crop_lo"]), int(z["crop_hi"])) if cfg["features"]["crop"] else (0, len(raw))

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
) -> dict[str, Any]:
    """전체 녹음의 특징을 parquet 하나로 모은다.

    **parquet 하나에 모으는 이유**: 학습은 전체를 한 번에 읽는다. 피험자 단위 분할은
    ``subject`` 열로 하므로 파일을 나눌 필요가 없다. 1단계에서 녹음당 파일로 나눈 것과
    목적이 다르다 — 거기서는 병렬 생성과 부분 재빌드가 이유였다.
    """
    epochs_dir = Path(epochs_dir or cfg["features"]["epochs_dir"])
    out_path = Path(out_path or cfg["features"]["out"])

    files = sorted(epochs_dir.glob("*.npz"))[: limit or None]
    if not files:
        raise FileNotFoundError(f"{epochs_dir} 에 *.npz 가 없습니다")
    print(f"녹음 {len(files)}개  →  {out_path}")

    tables, audits, lookahead = [], [], {}
    for path in files:
        table, la, audit = build_recording(path, cfg)
        tables.append(table)
        audits.append(audit)
        lookahead = la  # 모든 녹음이 같은 열 구성이다
        print(f"  {audit['key']:>9}  {audit['n_epochs']:>5} 에포크  특징 {audit['n_features']}")

    full = pd.concat(tables, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(out_path, index=False)

    manifest = {
        "config_hash": cfg.hash,
        "n_recordings": len(files),
        "n_subjects": int(full["subject"].nunique()),
        "n_epochs": len(full),
        "n_features": len(lookahead),
        "class_counts": {STAGE_NAMES[k]: int(v) for k, v in full["stage"].value_counts().items()},
        "settings": {k: cfg["features"][k] for k in ("filter", "normalize", "context", "crop")},
        "lookahead_epochs": lookahead,
        "bytes": out_path.stat().st_size,
        "recordings": audits,
    }
    out_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
