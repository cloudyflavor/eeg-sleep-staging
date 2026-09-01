"""저장된 녹음을 실제 시간처럼 흘려보내며 판정을 재현한다.

기기 없이도 실시간 동작을 확인할 수 있다. 한꺼번에 계산한 결과와 조각씩 계산한
결과가 같은지, 판정이 얼마나 늦게 나오는지, 조각 하나 처리에 몇 초가 걸리는지를 잰다.
"""

import time
from pathlib import Path
from typing import Any

import numpy as np

from sleepstage.experiment.pipeline.features import WARMUP_EPOCHS
from sleepstage.serving.export import load_artifact
from sleepstage.serving.stream import StreamingStager


def load_epochs(npz_path: str | Path) -> tuple[np.ndarray, np.ndarray, int]:
    """녹음 파일에서 신호와 라벨과 준비 구간 길이를 읽는다."""
    z = np.load(npz_path, allow_pickle=False)
    lo, hi = int(z["crop_lo"]), int(z["crop_hi"])
    start = max(0, lo - WARMUP_EPOCHS)
    return z["data"][start:hi], z["labels"][start:hi], lo - start


def replay(
    artifact_dir: str | Path,
    npz_path: str | Path,
    limit: int | None = None,
) -> dict[str, Any]:
    """녹음 하나를 조각씩 흘려보내며 판정을 모은다."""
    model, config, feature_names = load_artifact(artifact_dir)
    stager = StreamingStager(model, config, feature_names)

    signal, labels, warmup = load_epochs(npz_path)
    if limit:
        signal, labels = signal[: warmup + limit], labels[: warmup + limit]

    results, latencies = [], []
    first_answer_at = None
    for i, epoch in enumerate(signal):
        t0 = time.perf_counter()
        answer = stager.push(epoch)
        latencies.append((time.perf_counter() - t0) * 1000)
        if answer:
            if first_answer_at is None:
                first_answer_at = i
            results.append(answer)
    results.extend(stager.flush())

    scored = [r for r in results if r["epoch_idx"] >= warmup]
    correct = sum(
        1 for r in scored if config["stage_names"].index(r["stage"]) == labels[r["epoch_idx"]]
    )
    lat = np.asarray(latencies)
    return {
        "recording": Path(npz_path).stem,
        "warmup_epochs": warmup,
        "n_epochs_fed": len(signal),
        "n_decisions": len(scored),
        "accuracy": round(correct / max(len(scored), 1), 4),
        "delay_epochs": stager.lag,
        "first_answer_after": first_answer_at,
        "latency_ms": {
            "mean": round(float(lat.mean()), 2),
            "p95": round(float(np.percentile(lat, 95)), 2),
            "max": round(float(lat.max()), 2),
        },
        "results": scored,
    }


def compare_with_batch(
    artifact_dir: str | Path,
    features_parquet: str | Path,
    replay_result: dict[str, Any],
    subject: str,
    night: int,
) -> dict[str, Any]:
    """조각씩 낸 판정과 한꺼번에 낸 판정이 같은지 비교한다.

    같은 모델에 같은 특징 표를 넣어야 의미가 있다. 학습 중에 저장해둔 예측은 데이터를
    나눠 학습한 다른 모델이 낸 것이라 비교 대상이 아니다.

    두 결과의 조각 번호 기준도 다르다. 한꺼번에 계산한 쪽은 준비 구간이 이미 잘려
    있고, 조각씩 처리한 쪽은 준비 구간부터 센다. 그만큼 밀어서 맞춘다.
    """
    import pandas as pd

    model, config, names = load_artifact(artifact_dir)
    stage_names = config["stage_names"]

    table = pd.read_parquet(features_parquet)
    table = table[(table["subject"] == subject) & (table["night"] == night)]
    table = table.sort_values("epoch_idx").reset_index(drop=True)
    batch_pred = np.asarray(model.predict(table[names].to_numpy(dtype=np.float32))).ravel()

    warmup = replay_result["warmup_epochs"]
    stream = {
        r["epoch_idx"] - warmup: stage_names.index(r["stage"]) for r in replay_result["results"]
    }

    matched, mismatched, examples = 0, 0, []
    for position, expected in enumerate(batch_pred.astype(int)):
        got = stream.get(position)
        if got is None:
            continue
        if got == expected:
            matched += 1
        else:
            mismatched += 1
            if len(examples) < 5:
                examples.append(
                    {
                        "position": position,
                        "batch": stage_names[expected],
                        "stream": stage_names[got],
                    }
                )

    total = matched + mismatched
    return {
        "compared": total,
        "matched": matched,
        "mismatched": mismatched,
        "agreement": round(matched / max(total, 1), 4),
        "examples": examples,
    }
