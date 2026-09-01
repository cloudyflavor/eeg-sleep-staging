"""신경망을 학습시킨다.

특징을 쓰는 모델과 비교할 수 있도록 나누는 기준, 잘라내는 범위, 필터, 진폭 정규화,
단계별 가중치, 성능 지표를 전부 똑같이 맞춘다. 다른 점은 입력 형태뿐이다.

학습 회차는 평가 데이터가 아니라 학습 쪽에서 떼어낸 검증용 피험자로 고른다.
평가 점수로 회차를 고르면 성능이 부풀려진다.
"""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from sleepstage.config import Config, config_hash
from sleepstage.experiment import run_paths
from sleepstage.experiment.baselines.deep.data import build_epoch_split, build_sequence_split
from sleepstage.experiment.baselines.deep.encoder import SleepCNN
from sleepstage.experiment.baselines.deep.sequence import SleepCNNLSTM
from sleepstage.experiment.modeling.cross_validate import _metrics
from sleepstage.experiment.pipeline.folds import STAGE_NAMES, load_folds

#: 10묶음이 다 차면 평균을 낸다
N_FOLDS = 10


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _class_weights(labels: np.ndarray, mode: str, device) -> torch.Tensor | None:
    """드문 수면 단계에 더 큰 가중치를 준다. 다른 모델과 같은 방식이다."""
    if mode != "balanced":
        return None
    counts = np.bincount(labels, minlength=len(STAGE_NAMES))
    w = len(labels) / (len(STAGE_NAMES) * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32, device=device)


#: 결과를 정하는 설정들. 부스팅 쪽 RESULT_KEYS 와 같은 이유로 따로 적는다.
#: device 나 workers 처럼 값을 안 바꾸는 것을 넣으면, 코어 수만 건드려도
#: 같은 실험이 다른 실행으로 갈라져 이미 있는 결과를 못 찾는다.
RESULT_KEYS = (
    "splits",
    "seed",
    "epochs",
    "batch_size",
    "lr",
    "class_weight",
    "filter",
    "normalize",
    "val_ratio",
    "sequence_length",
    "sequence_stride",
)


def run_id(p: dict) -> str:
    return config_hash({k: p[k] for k in RESULT_KEYS})


def run_dir(p: dict) -> Path:
    """무엇을 돌렸는지 폴더 계층만 보고 알 수 있게 경로를 만든다."""
    window = p["sequence_length"] if p["sequence_length"] > 0 else 1
    return run_paths.run_dir(
        p["out_dir"],
        "lstm" if window > 1 else "cnn",
        run_paths.preprocess_slug(p, window),
        run_paths.options_slug(p["class_weight"]),
        run_id(p),
    )


@torch.no_grad()
def evaluate(model, loader, device, sequence: bool) -> dict[str, float]:
    """채점 대상으로 표시된 자리만 모아 지표를 낸다."""
    model.eval()
    trues, preds = [], []
    for batch in loader:
        if sequence:
            x, y, mask = batch
            logits = model(x.to(device, non_blocking=True)).argmax(-1).cpu()
            preds.append(logits[mask].numpy())
            trues.append(y[mask].numpy())
        else:
            x, y = batch
            preds.append(model(x.to(device, non_blocking=True)).argmax(1).cpu().numpy())
            trues.append(y.numpy())
    return _metrics(np.concatenate(trues), np.concatenate(preds))


def train_fold(cfg: Config, fold_idx: int = 0, limit: int | None = None) -> dict[str, Any]:
    p = cfg["deep"]
    device = _device(p["device"])
    torch.manual_seed(p["seed"])

    fold = load_folds(p["splits"])[fold_idx]
    length = p["sequence_length"]
    sequence = length > 0
    common = (p["epochs_dir"], fold, p["val_ratio"], p["seed"], p["filter"], p["normalize"])

    if sequence:
        train_ds, val_ds, test_ds = build_sequence_split(
            *common, length, p["sequence_stride"], limit
        )
        unit = f"묶음(각 {length}조각)"
    else:
        train_ds, val_ds, test_ds = build_epoch_split(*common, limit)
        unit = "조각"
    print(
        f"학습 {len(train_ds):,} / 검증 {len(val_ds):,} / 평가 {len(test_ds):,} {unit}, {device}",
        flush=True,
    )

    def loader(ds, train: bool):
        return DataLoader(
            ds,
            batch_size=p["batch_size"] if train else p["batch_size"] * 2,
            shuffle=train,
            drop_last=train,
            num_workers=p["workers"],
            pin_memory=device.type == "cuda",
        )

    train_loader, val_loader, test_loader = (
        loader(train_ds, True),
        loader(val_ds, False),
        loader(test_ds, False),
    )

    if sequence:
        model = SleepCNNLSTM().to(device)
        model(torch.zeros(2, 3, 2, 3000, device=device))
    else:
        model = SleepCNN().to(device)
        model(torch.zeros(2, 2, 3000, device=device))  # 한 번 흘려 마지막 층 크기를 정한다
    criterion = nn.CrossEntropyLoss(
        weight=_class_weights(train_ds.labels, p["class_weight"], device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=p["lr"])
    # 이 GPU 가 지원하는 절반 정밀도 방식으로 학습 속도를 높인다
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and p["amp"])

    history, best = [], {"val_macro_f1": -1.0}
    best_state = None
    for epoch in range(p["epochs"]):
        model.train()
        t0, total = time.perf_counter(), 0.0
        for x, y, *_ in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                logits = model(x)
                # 묶음 단위면 조각마다 손실을 내야 하므로 펼쳐서 계산한다
                loss = (
                    criterion(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
                    if sequence
                    else criterion(logits, y)
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(y)

        val = evaluate(model, val_loader, device, sequence)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(total / len(train_ds), 4),
                **{f"val_{k}": round(v, 4) for k, v in val.items()},
                "sec": round(time.perf_counter() - t0, 1),
            }
        )
        if val["macro_f1"] > best["val_macro_f1"]:
            best = {"epoch": epoch, "val_macro_f1": val["macro_f1"]}
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(
            f"  epoch {epoch:>2}  loss {history[-1]['train_loss']:.4f}  "
            f"검증 macro-F1 {val['macro_f1']:.4f}  ({history[-1]['sec']}s)",
            flush=True,
        )

    # 평가 데이터는 회차를 고른 뒤 여기서 한 번만 쓴다
    model.load_state_dict(best_state)
    best.update(evaluate(model, test_loader, device, sequence))

    run = run_dir(p)
    out_dir = run / f"fold{fold_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"fold": fold_idx, "best": best, "history": history, "params": dict(p)}
    (out_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    torch.save(best_state, out_dir / "model.pt")
    _log_mlflow(p, fold_idx, best, out_dir, run)
    _write_summary(run)
    print(
        f"fold {fold_idx} 최고 (epoch {best['epoch']})  macro-F1 {best['macro_f1']:.4f}  "
        f"acc {best['accuracy']:.4f}  κ {best['kappa']:.4f}",
        flush=True,
    )
    print(
        "클래스별 F1  " + " / ".join(f"{s} {best[f'f1_{s}']:.3f}" for s in STAGE_NAMES), flush=True
    )
    return result


def _write_summary(run_dir: Path) -> None:
    """묶음이 다 차면 평균을 한 파일에 모은다."""
    folds = sorted(run_dir.glob("fold*/metrics.json"))
    if len(folds) < N_FOLDS:
        return
    best = [json.loads(f.read_text())["best"] for f in folds]
    keys = [k for k, v in best[0].items() if isinstance(v, int | float) and k != "epoch"]
    summary = {
        "run": run_paths.label(run_dir),
        "n_folds": len(best),
        "mean": {k: round(float(np.mean([b[k] for b in best])), 4) for k in keys},
        "per_fold": {k: [round(b[k], 4) for b in best] for k in keys},
        "chosen_epoch": [b["epoch"] for b in best],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    mean = summary["mean"]["macro_f1"]
    print(f"=== {run_paths.label(run_dir)} 10묶음 평균 macro-F1 {mean:.4f} ===", flush=True)


def _log_mlflow(params: dict, fold_idx: int, best: dict, out_dir: Path, run: Path) -> None:
    """학습 결과를 조회 도구에 기록한다. 실패해도 결과 파일은 이미 저장돼 있다."""
    try:
        import mlflow

        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("sleepstage")
        with mlflow.start_run(run_name=f"{run_paths.label(run)}_fold{fold_idx}"):
            mlflow.log_params({k: v for k, v in params.items() if not isinstance(v, dict)})
            mlflow.log_metrics({k: v for k, v in best.items() if isinstance(v, int | float)})
            mlflow.log_artifacts(str(out_dir))
    except Exception as e:  # noqa: BLE001
        print(f"mlflow 기록 실패: {str(e).splitlines()[0][:80]}", flush=True)
