"""4단계 — 학습·교차검증·러닝 커브. 근거: docs/05-experiments.md.

산출물은 runs/<실행해시>/ (설정·환경·지표·예측). 예측을 저장해 두면 지표를
다시 정의해도 재학습이 필요 없다.
"""

import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from sleepstage.config import Config, config_hash
from sleepstage.evaluation.split import STAGE_NAMES, fold_masks, load_folds

#: 특징이 아닌 열. 나머지 전부가 모델 입력이다.
META_COLS = ("subject", "night", "epoch_idx", "stage")


def _histgb(p: dict):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(class_weight=p["class_weight"], random_state=p["seed"])


#: 모델 추가는 여기 한 줄.
MODELS = {"histgb": _histgb}


def make_model(cfg: Config):
    name = cfg["train"]["model"]
    if name not in MODELS:
        raise ValueError(f"모르는 모델입니다: {name!r} ({'/'.join(MODELS)})")
    return MODELS[name](cfg["train"])


def run_id(cfg: Config, features_path: Path) -> str:
    """실행의 식별자. 특징 표(=설정해시가 이름)와 학습 설정이 정체성을 정한다."""
    return config_hash({"features": features_path.stem, "train": cfg["train"]})


def feature_settings(features_path: Path) -> dict:
    """특징 parquet 의 사이드카 매니페스트에서 전처리 설정을 읽는다. 없으면 빈 dict."""
    try:
        side = features_path.with_suffix(".manifest.json")
        return json.loads(side.read_text(encoding="utf-8"))["settings"]
    except (OSError, KeyError, json.JSONDecodeError):
        return {}


def load_table(features_path: Path, drop_missing: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    table = pd.read_parquet(features_path)
    info = {"n_rows": len(table), "n_dropped_missing": 0}
    if drop_missing:
        keep = table.notna().all(axis=1)
        info["n_dropped_missing"] = int((~keep).sum())
        table = table[keep].reset_index(drop=True)
    return table, info


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    per_class = f1_score(y_true, y_pred, average=None, labels=range(4), zero_division=0)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        **{f"f1_{name}": float(v) for name, v in zip(STAGE_NAMES, per_class, strict=True)},
    }


def _environment() -> dict[str, str]:
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    }


def run_cv(cfg: Config, overwrite: bool = False) -> dict[str, Any]:
    """10-fold 교차검증. 같은 run_id 의 완료 산출물이 있으면 건너뛴다."""
    p = cfg["train"]
    features_path = Path(p["features"])
    rid = run_id(cfg, features_path)
    out_dir = Path(p["out_dir"]) / rid

    done = out_dir / "metrics.json"
    if done.exists() and not overwrite:
        print(f"건너뜀 — 같은 설정의 실행이 이미 있습니다: {out_dir}", flush=True)
        return {"run_id": rid, "out_dir": str(out_dir), **json.loads(done.read_text())}

    table, load_info = load_table(features_path, p["drop_missing"])
    folds = load_folds(p["splits"])
    feature_cols = [c for c in table.columns if c not in META_COLS]
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions, fold_metrics = [], []
    y_all = table["stage"].to_numpy()
    t0 = time.perf_counter()

    for i, fold in enumerate(folds):
        train_mask, test_mask = fold_masks(table, fold)
        model = make_model(cfg)
        model.fit(table.loc[train_mask, feature_cols], y_all[train_mask])

        # 열 순서는 model.classes_ 기준 — 학습 fold 에 없던 클래스가 있어도 어긋나지 않게 맞춘다
        raw_proba = model.predict_proba(table.loc[test_mask, feature_cols])
        proba = np.zeros((len(raw_proba), len(STAGE_NAMES)), dtype=np.float32)
        proba[:, model.classes_.astype(int)] = raw_proba
        pred = proba.argmax(axis=1)
        fold_metrics.append(_metrics(y_all[test_mask], pred))

        chunk = table.loc[test_mask, list(META_COLS)].copy()
        chunk["pred"] = pred.astype(np.int8)
        chunk["fold"] = np.int8(i)
        for k, name in enumerate(STAGE_NAMES):
            chunk[f"proba_{name}"] = proba[:, k].astype(np.float32)
        predictions.append(chunk)
        print(f"  fold {i}  macro-F1 {fold_metrics[-1]['macro_f1']:.4f}", flush=True)

    pred_df = pd.concat(predictions, ignore_index=True)
    pooled = _metrics(pred_df["stage"].to_numpy(), pred_df["pred"].to_numpy())
    elapsed = time.perf_counter() - t0

    pred_df.to_parquet(out_dir / "predictions.parquet", index=False)
    metrics = {
        "pooled": pooled,
        "folds": fold_metrics,
        "macro_f1_std_across_folds": float(np.std([m["macro_f1"] for m in fold_metrics])),
        "elapsed_sec": round(elapsed, 1),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    manifest = {
        "run_id": rid,
        "run_name": _run_name(features_path, p),
        "train": p,
        "feature_settings": feature_settings(features_path),
        "features_file": features_path.name,
        **load_info,
        "n_features": len(feature_cols),
        "environment": _environment(),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _log_mlflow(cfg, manifest, pooled, out_dir)
    return {"run_id": rid, "out_dir": str(out_dir), **metrics}


def _run_name(features_path: Path, train_params: dict) -> str:
    """사람이 읽는 실행 이름: 'causal-smooth · histgb'. 설정을 못 읽으면 해시로."""
    st = feature_settings(features_path)
    if st:
        return f"{st.get('filter', '?')}-{st.get('context', '?')} · {train_params['model']}"
    return features_path.stem


def _log_mlflow(cfg: Config, manifest: dict, pooled: dict, out_dir: Path) -> None:
    """조회 계층. 실패해도 runs/ 산출물은 이미 완성돼 있다."""
    if not cfg["train"].get("mlflow"):
        return
    try:
        import mlflow

        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("sleepstage")
        with mlflow.start_run(run_name=manifest["run_name"]):
            mlflow.log_params(
                {
                    **{k: v for k, v in cfg["train"].items() if not isinstance(v, dict)},
                    **manifest["feature_settings"],
                    "features_file": manifest["features_file"],
                    "n_features": manifest["n_features"],
                    "run_id": manifest["run_id"],
                }
            )
            mlflow.log_metrics(pooled)
            mlflow.log_artifacts(str(out_dir))
    except Exception as e:  # noqa: BLE001 — 조회 계층이 실행을 죽이면 안 된다
        print(f"mlflow 기록 실패 (runs/ 는 온전함): {str(e).splitlines()[0][:80]}", flush=True)


def learning_curve(cfg: Config, counts: list[int], n_eval_folds: int = 3) -> dict[str, Any]:
    """피험자 수를 늘려가며 학습/평가 점수를 잰다. 부분집합도 피험자 단위 — 에포크 단위면 누수."""
    p = cfg["train"]
    features_path = Path(p["features"])
    table, _ = load_table(features_path, p["drop_missing"])
    folds = load_folds(p["splits"])[:n_eval_folds]
    feature_cols = [c for c in table.columns if c not in META_COLS]
    y_all = table["stage"].to_numpy()

    points = []
    for i, fold in enumerate(folds):
        rng = np.random.default_rng(p["seed"] + i)
        order = rng.permutation(sorted(fold["train"]))
        _, test_mask = fold_masks(table, fold)

        for n in counts:
            subset = set(order[:n])
            train_mask = table["subject"].isin(subset).to_numpy()
            model = make_model(cfg)
            t0 = time.perf_counter()
            model.fit(table.loc[train_mask, feature_cols], y_all[train_mask])
            point = {
                "fold": i,
                "n_subjects": n,
                "n_train_epochs": int(train_mask.sum()),
                "fit_sec": round(time.perf_counter() - t0, 1),
                "train_macro_f1": _metrics(
                    y_all[train_mask], model.predict(table.loc[train_mask, feature_cols])
                )["macro_f1"],
                "test_macro_f1": _metrics(
                    y_all[test_mask], model.predict(table.loc[test_mask, feature_cols])
                )["macro_f1"],
            }
            points.append(point)
            print(
                f"  fold {i}  n={n:>3}  train {point['train_macro_f1']:.4f}"
                f"  test {point['test_macro_f1']:.4f}  ({point['fit_sec']}s)",
                flush=True,
            )

    result = {
        "features_file": features_path.name,
        "train": p,
        "counts": counts,
        "points": points,
        "environment": _environment(),
    }
    out = Path(p["out_dir"]) / f"curve_{run_id(cfg, features_path)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["out"] = str(out)
    return result
