"""모델을 학습하고 성능을 평가한다.

데이터를 여러 묶음으로 나눠 번갈아 학습하고 평가한다. 실행 결과는 설정, 환경,
지표, 그리고 에포크별 예측값까지 한 폴더에 모아둔다.

예측값을 저장해두면 나중에 다른 지표로 다시 계산할 때 학습을 반복하지 않아도 된다.
"""

import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

from sleepstage.config import Config, config_hash
from sleepstage.experiment import naming, subjects
from sleepstage.experiment.split import STAGE_NAMES, fold_masks, load_folds

#: 모델 입력이 아닌 열들. 이 열들을 뺀 나머지가 전부 특징이다.
META_COLS = ("subject", "night", "epoch_idx", "stage")

#: 일부 열만 골라 쓰기 위한 조건들. 어떤 특징이 실제로 기여하는지 확인할 때 쓴다.
#: 두 채널 비율 열은 채널 하나만 쓸 때 계산할 수 없으므로 함께 빠진다.
SUBSETS = {
    "all": lambda c: True,
    "fpz_only": lambda c: c.startswith("EEG Fpz-Cz") or c == "time_hour",
    "pz_only": lambda c: c.startswith("EEG Pz-Oz") or c == "time_hour",
    "no_distance": lambda c: "mmd" not in c and "esis" not in c,
}


def _majority(p: dict):
    from sklearn.dummy import DummyClassifier

    return DummyClassifier(strategy="prior")


def _ridge(p: dict):
    """비선형이 필요한 문제인지 확인하는 선형 비교군."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # 선형 모델은 값의 크기에 민감하다. 크기 조정을 모델과 함께 묶어야
    # 평가 데이터의 정보가 학습에 새지 않는다
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(penalty="l2", solver="lbfgs", max_iter=2000, random_state=p["seed"]),
    )


def _histgb(p: dict):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(random_state=p["seed"])


def _xgboost(p: dict):
    from xgboost import XGBClassifier

    return XGBClassifier(
        random_state=p["seed"], n_jobs=_threads(p), tree_method="hist", eval_metric="mlogloss"
    )


def _lightgbm(p: dict):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(random_state=p["seed"], n_jobs=_threads(p), verbosity=-1)


def _catboost(p: dict):
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        random_seed=p["seed"],
        thread_count=_threads(p),
        verbose=False,
        loss_function="MultiClass",
    )


#: 결과를 정하는 설정들. 실행 이름은 이것만 보고 짓는다.
#: 저장 위치나 코어 수처럼 값을 안 바꾸는 것을 넣으면, 그런 걸 건드릴 때마다
#: 같은 실험이 다른 실행으로 갈라져 이미 있는 결과를 못 찾는다.
RESULT_KEYS = (
    "splits",
    "model",
    "class_weight",
    "seed",
    "drop_missing",
    "feature_subset",
    "subject_info",
    "age_weight",
)
# subject_table 은 뺐다. 원본 EDF 처럼 고정된 참조 파일이라 바뀌지 않고,
# 넣어두면 경로를 옮기는 것만으로 모든 실행이 갈라진다.
# 그 표를 쓰는지 마는지는 subject_info 가 이미 말해준다.

#: 이름에 안 들어가는 설정의 기본값. 옛 실행의 이름을 다시 계산할 때 채운다.
RESULT_DEFAULTS = {"subject_info": "none", "age_weight": False}


def run_id(features_stem: str, train_params: dict) -> str:
    """실행을 구분하는 짧은 문자열. 결과를 정하는 것만 들어간다."""
    used = {k: train_params.get(k, RESULT_DEFAULTS.get(k)) for k in RESULT_KEYS}
    return config_hash({"features": features_stem, "train": used})


#: 부스팅이 쓸 코어 수. 공용 서버라 40개를 다 쓰지 않는다.
#: 실험을 여러 개 동시에 돌릴 때는 `train.threads` 로 나눠 준다.
#: 이 값은 결과를 바꾸지 않으므로 실행 이름에도 들어가지 않는다.
DEFAULT_THREADS = 24


def _threads(p: dict) -> int:
    return int(p.get("threads") or DEFAULT_THREADS)


#: 모델 이름 -> 만드는 함수. 새 모델은 여기 한 줄만 추가하면 된다.
#: 수면 단계별 개수가 달라 생기는 편향은 모든 모델에 같은 방식으로 보정한다.
MODELS = {
    "majority": _majority,
    "ridge": _ridge,
    "histgb": _histgb,
    "xgboost": _xgboost,
    "lightgbm": _lightgbm,
    "catboost": _catboost,
}


def _fit(model, X, y, sample_weight):
    """모델을 학습시킨다. 여러 단계로 묶인 모델이면 마지막 단계에 가중치를 넘긴다."""
    from sklearn.pipeline import Pipeline

    if sample_weight is None:
        model.fit(X, y)
    elif isinstance(model, Pipeline):
        last = model.steps[-1][0]
        model.fit(X, y, **{f"{last}__sample_weight": sample_weight})
    else:
        model.fit(X, y, sample_weight=sample_weight)
    return model


def _importance(model) -> np.ndarray | None:
    """어떤 특징이 판단에 많이 쓰였는지 뽑는다. 모델이 지원하지 않으면 None."""
    from sklearn.pipeline import Pipeline

    est = model.steps[-1][1] if isinstance(model, Pipeline) else model
    if hasattr(est, "coef_"):
        return np.abs(est.coef_).mean(axis=0)
    imp = getattr(est, "feature_importances_", None)
    return np.asarray(imp).ravel() if imp is not None else None


def make_model(cfg: Config):
    name = cfg["train"]["model"]
    if name not in MODELS:
        raise ValueError(f"모르는 모델입니다: {name!r} ({'/'.join(MODELS)})")
    return MODELS[name](cfg["train"])


def feature_settings(features_path: Path) -> dict:
    """입력 데이터를 만들 때 쓴 전처리 설정을 읽어온다.

    빈 값으로 넘어가면 실행 폴더 이름이 ?-?-? 가 되어 무엇을 돌렸는지 알 수 없게
    된다. 기록은 표와 함께 반드시 만들어지므로 없으면 멈춘다.
    """
    side = features_path.with_suffix(".manifest.json")
    if not side.exists():
        raise SystemExit(f"특징 표의 기록이 없습니다: {side}")
    return json.loads(side.read_text(encoding="utf-8"))["settings"]


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


def features_file(train_cfg: dict) -> Path:
    """학습에 쓸 특징 표 경로. 설정 파일에 적어 두지 않고 실행마다 받는다."""
    path = train_cfg.get("features")
    if not path:
        raise SystemExit(
            "특징 표 경로를 --set train.features 로 주세요.\n"
            "이름이 특징 설정에서 계산되므로 설정 파일에 적어 두면 설정이 바뀔 때 낡습니다.\n"
            "경로는 sleepstage feature-path 가 냅니다."
        )
    if not Path(path).exists():
        raise SystemExit(f"특징 표가 없습니다: {path}")
    return Path(path)


def run_cv(cfg: Config, overwrite: bool = False) -> dict[str, Any]:
    """묶음별로 번갈아 학습하고 평가한 뒤 전체 성능을 계산한다.

    같은 설정으로 이미 실행한 결과가 있으면 다시 학습하지 않는다.
    """
    p = dict(cfg["train"])
    features_path = features_file(p)
    rid = run_id(features_path.stem, p)
    settings = feature_settings(features_path)
    added = naming.addition_slug(settings, p)
    out_dir = naming.run_dir(
        p["out_dir"],
        p["model"],
        naming.preprocess_slug(settings),
        naming.options_slug(p["class_weight"], p.get("feature_subset")),
        rid,
        added,
    )

    done = out_dir / "metrics.json"
    if done.exists() and not overwrite:
        print(f"건너뜀 — 같은 설정의 실행이 이미 있습니다: {out_dir}", flush=True)
        return {"run_id": rid, "out_dir": str(out_dir), **json.loads(done.read_text())}

    table, load_info = load_table(features_path, p["drop_missing"])
    folds = load_folds(p["splits"])
    subset = p.get("feature_subset", "all")
    if subset not in SUBSETS:
        raise ValueError(f"모르는 feature_subset: {subset!r} ({'/'.join(SUBSETS)})")
    feature_cols = [c for c in table.columns if c not in META_COLS and SUBSETS[subset](c)]

    # 사람에 대해 원래 아는 값은 특징 표에 굽지 않고 여기서 붙인다
    table, extra = subjects.attach(table, p["subject_table"], p.get("subject_info", "none"))
    feature_cols += extra
    out_dir.mkdir(parents=True, exist_ok=True)

    # 표를 배열로 한 번만 바꾼다. 라이브러리마다 열 이름 규칙이 달라 생기는
    # 문제도 함께 피한다
    X_all = table[feature_cols].to_numpy(dtype=np.float32)
    y_all = table["stage"].to_numpy()

    predictions, fold_metrics, importances = [], [], []
    t0 = time.perf_counter()

    for i, fold in enumerate(folds):
        train_mask, test_mask = fold_masks(table, fold)
        sw = None
        if p["class_weight"] == "balanced":
            sw = compute_sample_weight("balanced", y_all[train_mask])
        if p.get("age_weight"):
            band = subjects.age_band_weight(table, train_mask)
            sw = band if sw is None else sw * band
        model = _fit(make_model(cfg), X_all[train_mask], y_all[train_mask], sw)
        imp = _importance(model)
        if imp is not None:
            importances.append(imp)

        # 모델이 아는 클래스 순서에 맞춰 확률을 배치한다. 학습 데이터에 없던
        # 단계가 있어도 열이 밀리지 않는다
        raw_proba = model.predict_proba(X_all[test_mask])
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
    if importances:
        pd.DataFrame(
            {"feature": feature_cols, "importance": np.mean(importances, axis=0)}
        ).sort_values("importance", ascending=False).to_csv(out_dir / "importance.csv", index=False)
    metrics = {
        "pooled": pooled,
        "folds": fold_metrics,
        "macro_f1_std_across_folds": float(np.std([m["macro_f1"] for m in fold_metrics])),
        "elapsed_sec": round(elapsed, 1),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    manifest = {
        "run_id": rid,
        "run_name": naming.label(out_dir),
        "train": p,
        "feature_settings": settings,
        "condition": _condition(settings, p, added),
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


def _condition(settings: dict, train_params: dict, added: str) -> dict:
    """무엇을 고정했고 무엇을 바꿨는지 한 자리에 적는다.

    폴더 이름에는 바뀐 것만 넣는다. 고정된 조건까지 이름에 넣으면 길기만 하고
    무엇을 더한 실험인지가 안 보인다. 대신 여기에 빠짐없이 남긴다.
    """
    changed = {key for key, default, _ in naming.ADDITIONS if settings.get(key, default) != default}
    fixed = {k: v for k, v in settings.items() if k not in changed}
    for key, default in (("subject_info", "none"), ("age_weight", False)):
        if train_params.get(key, default) != default:
            fixed.pop(key, None)
    return {
        "changed": added or "없음",
        "fixed": {
            **fixed,
            "model": train_params["model"],
            "class_weight": train_params["class_weight"],
            "drop_missing": train_params["drop_missing"],
            "feature_subset": train_params.get("feature_subset", "all"),
            "subject_info": train_params.get("subject_info", "none"),
            "age_weight": bool(train_params.get("age_weight")),
            "splits": train_params["splits"],
            "seed": train_params["seed"],
        },
    }


def _log_mlflow(cfg: Config, manifest: dict, pooled: dict, out_dir: Path) -> None:
    """실험 결과를 조회 도구에 기록한다.

    보기 편하라고 남기는 것이므로 실패해도 실행에 영향을 주지 않는다. 결과 파일은
    이미 저장이 끝난 상태다.
    """
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
