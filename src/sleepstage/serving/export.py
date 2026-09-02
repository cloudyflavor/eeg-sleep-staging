"""실제로 서비스할 모델 파일을 만든다.

실험은 데이터를 나눠가며 모델을 여러 번 만들었다 버리므로 남는 파일이 없다.
여기서 전체 데이터로 한 번 학습해 파일로 저장한다.

모델 파일만으로는 어떤 순서로 값을 넣어야 하는지 알 수 없다. 순서가 어긋나면
오류 없이 엉뚱한 답이 나오므로, 열 이름과 전처리 설정을 함께 저장한다.
"""

import contextlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sleepstage.config import Config
from sleepstage.experiment.modeling import feature_selection
from sleepstage.experiment.modeling.cross_validate import (
    META_COLS,
    SUBSETS,
    _fit,
    _threads,
    features_file,
    load_table,
    make_model,
)
from sleepstage.experiment.modeling.stage_transitions import RECORDING, transition_log_prob
from sleepstage.experiment.pipeline.folds import STAGE_NAMES

#: 모델 종류별 저장 파일 이름
MODEL_FILES = {"catboost": "model.cbm", "xgboost": "model.json", "lightgbm": "model.txt"}


def save_model(model, model_type: str, out_dir: Path) -> str:
    """모델을 파일로 저장한다.

    라이브러리가 제공하는 자체 형식을 쓴다. 파이썬 객체를 통째로 저장하는 방식은
    라이브러리 버전이 바뀌면 읽지 못하게 된다.
    """
    name = MODEL_FILES.get(model_type)
    if name is None:
        raise ValueError(f"네이티브 저장을 지원하지 않는 모델입니다: {model_type}")
    path = out_dir / name
    if model_type == "lightgbm":
        model.booster_.save_model(str(path))
    else:
        model.save_model(str(path))
    return name


def _versions() -> dict[str, str]:
    import sklearn

    out = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
    }
    for name in ("catboost", "xgboost", "lightgbm", "scipy", "mne", "antropy"):
        with contextlib.suppress(Exception):  # 설치돼 있지 않으면 기록하지 않는다
            out[name] = __import__(name).__version__
    return out


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def transition_block(table, path: str | Path | None) -> dict | None:
    """학습에 쓴 라벨로 전이 표를 만들고 세기를 정한다.

    세기는 교차검증이 묶음마다 나머지 묶음 안에서 고른 값 중 가장 많이 나온 것을 쓴다.
    여기서 다시 고르면 학습에 쓴 라벨을 보고 고르는 것이 되어 낙관적으로 부풀려진다.
    """
    if not path:
        return None
    chosen = json.loads(Path(path).read_text(encoding="utf-8"))["weights"]
    weights = [float(v) for v in chosen.values()]
    ordered = table.sort_values([*RECORDING, "epoch_idx"])
    labels = [g["stage"].to_numpy() for _, g in ordered.groupby(RECORDING, sort=False)]
    return {
        "weight": max(set(weights), key=weights.count),
        "log_prob": transition_log_prob(labels).tolist(),
    }


def export(
    cfg: Config,
    out_dir: str | Path,
    version: str,
    metrics_json: str | None = None,
    transitions_json: str | Path | None = None,
) -> dict[str, Any]:
    """전체 데이터로 학습하고 서비스에 필요한 파일들을 저장한다.

    모델 파일, 입력 열 이름과 순서, 전처리 설정, 성능과 한계를 적은 설명서를 만든다.
    """
    p = cfg["train"]
    features_path = features_file(p)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 실험과 같은 방식으로 읽는다. 여기서만 결측을 버리면 실험에서 이긴 모델과
    # 다른 모델이 배포된다
    # 학습과 같은 목록을 봐야 한다. 여기서 빠뜨리면 배포 모델만 걸러내지 않은
    # 녹음으로 학습되고, 파이프라인 설명과 실제 산출물이 어긋난다.
    from sleepstage.experiment.pipeline.quality import load_excluded

    table, load_info = load_table(
        features_path, p["drop_missing"], load_excluded(p.get("quality_report"))
    )

    subset = p.get("feature_subset", "all")
    if subset not in SUBSETS:
        raise ValueError(f"모르는 feature_subset: {subset!r} ({'/'.join(SUBSETS)})")
    feature_names = [c for c in table.columns if c not in META_COLS and SUBSETS[subset](c)]
    X = table[feature_names].to_numpy(dtype=np.float32)
    y = table["stage"].to_numpy()

    sw = None
    if p["class_weight"] == "balanced":
        from sklearn.utils.class_weight import compute_sample_weight

        sw = compute_sample_weight("balanced", y)

    # 값을 보고 고르는 가지치기는 배포 학습에 쓰는 데이터 전체에서 한 번 고른다.
    # 실험에서는 묶음마다 골랐지만 여기에는 평가 대상이 없다
    picked = feature_selection.choose(p.get("feature_select", "none"), X, y, sw, _threads(p))
    if picked is not None:
        feature_names = [feature_names[i] for i in picked]
        X = X[:, picked]

    print(f"학습 {len(table):,} 에포크 × {len(feature_names)} 특징 ({p['model']})", flush=True)
    model = _fit(make_model(cfg), X, y, sw)
    model_file = save_model(model, p["model"], out_dir)

    feature_manifest = json.loads(features_path.with_suffix(".manifest.json").read_text())
    # 신호 조건은 특징 표가 만들어질 때 기록된다. 여기 손으로 적으면 표와 어긋난다.
    missing = [k for k in ("sfreq", "epoch_seconds", "channels") if k not in feature_manifest]
    if missing:
        raise SystemExit(f"특징 표 기록에 신호 조건이 없습니다: {missing}")
    (out_dir / "feature_names.json").write_text(
        json.dumps(
            {
                "names": feature_names,
                # 내보낸 열만 남긴다. 안 쓰는 열의 미래 참조가 섞여 있으면
                # 이 모델의 지연을 실제보다 길게 보고하게 된다
                "lookahead_epochs": {
                    c: feature_manifest["lookahead_epochs"][c]
                    for c in feature_names
                    if c in feature_manifest["lookahead_epochs"]
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "preprocess": feature_manifest["settings"],
                "epoch_seconds": feature_manifest["epoch_seconds"],
                "sfreq": feature_manifest["sfreq"],
                "channels": feature_manifest["channels"],
                "stage_names": list(STAGE_NAMES),
                "model_type": p["model"],
                "transitions": transition_block(table, transitions_json),
                "model_file": model_file,
                "class_weight": p["class_weight"],
                "seed": p["seed"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    card: dict[str, Any] = {
        "version": version,
        "trained_on": {
            "dataset": "Sleep-EDF Expanded, Sleep-Cassette",
            "subjects": int(table["subject"].nunique()),
            "recordings": int(table.groupby(["subject", "night"]).ngroups),
            "epochs": len(table),
            "dropped_missing_rows": load_info["n_dropped_missing"],
        },
        "environment": _versions(),
        "git_commit": _git_commit(),
        "limitations": [
            "건강한 성인 코호트로만 학습했다, 수면장애 환자에서는 검증되지 않았다",
            "4-class 로 N1 과 N2 를 합쳤다, 5-class 문헌과 직접 비교할 수 없다",
            "100 Hz 2채널 EEG 를 전제한다, 다른 기기에서는 진폭 분포가 달라질 수 있다",
            "wake 절단 경계를 정답 라벨로 정했다, 취침 시 착용을 전제한다",
        ],
    }
    if metrics_json:
        m = json.loads(Path(metrics_json).read_text())["pooled"]
        card["performance"] = {
            "protocol": "subject-wise 10-fold CV",
            **{k: round(v, 4) for k, v in m.items()},
        }
        # 서빙은 전이 후처리를 걸고 답한다. 모델 단독 값만 적으면 기기가 내는 값과 다르다.
        if transitions_json:
            after = json.loads(Path(transitions_json).read_text())["after"]
            card["performance"]["macro_f1_with_transitions"] = round(after["macro_f1"], 4)
    (out_dir / "card.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    sizes = {f.name: round(f.stat().st_size / 1e6, 2) for f in out_dir.iterdir()}
    return {"out_dir": str(out_dir), "n_features": len(feature_names), "files_mb": sizes}


def _load_catboost(path: Path):
    from catboost import CatBoostClassifier

    model = CatBoostClassifier()
    model.load_model(str(path))
    return model


def _load_xgboost(path: Path):
    from xgboost import XGBClassifier

    model = XGBClassifier()
    model.load_model(str(path))
    return model


def _load_lightgbm(path: Path):
    import lightgbm as lgb

    return lgb.Booster(model_file=str(path))


#: 모델 종류 -> 읽는 함수. 새 종류를 붙이려면 여기에 한 줄만 더한다.
MODEL_LOADERS = {
    "catboost": _load_catboost,
    "xgboost": _load_xgboost,
    "lightgbm": _load_lightgbm,
}


def load_artifact(artifact_dir: str | Path):
    """저장된 파일들을 읽어 모델과 설정과 열 이름을 돌려준다.

    서비스 쪽은 이 함수만 알면 되고 모델 종류가 무엇인지 신경 쓰지 않아도 된다.
    모르는 종류는 다른 것으로 읽지 않고 바로 오류를 낸다.
    """
    artifact_dir = Path(artifact_dir)
    config = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
    names = json.loads((artifact_dir / "feature_names.json").read_text(encoding="utf-8"))

    kind = config["model_type"]
    if kind not in MODEL_LOADERS:
        raise ValueError(f"읽는 방법을 모르는 모델입니다: {kind!r} ({'/'.join(MODEL_LOADERS)})")
    model = MODEL_LOADERS[kind](artifact_dir / config["model_file"])
    return model, config, names["names"]


def artifact_version(artifact_dir: str | Path) -> str:
    """설명서에 적힌 버전을 읽는다. 없으면 폴더 이름을 쓴다."""
    artifact_dir = Path(artifact_dir)
    try:
        return json.loads((artifact_dir / "card.json").read_text(encoding="utf-8"))["version"]
    except (OSError, KeyError, json.JSONDecodeError):
        return artifact_dir.name
