"""실험 묶음을 설정 파일 하나로 돌린다.

전에는 실험군마다 셸 스크립트를 하나씩 뒀다. 스크립트가 열 개가 되자 고정 조건이
파일마다 따로 적히기 시작했고, 실제로 어떤 스크립트는 결측 행을 버리고 어떤
스크립트는 안 버렸다. 그래서 같은 축을 훑은 실행들 사이에 조건이 갈라졌고,
그건 결과 표를 봐도 안 보였다.

여기서는 **고정 조건을 한 번만 적는다.** 갈라질 자리가 없다.

    fixed:  모든 실행에 똑같이 적용
    runs:   여기 적은 것만 실행마다 다르다

특징 설정이 바뀌면 표를 먼저 만들어야 하는데, 그 판단도 여기서 한다. 바뀐 설정이
``features`` 아래 있으면 표를 만들고, ``train`` 아래만 있으면 이미 있는 표를 쓴다.
"""

import json
import time
from pathlib import Path
from typing import Any

import yaml

from sleepstage.config import Config
from sleepstage.experiment.pipeline import features as feature_build

FEATURE_CONFIG = "configs/features/base.yaml"
TRAIN_CONFIG = "configs/experiment/base.yaml"
DEEP_CONFIG = "configs/experiment/deep.yaml"

#: 묶음이 무엇을 돌리는가. 신경망은 묶음마다 따로 돌려야 해서 모양이 다르다.
KINDS = ("train", "deep")


def load(path: str | Path) -> dict[str, Any]:
    """묶음 설정을 읽고 모양이 맞는지 본다."""
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for key in ("name", "runs"):
        if key not in spec:
            raise SystemExit(f"{path} 에 {key} 가 없습니다")
    if spec.get("kind", "train") not in KINDS:
        raise SystemExit(f"{path} 의 kind 가 이상합니다: {spec['kind']!r} ({'/'.join(KINDS)})")
    for section in ("fixed", *[f"runs[{i}]" for i in range(len(spec["runs"]))]):
        block = spec.get("fixed", {}) if section == "fixed" else spec["runs"][int(section[5:-1])]
        unknown = set(block) - {"features", "train", "deep", "name"}
        if unknown:
            raise SystemExit(f"{path} 의 {section} 에 모르는 항목이 있습니다: {sorted(unknown)}")
    return spec


def _merged(spec: dict, run: dict, section: str) -> dict:
    """고정 조건 위에 이 실행에서 바꾼 것만 덮는다."""
    return {**spec.get("fixed", {}).get(section, {}), **run.get(section, {})}


def _overrides(values: dict, prefix: str) -> list[str]:
    return [
        f"{prefix}.{k}={json.dumps(v) if isinstance(v, bool) else v}" for k, v in values.items()
    ]


def plan(spec: dict) -> list[dict]:
    """무엇을 돌릴지 미리 펼친다. 돌리기 전에 눈으로 확인하려고 따로 둔다."""
    if spec.get("kind") == "deep":
        return [
            {"name": r.get("name", str(i)), "deep": _merged(spec, r, "deep")}
            for i, r in enumerate(spec["runs"])
        ]
    out = []
    for i, run in enumerate(spec["runs"]):
        feats = _merged(spec, run, "features")
        train = _merged(spec, run, "train")
        cfg = Config.load(FEATURE_CONFIG).override(_overrides(feats, "features"))
        out.append(
            {
                "name": run.get("name", f"{i}"),
                "features": feats,
                "train": train,
                "table": str(feature_build.output_path(cfg)),
                "table_exists": feature_build.output_path(cfg).exists(),
            }
        )
    return out


def _run_deep(steps: list[dict]) -> dict[str, Any]:
    """신경망은 묶음 하나씩 따로 돌린다. 모든 묶음이 차야 평균이 나온다."""
    from sleepstage.experiment.baselines.deep.train import N_FOLDS, train_fold

    for s in steps:
        print(f"\n=== {s['name']} ===", flush=True)
        for fold in range(N_FOLDS):
            train_fold(Config.load(DEEP_CONFIG).override(_overrides(s["deep"], "deep")), fold)
    return {"results": [{"name": s["name"]} for s in steps]}


def run_sweep(path: str | Path, jobs: int = 8, dry_run: bool = False) -> dict[str, Any]:
    """묶음을 순서대로 돌린다. 하나라도 실패하면 멈춘다."""
    from sleepstage.experiment.modeling.cross_validate import run_cv

    spec = load(path)
    steps = plan(spec)
    print(f"=== {spec['name']} · 실행 {len(steps)}개 ===", flush=True)
    for s in steps:
        if spec.get("kind") == "deep":
            print(f"  {s['name']:<24} {s['deep']}", flush=True)
        else:
            mark = "표 있음" if s["table_exists"] else "표 만들어야 함"
            print(f"  {s['name']:<24} {Path(s['table']).name}  {mark}", flush=True)
    if dry_run:
        return {"planned": steps}
    if spec.get("kind") == "deep":
        return _run_deep(steps)

    results, t0 = [], time.perf_counter()
    for s in steps:
        print(f"\n=== {s['name']} ===", flush=True)
        if not s["table_exists"]:
            cfg = Config.load(FEATURE_CONFIG).override(_overrides(s["features"], "features"))
            feature_build.extract_all(cfg, jobs=jobs)

        overrides = [f"train.features={s['table']}", *_overrides(s["train"], "train")]
        r = run_cv(Config.load(TRAIN_CONFIG).override(overrides))
        macro = r["pooled"]["macro_f1"]
        print(f"  {s['name']}  macro-F1 {macro:.4f}  →  {r['out_dir']}", flush=True)
        results.append({"name": s["name"], "macro_f1": macro, "out_dir": r["out_dir"]})

    print(f"\n=== 완료 · {time.perf_counter() - t0:.0f}초 ===", flush=True)
    for r in results:
        print(f"  {r['name']:<24} {r['macro_f1']:.4f}")
    return {"name": spec["name"], "results": results}
