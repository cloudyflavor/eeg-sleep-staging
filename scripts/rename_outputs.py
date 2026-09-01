"""이름 규칙이 바뀌었을 때 이미 있는 산출물을 새 이름으로 옮깁니다.

특징 표와 학습 실행의 이름은 설정을 요약한 값이다. 설정 목록에 항목을 더하면
요약값이 달라져서, 같은 산출물이 다른 이름을 갖게 된다. 그러면 다음에 돌릴 때
이미 있는 것을 못 찾고 처음부터 다시 만든다. 표 하나가 5분, 학습 하나가 40분이다.

값은 그대로이므로 다시 만들 필요가 없고 이름만 옮기면 된다.

**표를 먼저 옮기고 실행을 나중에 옮긴다.** 실행 이름이 표 이름에서 계산되므로
순서가 바뀌면 실행이 옛 표를 가리킨 채로 새 이름을 갖는다.

    python scripts/rename_outputs.py            무엇이 바뀌는지만 봅니다
    python scripts/rename_outputs.py --apply    실제로 옮깁니다
"""

import argparse
import json
from pathlib import Path

from sleepstage.config import Config, config_hash
from sleepstage.experiment import naming
from sleepstage.experiment.extract import SETTING_KEYS
from sleepstage.experiment.train import run_id

FEATURES = Path("data/features")
RUNS = Path("runs")
FEATURE_CONFIG = "configs/features/base.yaml"


def _defaults() -> dict:
    """설정 파일에 적힌 기본값. 옛 기록에 없는 항목을 이것으로 채운다."""
    return dict(Config.load(FEATURE_CONFIG)["features"])


def rename_tables(apply: bool) -> dict[str, str]:
    """특징 표를 새 이름으로 옮기고 옛 이름에서 새 이름으로 가는 표를 돌려준다."""
    defaults = _defaults()
    mapping: dict[str, str] = {}
    for manifest_path in sorted(FEATURES.glob("*.manifest.json")):
        man = json.loads(manifest_path.read_text(encoding="utf-8"))
        settings = {k: man.get("settings", {}).get(k, defaults[k]) for k in SETTING_KEYS}
        new = config_hash(settings)
        old = manifest_path.name.split(".")[0]
        if new == old:
            continue

        parquet = FEATURES / f"{old}.parquet"
        if not parquet.exists():
            raise SystemExit(f"표가 없습니다: {parquet}")
        if (FEATURES / f"{new}.parquet").exists():
            raise SystemExit(f"새 이름이 이미 있습니다: {new}")

        mapping[old] = new
        print(f"  표    {old} -> {new}  {settings['filter']}/{settings['context']}")
        if not apply:
            continue
        man["settings"] = settings
        man["settings_hash"] = new
        man["renamed_from"] = old
        parquet.rename(FEATURES / f"{new}.parquet")
        (FEATURES / f"{new}.manifest.json").write_text(
            json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        manifest_path.unlink()
    return mapping


def rename_runs(apply: bool, tables: dict[str, str]) -> int:
    """학습 실행을 새 이름으로 옮긴다. 표가 옮겨졌으면 가리키는 경로도 고친다."""
    paths = sorted(RUNS.glob("*/*/*/*/manifest.json")) + sorted(
        RUNS.glob("improve/*/*/manifest.json")
    )
    moved = 0
    for manifest_path in paths:
        man = json.loads(manifest_path.read_text(encoding="utf-8"))
        train = dict(man["train"])
        stem = Path(train["features"]).stem
        if stem in tables:
            stem = tables[stem]
            train["features"] = str(Path(train["features"]).with_stem(stem))

        new = run_id(stem, train)
        old_dir = manifest_path.parent
        if new[: naming.ID_LENGTH] == old_dir.name and train == man["train"]:
            continue

        new_dir = old_dir.with_name(new[: naming.ID_LENGTH])
        moved += 1
        print(f"  실행  {old_dir.relative_to(RUNS)} -> {new_dir.name}")
        if not apply:
            continue
        if new_dir != old_dir and new_dir.exists():
            raise SystemExit(f"실행 폴더가 이미 있습니다: {new_dir}")
        man["train"] = train
        man["run_id"] = new
        man["features_file"] = f"{stem}.parquet"
        man["renamed_from"] = old_dir.name
        manifest_path.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
        if new_dir != old_dir:
            old_dir.rename(new_dir)
    return moved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="실제로 옮깁니다")
    args = ap.parse_args()

    tables = rename_tables(args.apply)
    runs = rename_runs(args.apply, tables)
    done = "옮겼습니다" if args.apply else "바뀝니다"
    print(f"\n표 {len(tables)}개, 실행 {runs}개가 {done}.")
    if not args.apply:
        print("실제로 옮기려면 --apply 를 붙이세요.")


if __name__ == "__main__":
    main()
