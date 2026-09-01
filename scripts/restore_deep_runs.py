"""지워진 신경망 실행 폴더를 기록 도구에 남은 지표로 되살립니다.

신경망은 폴드마다 metrics.json 을 쓰고 열 폴드가 다 차면 summary.json 을 쓴다.
부스팅처럼 실행 폴더 바로 아래에 metrics.json 이 없어서, 그것만 보고 미완성이라
판단하는 정리 명령에 지워지기 쉽다.

가중치는 되살리지 못한다. 지표만 돌아온다.

    python scripts/restore_deep_runs.py            무엇이 돌아오는지만 봅니다
    python scripts/restore_deep_runs.py --apply    실제로 되살립니다
"""

import argparse
import collections
import json
import sqlite3
from pathlib import Path

import numpy as np

RUNS = Path("runs")
DB = "mlflow.db"
N_FOLDS = 10


def collect() -> dict[str, dict[int, dict]]:
    """기록 도구에서 신경망 실행의 폴드별 지표를 모은다."""
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT r.name, m.key, m.value FROM runs r JOIN metrics m ON m.run_uuid = r.run_uuid "
        "WHERE r.status = 'FINISHED' AND (r.name LIKE 'cnn/%' OR r.name LIKE 'lstm/%')"
    ).fetchall()
    per_run: dict[str, dict] = collections.defaultdict(dict)
    for name, key, value in rows:
        per_run[name][key] = value

    grouped: dict[str, dict[int, dict]] = collections.defaultdict(dict)
    for name, metrics in per_run.items():
        if "_fold" not in name:
            continue
        run, fold = name.rsplit("_fold", 1)
        grouped[run][int(fold)] = metrics
    return {run: folds for run, folds in grouped.items() if len(folds) >= N_FOLDS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="실제로 되살립니다")
    args = ap.parse_args()

    for run, folds in sorted(collect().items()):
        run_dir = RUNS / run
        if (run_dir / "summary.json").exists():
            print(f"그대로  {run}")
            continue
        best = []
        for idx in range(N_FOLDS):
            metrics = {
                k: (int(v) if k == "epoch" else round(float(v), 6)) for k, v in folds[idx].items()
            }
            best.append(metrics)
            if args.apply:
                out = run_dir / f"fold{idx}"
                out.mkdir(parents=True, exist_ok=True)
                (out / "metrics.json").write_text(
                    json.dumps({"best": metrics}, ensure_ascii=False, indent=2), encoding="utf-8"
                )
        keys = [k for k, v in best[0].items() if isinstance(v, int | float) and k != "epoch"]
        summary = {
            "run": run,
            "n_folds": len(best),
            "mean": {k: round(float(np.mean([b[k] for b in best])), 4) for k in keys},
            "per_fold": {k: [round(b[k], 4) for b in best] for k in keys},
            "chosen_epoch": [b["epoch"] for b in best],
            # 가중치는 못 되살린다. 다시 쓰려면 그 실행만 학습을 다시 돌려야 한다.
            "restored_without_weights": True,
        }
        print(f"되살림  {run}  macro-F1 {summary['mean']['macro_f1']:.4f}")
        if args.apply:
            (run_dir / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    if not args.apply:
        print("\n실제로 되살리려면 --apply 를 붙이세요.")


if __name__ == "__main__":
    main()
