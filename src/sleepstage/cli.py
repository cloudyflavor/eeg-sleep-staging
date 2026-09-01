"""명령줄 진입점.

전처리부터 학습, 평가, 배포 파일 생성까지 모든 작업을 하위 명령으로 제공한다.
"""

import argparse

from sleepstage import __version__
from sleepstage.config import Config


def _cmd_prepare(args: argparse.Namespace) -> int:
    """EDF 를 30초 에포크 npz 로 바꾼다."""
    from sleepstage.experiment.pipeline.epochs import prepare_all

    cfg = Config.load(args.config)
    print(f"설정 {cfg.path}  해시 {cfg.hash}")
    m = prepare_all(
        cfg,
        root=args.root,
        out_dir=args.out,
        limit=args.limit,
        overwrite=args.overwrite,
        jobs=args.jobs,
    )
    print(
        f"\n녹음 {m['n_written']}개 / 이번에 건너뜀 {m['n_skipped']}개  "
        f"피험자 {m['n_subjects']}명\n"
        f"에포크 {m['n_epochs_kept']:,}  (절단 적용 시 {m['n_epochs_after_crop']:,})\n"
        f"클래스 {m['class_counts']}\n"
        f"제거   {m['dropped']}\n"
        f"용량   {m['bytes'] / 1e9:.2f} GB"
    )
    return 0


def _cmd_features(args: argparse.Namespace) -> int:
    """에포크 npz 에서 특징을 뽑아 parquet 표로 만든다."""
    from sleepstage.experiment.pipeline.features import extract_all, settings_hash

    cfg = Config.load(args.config).override(args.set or [])
    print(f"설정 {cfg.path}  설정해시 {settings_hash(cfg)}")
    m = extract_all(
        cfg,
        epochs_dir=args.epochs,
        out_path=args.out,
        limit=args.limit,
        jobs=args.jobs,
        overwrite=args.overwrite,
    )
    s = m["settings"]
    print(
        f"\n녹음 {m['n_recordings']}개  피험자 {m['n_subjects']}명\n"
        f"에포크 {m['n_epochs']:,}  특징 {m['n_features']}개\n"
        f"설정   필터 {s['filter']} / 정규화 {s['normalize']}"
        f" / 문맥 {s['context']}\n"
        f"클래스 {m['class_counts']}\n"
        f"용량   {m['bytes'] / 1e6:.1f} MB"
    )
    return 0


def _cmd_feature_path(args: argparse.Namespace) -> int:
    """설정이 만드는 특징 표 경로만 출력한다. 스크립트가 해시를 박아 두지 않도록."""
    from sleepstage.experiment.pipeline.features import output_path

    cfg = Config.load(args.config).override(args.set or [])
    print(output_path(cfg))
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    """만들어진 에포크 npz 를 원본 EDF 와 대조한다."""
    from sleepstage.experiment.pipeline.crosscheck import verify_all

    cfg = Config.load(args.config)
    r = verify_all(cfg, root=args.root, npz_dir=args.epochs, samples=args.samples)
    print(
        f"\n녹음 {r['n_recordings']}개 라벨 대조 · {r['n_signal_checked']}개 신호 대조"
        f"\n문제 {len(r['problems'])}건"
    )
    for problem in r["problems"][:20]:
        print(f"  {problem}")
    return 1 if r["problems"] else 0


def _cmd_split(args: argparse.Namespace) -> int:
    """피험자 단위 fold 배정을 파일로 고정한다."""
    from sleepstage.experiment.pipeline.folds import write_folds

    m = write_folds(args.features, args.out, n_splits=args.folds, seed=args.seed)
    print(f"피험자 {m['n_subjects']}명 · 에포크 {m['n_epochs']:,} → {m['n_splits']} fold")
    print(f"저장 {args.out}\n")
    print("fold  평가 피험자  평가 에포크   W      LS     DS     R")
    for i, d in enumerate(m["diagnostics"]["folds"]):
        r = d["test_class_ratio"]
        print(
            f"  {i:>2}  {d['n_test_subjects']:>10}  {d['n_test_epochs']:>10,}"
            f"  {r['W']:.3f}  {r['LS']:.3f}  {r['DS']:.3f}  {r['R']:.3f}"
        )
    print("\n클래스 비율 폭 (fold 간 최대−최소)")
    for stage, v in m["diagnostics"]["class_ratio_spread"].items():
        print(f"  {stage:<3} {v['min']:.3f} ~ {v['max']:.3f}   폭 {v['range']:.3f}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    """교차검증으로 학습하고 예측과 지표를 남긴다."""
    from sleepstage.experiment.modeling.cross_validate import run_cv

    cfg = Config.load(args.config).override(args.set or [])
    r = run_cv(cfg, overwrite=args.overwrite)
    m = r["pooled"]
    print(
        f"\nrun {r['run_id']}  →  {r['out_dir']}\n"
        f"macro-F1 {m['macro_f1']:.4f} (fold 표준편차 {r['macro_f1_std_across_folds']:.4f})"
        f"  acc {m['accuracy']:.4f}  κ {m['kappa']:.4f}\n"
        f"클래스별 F1  W {m['f1_W']:.3f} / LS {m['f1_LS']:.3f}"
        f" / DS {m['f1_DS']:.3f} / R {m['f1_R']:.3f}\n"
        f"소요 {r['elapsed_sec']}초"
    )
    return 0


def _cmd_budget(args: argparse.Namespace) -> int:
    """지연 예산별 성능 곡선."""
    from sleepstage.experiment.delay_curve import plot_curve, run_budgets

    cfg = Config.load(args.config).override(args.set or [])
    budgets = [None if x == "inf" else float(x) for x in args.budgets.split(",")]
    r = run_budgets(cfg, budgets)
    print(f"\n저장 {r['out']}")
    if args.plot:
        print(f"그림 {plot_curve(r['out'], args.plot)}")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    """실험 결과로 배포용 모델 아티팩트를 만든다."""
    from sleepstage.serving.export import export

    cfg = Config.load(args.config).override(args.set or [])
    r = export(cfg, args.out, args.version, args.metrics)
    print(f"\n저장 {r['out_dir']}  특징 {r['n_features']}개")
    for name, mb in sorted(r["files_mb"].items()):
        print(f"  {name:22} {mb:>7.2f} MB")
    return 0


def _cmd_deep(args: argparse.Namespace) -> int:
    """원신호를 그대로 넣는 신경망을 학습한다. 특징을 쓰는 모델과 조건을 맞춰 비교한다."""
    from sleepstage.experiment.baselines.deep.train import train_fold

    train_fold(Config.load(args.config).override(args.set or []), args.fold, args.limit)
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    """추론 API 를 띄웁니다."""
    import os

    import uvicorn

    os.environ["SLEEPSTAGE_ARTIFACT"] = args.artifact
    uvicorn.run("sleepstage.serving.api:app", host=args.host, port=args.port)
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    """녹음을 조각씩 흘려보내며 실시간 판정을 재현한다."""
    from sleepstage.serving.replay import compare_with_batch, replay

    r = replay(args.artifact, args.npz, limit=args.limit)
    print(
        f"{r['recording']}  조각 {r['n_epochs_fed']}개 투입, 판정 {r['n_decisions']}개\n"
        f"판정 지연 {r['delay_epochs']} 조각 ({r['delay_epochs'] * 0.5:.1f}분), "
        f"첫 판정은 {r['first_answer_after']}번째 조각에서\n"
        f"조각당 처리 시간 평균 {r['latency_ms']['mean']} ms, "
        f"95백분위 {r['latency_ms']['p95']} ms, 최대 {r['latency_ms']['max']} ms\n"
        f"라벨 대비 정확도 {r['accuracy']:.4f}"
    )
    if args.compare:
        subject, night = r["recording"].split("N")
        c = compare_with_batch(args.artifact, args.compare, r, subject, int(night))
        print(
            f"\n한꺼번에 계산한 결과와 비교: {c['matched']}/{c['compared']} 일치"
            f" ({c['agreement']:.4f})"
        )
        for e in c["examples"]:
            print(f"  {e['position']}번째: 배치 {e['batch']} vs 스트리밍 {e['stream']}")
    return 0


def _cmd_postprocess(args: argparse.Namespace) -> int:
    """저장된 확률에 단계 전이 후처리를 걸어 다시 채점한다."""
    import json
    from pathlib import Path

    import pandas as pd
    from scipy.stats import wilcoxon

    from sleepstage.experiment.modeling import stage_transitions

    run = Path(args.run)
    pred = pd.read_parquet(run / "predictions.parquet")
    lag = None if args.lag < 0 else args.lag
    out, info = stage_transitions.apply(pred, lag=lag, weight=args.weight)

    before = stage_transitions.score(out, "pred")
    after = stage_transitions.score(out, "pred_smoothed")
    diff = [a - b for a, b in zip(after["folds"], before["folds"], strict=True)]
    result = {
        **info,
        "before": before,
        "after": after,
        "gain_pp": (after["macro_f1"] - before["macro_f1"]) * 100,
        "folds_improved": sum(d > 0 for d in diff),
        "wilcoxon_p": float(wilcoxon(diff).pvalue) if any(diff) else 1.0,
    }
    (run / "transitions.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    out.to_parquet(run / "predictions_smoothed.parquet", index=False)

    picked = sorted(info["weights"].values())
    print(
        f"뒤를 보는 조각 {info['lag']}  고른 세기 {picked}\n"
        f"macro-F1 {before['macro_f1']:.4f} → {after['macro_f1']:.4f}"
        f"  {result['gain_pp']:+.2f}%p\n"
        f"이긴 묶음 {result['folds_improved']}/{len(diff)}"
        f"  p={result['wilcoxon_p']:.3f}\n"
        f"저장 {run / 'transitions.json'}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sleepstage", description=__doc__)
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("prepare", help="EDF 를 30초 에포크 npz 로")
    p1.add_argument("--config", required=True)
    p1.add_argument("--root", help="설정의 dataset.root 를 덮어씀")
    p1.add_argument("--out", help="설정의 output.dir 을 덮어씀")
    p1.add_argument("--limit", type=int, help="앞에서 N개만 (시험 실행용)")
    p1.add_argument("--jobs", type=int, default=1, help="병렬 프로세스 수")
    p1.add_argument("--overwrite", action="store_true", help="설정이 같아도 다시 만듦")
    p1.set_defaults(func=_cmd_prepare)

    p2 = sub.add_parser("features", help="에포크 npz 에서 특징 표 만들기")
    p2.add_argument("--config", required=True)
    p2.add_argument("--epochs", help="설정의 features.epochs_dir 을 덮어씀")
    p2.add_argument("--out", help="설정의 features.out 을 덮어씀")
    p2.add_argument("--limit", type=int, help="앞에서 N개만 (시험 실행용)")
    p2.add_argument("--jobs", type=int, default=1, help="병렬 프로세스 수")
    p2.add_argument("--overwrite", action="store_true", help="산출물이 있어도 다시 만듦")
    p2.add_argument(
        "--set",
        action="append",
        metavar="키=값",
        help="설정 덮어쓰기. 예: --set features.filter=none (여러 번 쓸 수 있음)",
    )
    p2.set_defaults(func=_cmd_features)

    v = sub.add_parser("verify", help="만들어진 에포크 npz 를 원본과 대조")
    v.add_argument("--config", required=True)
    v.add_argument("--root", help="설정의 dataset.root 를 덮어씀")
    v.add_argument("--epochs", help="설정의 output.dir 을 덮어씀")
    v.add_argument("--samples", type=int, default=3, help="녹음당 신호 대조 에포크 수")
    v.set_defaults(func=_cmd_verify)

    pp = sub.add_parser("feature-path", help="설정이 만드는 특징 표 경로 출력")
    pp.add_argument("--config", required=True)
    pp.add_argument("--set", action="append", metavar="키=값")
    pp.set_defaults(func=_cmd_feature_path)

    p3 = sub.add_parser("split", help="피험자 단위 fold 배정을 파일로 고정")
    p3.add_argument("--features", required=True, help="특징 parquet 경로")
    p3.add_argument("--out", required=True, help="fold JSON 저장 경로")
    p3.add_argument("--folds", type=int, default=10)
    p3.add_argument("--seed", type=int, default=0)
    p3.set_defaults(func=_cmd_split)

    p4 = sub.add_parser("train", help="교차검증으로 학습")
    p4.add_argument("--config", required=True)
    p4.add_argument("--set", action="append", metavar="키=값", help="설정 덮어쓰기")
    p4.add_argument("--overwrite", action="store_true", help="산출물이 있어도 다시 학습")
    p4.set_defaults(func=_cmd_train)

    p8 = sub.add_parser("budget", help="지연 예산별 성능 곡선")
    p8.add_argument("--config", required=True)
    p8.add_argument("--budgets", default="0,1,2,4,10,inf", help="분 단위, inf 는 무제한")
    p8.add_argument("--plot", help="곡선 PNG 저장 경로")
    p8.add_argument("--set", action="append", metavar="키=값")
    p8.set_defaults(func=_cmd_budget)

    p9 = sub.add_parser("export", help="배포용 모델 아티팩트 생성")
    p9.add_argument("--config", required=True)
    p9.add_argument("--out", required=True, help="아티팩트 디렉터리")
    p9.add_argument("--version", default="v1")
    p9.add_argument("--metrics", help="card 에 넣을 metrics.json 경로")
    p9.add_argument("--set", action="append", metavar="키=값")
    p9.set_defaults(func=_cmd_export)

    p10 = sub.add_parser("deep", help="신경망 학습 (딥러닝 비교군)")
    p10.add_argument("--config", required=True)
    p10.add_argument("--fold", type=int, default=0)
    p10.add_argument("--limit", type=int, help="앞에서 N개 녹음만 (시험 실행용)")
    p10.add_argument("--set", action="append", metavar="키=값")
    p10.set_defaults(func=_cmd_deep)

    s = sub.add_parser("serve", help="추론 API 를 띄움")
    s.add_argument("--artifact", default="artifacts/model-v1", help="모델 파일 디렉터리")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=_cmd_serve)

    p11 = sub.add_parser("replay", help="녹음을 조각씩 흘려보내 실시간 판정 재현")
    p11.add_argument("--artifact", required=True, help="모델 파일 디렉터리")
    p11.add_argument("--npz", required=True, help="녹음 파일")
    p11.add_argument("--limit", type=int, help="앞에서 N 조각만")
    p11.add_argument("--compare", help="비교할 특징 parquet 경로")
    p11.set_defaults(func=_cmd_replay)

    p12 = sub.add_parser("postprocess", help="저장된 확률에 단계 전이 후처리를 걸어 다시 채점")
    p12.add_argument("--run", required=True, help="predictions.parquet 이 있는 실행 폴더")
    p12.add_argument(
        "--lag", type=int, default=0, help="판정 전에 볼 뒤쪽 조각 수, 0 은 과거만, 음수는 밤 전체"
    )
    p12.add_argument(
        "--weight", type=float, help="표를 믿는 세기. 비우면 묶음마다 나머지 묶음 안에서 고른다"
    )
    p12.set_defaults(func=_cmd_postprocess)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
