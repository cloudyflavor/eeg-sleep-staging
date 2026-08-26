"""단일 진입점 CLI. 구현된 명령만 등록한다."""

import argparse

import yaml

from sleepstage import __version__
from sleepstage.config import Config


def _cmd_config(args: argparse.Namespace) -> int:
    """설정 파일이 어떤 해시를 만드는지 미리 확인한다."""
    cfg = Config.load(args.path)
    print(f"출처 : {cfg.path}")
    print(f"해시 : {cfg.hash}\n")
    print(yaml.safe_dump(cfg.data, allow_unicode=True, sort_keys=False))
    return 0


def _cmd_prepare(args: argparse.Namespace) -> int:
    """1단계 — EDF → 에포크 npz."""
    from sleepstage.preprocess.prepare import prepare_all

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
    """2단계 — 에포크 npz → 특징 parquet."""
    from sleepstage.features.extract import extract_all, settings_hash

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


def _cmd_verify(args: argparse.Namespace) -> int:
    """1단계 산출물을 원본 EDF 와 대조한다."""
    from sleepstage.preprocess.verify import verify_all

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
    """3단계 — 피험자 단위 fold 배정을 파일로 고정."""
    from sleepstage.evaluation.split import write_folds

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
    """4단계 — 학습 + 교차검증."""
    from sleepstage.evaluation.train import run_cv

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


def _cmd_curve(args: argparse.Namespace) -> int:
    """러닝 커브 — 피험자 수를 늘려가며 학습이 되는지 본다."""
    from sleepstage.evaluation.train import learning_curve

    cfg = Config.load(args.config).override(args.set or [])
    counts = [int(x) for x in args.counts.split(",")]
    r = learning_curve(cfg, counts, n_eval_folds=args.eval_folds)
    print(f"\n저장 {r['out']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sleepstage", description=__doc__)
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("config", help="설정 파일 확인 및 해시 출력")
    c.add_argument("path", help="YAML 경로")
    c.set_defaults(func=_cmd_config)

    p1 = sub.add_parser("prepare", help="EDF → 에포크 npz (1단계)")
    p1.add_argument("--config", required=True)
    p1.add_argument("--root", help="설정의 dataset.root 를 덮어씀")
    p1.add_argument("--out", help="설정의 output.dir 을 덮어씀")
    p1.add_argument("--limit", type=int, help="앞에서 N개만 (시험 실행용)")
    p1.add_argument("--jobs", type=int, default=1, help="병렬 프로세스 수")
    p1.add_argument("--overwrite", action="store_true", help="설정이 같아도 다시 만듦")
    p1.set_defaults(func=_cmd_prepare)

    p2 = sub.add_parser("features", help="에포크 npz → 특징 parquet (2단계)")
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

    v = sub.add_parser("verify", help="에포크 npz 를 원본 EDF 와 대조 (1단계 검증)")
    v.add_argument("--config", required=True)
    v.add_argument("--root", help="설정의 dataset.root 를 덮어씀")
    v.add_argument("--epochs", help="설정의 output.dir 을 덮어씀")
    v.add_argument("--samples", type=int, default=3, help="녹음당 신호 대조 에포크 수")
    v.set_defaults(func=_cmd_verify)

    p3 = sub.add_parser("split", help="피험자 단위 fold 배정을 파일로 고정 (3단계)")
    p3.add_argument("--features", required=True, help="특징 parquet 경로")
    p3.add_argument("--out", required=True, help="fold JSON 저장 경로")
    p3.add_argument("--folds", type=int, default=10)
    p3.add_argument("--seed", type=int, default=0)
    p3.set_defaults(func=_cmd_split)

    p4 = sub.add_parser("train", help="학습 + 교차검증 (4단계)")
    p4.add_argument("--config", required=True)
    p4.add_argument("--set", action="append", metavar="키=값", help="설정 덮어쓰기")
    p4.add_argument("--overwrite", action="store_true", help="산출물이 있어도 다시 학습")
    p4.set_defaults(func=_cmd_train)

    p5 = sub.add_parser("curve", help="러닝 커브 (피험자 수 대비 성능)")
    p5.add_argument("--config", required=True)
    p5.add_argument("--counts", default="8,16,24,32,40,48,56,70")
    p5.add_argument("--eval-folds", type=int, default=3)
    p5.add_argument("--set", action="append", metavar="키=값", help="설정 덮어쓰기")
    p5.set_defaults(func=_cmd_curve)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
