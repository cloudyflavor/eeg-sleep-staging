"""단일 진입점 CLI.

U-Time 의 ``ut`` 처럼 하위 명령을 하나의 실행 파일 아래 모은다.
설치 후 ``sleepstage <명령>`` 으로 실행된다.

명령은 **구현된 것만 등록한다.** 아직 없는 단계를 자리만 잡아 두면
"구현되지 않았습니다" 를 출력하는 코드가 늘 뿐이고, 등록하지 않으면
argparse 가 알아서 알려준다.
"""

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

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
