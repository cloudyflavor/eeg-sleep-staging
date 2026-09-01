"""실행 기록에서 결과 표를 만듭니다.

손으로 적으면 실험을 다시 돌릴 때마다 표가 어긋납니다. runs 아래를 읽어
markdown 표로 출력하므로 언제든 다시 만들 수 있습니다.

    python scripts/experiment_table.py > docs/08-results.md
    python scripts/experiment_table.py --readme README.md
"""

import json
from pathlib import Path

RUNS = Path("runs")
STAGES = ["W", "LS", "DS", "R"]
COLS = ["macro_f1", "accuracy", "kappa", *[f"f1_{s}" for s in STAGES]]
HEAD = "macro-F1 | 정확도 | kappa | W | LS | DS | R"


def fmt(m):
    return " | ".join(f"{m[c]:.3f}" for c in COLS)


def tree_runs():
    """축을 훑은 탐색 단계 실행을 읽는다."""
    rows = []
    for path in sorted(RUNS.glob("*/*/*/*/metrics.json")):
        model, preprocess, options, _ = path.parent.relative_to(RUNS).parts
        man = path.parent / "manifest.json"
        info = json.loads(man.read_text()) if man.exists() else {}
        rows.append(
            {
                "model": model,
                "preprocess": preprocess,
                "options": options,
                "n_features": info.get("n_features"),
                "n_rows": info.get("n_rows", 0),
                "n_dropped_missing": info.get("n_dropped_missing", 0),
                **json.loads(path.read_text())["pooled"],
            }
        )
    return rows


def improve_runs():
    """기준선에 무언가를 더해 본 개선 단계 실행을 읽는다."""
    rows = []
    for path in sorted(RUNS.glob("improve/*/*/metrics.json")):
        man = json.loads((path.parent / "manifest.json").read_text())
        rows.append(
            {
                "added": path.parent.parent.name,
                "n_features": man["n_features"],
                "fixed": man.get("condition", {}).get("fixed", {}),
                "n_rows": man.get("n_rows", 0),
                "n_dropped_missing": man.get("n_dropped_missing", 0),
                **json.loads(path.read_text())["pooled"],
            }
        )
    return rows


def deep_runs():
    rows = []
    for path in sorted(RUNS.glob("*/*/*/*/summary.json")):
        model, preprocess, options, _ = path.parent.relative_to(RUNS).parts
        blob = json.loads(path.read_text())
        rows.append(
            {
                "model": model,
                "window": int(preprocess.split("win")[1]),
                "options": options,
                "folds": blob["n_folds"],
                **blob["mean"],
            }
        )
    return rows


def curve(name):
    path = RUNS / "curves" / name
    return json.loads(path.read_text())["points"] if path.exists() else []


def section(title, note=""):
    print(f"\n## {title}\n")
    if note:
        print(note + "\n")


print("# 실험 결과")
print()
print("`scripts/experiment_table.py` 로 다시 만듭니다. 손으로 고치지 마세요.")
print()
print("기준선은 한 클래스만 답하는 경우이고 macro-F1 0.126 입니다.")

tree = tree_runs()
deep = deep_runs()

section(
    "1. 전처리 축",
    "필터 3가지와 문맥 3가지를 조합했습니다. 모델은 histgb 로 고정해서\n"
    "차이가 전처리에서만 오게 했습니다.",
)
print(f"| 필터 | 정규화 | 문맥 | 열 수 | {HEAD} |")
print("|---|---|---|---:|" + "---:|" * 7)
for r in sorted((r for r in tree if r["model"] == "histgb"), key=lambda r: -r["macro_f1"]):
    f, n, c = r["preprocess"].split("-")
    print(f"| {f} | {n} | {c} | {r['n_features']} | {fmt(r)} |")

section(
    "2. 모델 축",
    "전처리를 앞뒤 보는 필터와 앞뒤 평균으로 고정하고 모델만 바꿨습니다.",
)
print(f"| 모델 | 전처리 | 옵션 | {HEAD} |")
print("|---|---|---|" + "---:|" * 7)
for r in sorted(
    (r for r in tree if r["model"] not in ("histgb", "cnn", "lstm")),
    key=lambda r: -r["macro_f1"],
):
    print(f"| {r['model']} | {r['preprocess']} | {r['options']} | {fmt(r)} |")


cat = [r for r in tree if r["model"] == "catboost"]

section(
    "3. 특징 제거",
    "catboost 를 고정하고 채널과 특징군과 정규화를 각각 뺐습니다.",
)
print(f"| 뺀 것 | {HEAD} |")
print("|---|" + "---:|" * 7)
for r in sorted(cat, key=lambda r: -r["macro_f1"]):
    what = r["options"].replace("cw-balanced", "").strip("-") or "없음"
    if r["preprocess"].split("-")[1] == "none":
        what = "정규화"
    print(f"| {what} | {fmt(r)} |")

improve = improve_runs()
base = next((r for r in cat if r["options"] == "cw-balanced" and "smooth" in r["preprocess"]), None)
if improve and base:
    fixed = improve[0]["fixed"]
    section(
        "3-1. 특징 추가",
        "기준선에 하나씩 더했습니다. 한꺼번에 넣으면 어느 것이 기여했는지 알 수 없습니다.\n\n"
        "고정한 조건은 " + " / ".join(f"{k} {v}" for k, v in fixed.items()) + " 입니다.\n\n"
        "mean 은 5초 조각을 평균으로 합친 것, extremes 는 조각별 최댓값과 최솟값,\n"
        "spindles 는 수면방추 개수와 지속시간입니다.",
    )
    print(f"| 더한 것 | 열 수 | {HEAD} | 기준선 대비 |")
    print("|---|---:|" + "---:|" * 8)
    print(f"| 없음 | {base['n_features']} | {fmt(base)} | — |")
    for r in sorted(improve, key=lambda r: r["n_features"]):
        gap = (r["macro_f1"] - base["macro_f1"]) * 100
        print(f"| {r['added']} | {r['n_features']} | {fmt(r)} | {gap:+.2f}%p |")

section("4. 신경망 맥락 창 크기", "창 1 은 조각 하나만 보는 경우입니다.")
if deep:
    print(f"| 모델 | 창 | 지연 | 묶음 | {HEAD} |")
    print("|---|---:|---:|---:|" + "---:|" * 7)
    for r in sorted(deep, key=lambda r: r["window"]):
        lag = (r["window"] // 2) * 0.5
        print(f"| {r['model']} | {r['window']} | {lag:g}분 | {r['folds']} | {fmt(r)} |")
else:
    print("아직 결과가 없습니다.")

section(
    "5. 허용 지연별 성능",
    "특징마다 필요한 미래 시간을 기록해 두고 걸러냅니다.\n"
    "가장 멀리 보는 특징이 3.5분이라 4분부터는 모든 열이 살아남고 값이 같습니다.",
)
pts = curve("budget_curve.json")
if pts:
    print(f"| 허용 지연 | 쓸 수 있는 열 | {HEAD} |")
    print("|---|---:|" + "---:|" * 7)
    for p in pts:
        label = "제한 없음" if p["budget_min"] is None else f"{p['budget_min']:g}분"
        print(f"| {label} | {p['n_features']} | {fmt(p)} |")
else:
    print("아직 결과가 없습니다.")

print(f"\n---\n\n탐색 {len(tree)}건, 개선 {len(improve)}건, 신경망 {len(deep)}건.")


README_START = (
    "<!-- 결과: scripts/experiment_table.py --readme 가 만듭니다. 손으로 고치지 마세요 -->"
)
README_END = "<!-- 결과 끝 -->"


def readme_block() -> str:
    """README 에 넣을 요약. 손으로 적으면 실행이 바뀔 때마다 어긋난다.

    평가 행 수를 함께 낸다. 결측을 버린 실행과 안 버린 실행이 나란히 놓이면
    다른 시험지에서 받은 점수를 비교하게 되는데, 그건 표만 봐서는 안 보인다.
    """

    def rows_of(entry):
        return entry.get("n_rows", 0) - entry.get("n_dropped_missing", 0)

    best = {}
    for r in tree_runs():
        if r["options"] != "cw-balanced":
            continue  # 열을 뺀 실행은 모델 비교가 아니다
        keep = best.get(r["model"])
        if keep is None or r["macro_f1"] > keep["macro_f1"]:
            best[r["model"]] = r

    lines = [
        README_START,
        "",
        f"| | {HEAD} | 평가 조각 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    top = max(improve_runs(), key=lambda r: r["macro_f1"], default=None)
    if top:
        cells = " | ".join(f"**{v}**" for v in fmt(top).split(" | "))
        lines.append(f"| **특징을 더한 catboost** | {cells} | {rows_of(top):,} |")
    for name, r in sorted(best.items(), key=lambda kv: -kv[1]["macro_f1"]):
        if name == "majority":
            continue
        lines.append(f"| {name} | {fmt(r)} | {rows_of(r):,} |")
    if "majority" in best:
        m = best["majority"]
        lines.append(f"| 기준선, 한 클래스만 답하기 | {fmt(m)} | {rows_of(m):,} |")
    lines += ["", README_END]
    return "\n".join(lines)


def write_readme(path: Path) -> None:
    """README 안의 표시 구간만 바꿔 쓴다. 실행 기록이 없으면 건드리지 않는다."""
    if not tree_runs():
        raise SystemExit(f"runs 아래에 실행이 없어 {path} 를 고치지 않습니다")
    text = path.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        raise SystemExit(f"{path} 에 표시가 없습니다: {README_START}")
    head, rest = text.split(README_START, 1)
    _, tail = rest.split(README_END, 1)
    path.write_text(head + readme_block() + tail, encoding="utf-8")
    print(f"{path} 의 결과 표를 다시 만들었습니다")


if __name__ == "__main__":
    import sys

    if "--readme" in sys.argv:
        write_readme(Path(sys.argv[sys.argv.index("--readme") + 1]))
