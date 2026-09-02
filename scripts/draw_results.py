"""실행 기록에서 결과 그림을 만듭니다. 저장소 최상위에서 실행하세요.

    python scripts/draw_results.py

손으로 그리면 실험을 다시 돌릴 때마다 그림이 어긋납니다. runs 아래의 지표를
읽어 그리므로 언제든 다시 만들 수 있습니다. 도식과 달리 이 그림들은 측정값에서
나오므로 코드가 없으면 재현이 안 됩니다.

글꼴과 색은 assets/architecture.svg 와 맞춥니다. Nimbus Sans 가 그 도식이 쓰는
Helvetica 의 대체 글꼴입니다.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.edgecolor": "#C6CED6",
        "axes.labelcolor": "#5C6B7A",
        "xtick.color": "#5C6B7A",
        "ytick.color": "#5C6B7A",
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "axes.labelsize": 10.5,
    }
)

RUNS = Path("runs")
OUT = Path("assets")
#: assets/architecture.svg 의 색을 그대로 쓴다
INK, MUTED, BLUE, RED, GRID = "#1F2933", "#5C6B7A", "#2F6FB5", "#B03A48", "#E9EDF1"

#: 축 이름 -> (기준 실행, 뺀 실행). 두 실행은 그 축 말고 전부 같아야 한다.
#: 하나라도 다르면 막대가 그 축의 기여가 아니라 두 조건의 차이가 된다.
AXES = {
    "EEG Fpz-Cz channel": (
        "catboost/zerophase-expanding-smooth/cw-balanced",
        "catboost/zerophase-expanding-smooth/cw-balanced-pz_only",
    ),
    "3.5 min context": (
        "histgb/zerophase-expanding-smooth/cw-balanced",
        "histgb/zerophase-expanding-none/cw-balanced",
    ),
    "Gradient boosting": (
        "catboost/zerophase-expanding-smooth/cw-balanced",
        "ridge/zerophase-expanding-smooth/cw-balanced",
    ),
    "Normalization": (
        "catboost/zerophase-expanding-smooth/cw-balanced",
        "catboost/zerophase-none-smooth/cw-balanced",
    ),
    "Bandpass filter": (
        "histgb/zerophase-expanding-smooth/cw-balanced",
        "histgb/none-expanding-smooth/cw-balanced",
    ),
}


def _run(rel: str) -> Path:
    """실행 폴더 하나를 찾는다. 여러 개면 어느 것을 쓸지 알 수 없으므로 멈춘다."""
    hits = sorted((RUNS / rel).glob("*/metrics.json"))
    if len(hits) != 1:
        raise SystemExit(f"{rel} 아래 실행이 {len(hits)}개입니다. 하나여야 합니다.")
    return hits[0].parent


def _pooled(rel: str) -> float:
    return json.loads((_run(rel) / "metrics.json").read_text())["pooled"]["macro_f1"]


def _title(ax, text: str) -> None:
    ax.set_title(text, fontsize=13, color=INK, pad=16, fontweight="medium")


def contribution() -> Path:
    """축마다 그것을 뺐을 때 잃는 macro-F1."""
    gaps = {name: (_pooled(a) - _pooled(b)) * 100 for name, (a, b) in AXES.items()}
    order = sorted(gaps, key=gaps.get)
    top = max(gaps, key=gaps.get)

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    bars = ax.barh(
        order,
        [gaps[k] for k in order],
        height=0.62,
        color=[RED if k == top else BLUE for k in order],
    )
    for bar, k in zip(bars, order, strict=True):
        ax.text(
            gaps[k] + 0.13,
            bar.get_y() + bar.get_height() / 2,
            f"{gaps[k]:.1f}",
            va="center",
            fontsize=11,
            color=INK,
        )
    ax.set_xlim(0, max(gaps.values()) * 1.18)
    ax.set_xlabel("macro-F1 drop when removed, %p", labelpad=10)
    _title(ax, "Contribution by design choice")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=3)
    ax.grid(axis="x", color=GRID, lw=1)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = OUT / "contribution.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _pruning_points():
    """전수 평가한 가지치기 실행만 모은다.

    품질 배제를 켠 실행은 평가 행이 달라서 같은 그림에 놓으면 안 된다.
    """
    picked, dropped, ref = [], [], None
    for f in sorted(RUNS.glob("improve/*/*/manifest.json")):
        m = json.loads(f.read_text())
        t = m["train"]
        if m["n_rows"] - m.get("n_dropped_missing", 0) != 195479 or m.get("n_rows_excluded"):
            continue
        if "corr" not in f.parent.parent.name:
            continue
        met = json.loads((f.parent / "metrics.json").read_text())
        kept = met.get("n_kept_per_fold")
        n = min(kept) if kept else m["n_features"]
        point = (n, met["pooled"]["macro_f1"], f.parent)
        if t.get("feature_select", "none") != "none":
            picked.append(point)
        elif t.get("feature_subset", "all") != "all":
            dropped.append(point)
        else:
            ref = point
    return picked, dropped, ref


def pruning() -> Path:
    """열 수를 줄였을 때 성능이 어떻게 되는지."""
    picked, dropped, ref = _pruning_points()
    if ref is None:
        raise SystemExit("기준이 되는 전체 열 실행을 못 찾았습니다")

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    y0 = ref[1] * 100
    ax.axhline(y0, color="#C6CED6", lw=1.1, ls=(0, (6, 4)), zorder=1)

    xs = sorted(picked + [ref], key=lambda p: p[0])
    ax.plot(
        [p[0] for p in xs],
        [p[1] * 100 for p in xs],
        "-o",
        color=BLUE,
        lw=2,
        ms=7.5,
        mec="white",
        mew=1.4,
        zorder=3,
        label="Top-N columns by importance",
    )
    ax.plot(
        [p[0] for p in dropped],
        [p[1] * 100 for p in dropped],
        "s",
        color=RED,
        ms=8,
        mec="white",
        mew=1.2,
        zorder=3,
        label="One feature group removed",
    )
    for n, f1, _d in dropped:
        ax.annotate(
            f"{(f1 - ref[1]) * 100:+.2f}%p",
            (n, f1 * 100),
            textcoords="offset points",
            xytext=(12, -4),
            fontsize=10.5,
            color=RED,
        )
    n200 = min(picked, key=lambda p: p[0])
    ax.annotate(
        "200 columns",
        (n200[0], n200[1] * 100),
        textcoords="offset points",
        xytext=(-4, 15),
        fontsize=11,
        color=BLUE,
        fontweight="bold",
    )

    ax.set_xlabel("Number of feature columns", labelpad=10)
    ax.set_ylabel("macro-F1, %", labelpad=10)
    _title(ax, "Performance after reducing feature columns")
    ax.set_xlim(170, 545)
    ax.set_ylim(82.6, 84.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=10.5, labelcolor=INK)
    fig.tight_layout()
    path = OUT / "pruning.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for make in (contribution, pruning):
        print(f"생성: {make()}")
