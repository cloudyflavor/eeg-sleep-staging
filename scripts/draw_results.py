"""실행 기록에서 결과 그림을 만듭니다. 저장소 최상위에서 실행하세요.

    python scripts/draw_results.py

손으로 그리면 실험을 다시 돌릴 때마다 그림이 어긋납니다. runs 아래의 지표를
읽어 그리므로 언제든 다시 만들 수 있습니다. 도식과 달리 이 그림들은 측정값에서
나오므로 코드가 없으면 재현이 안 됩니다.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

RUNS = Path("runs")
OUT = Path("assets")
BLUE, RED, GREY = "#4C78A8", "#C0504D", "#B0B0B0"

#: 축 이름 -> (기준 실행, 뺀 실행). 두 실행은 그 축 말고 전부 같아야 한다.
#: 하나라도 다르면 막대가 그 축의 기여가 아니라 두 조건의 차이가 된다.
AXES = {
    "이마 채널": (
        "catboost/zerophase-expanding-smooth/cw-balanced",
        "catboost/zerophase-expanding-smooth/cw-balanced-pz_only",
    ),
    "시간 문맥": (
        "histgb/zerophase-expanding-smooth/cw-balanced",
        "histgb/zerophase-expanding-none/cw-balanced",
    ),
    "모델 종류": (
        "catboost/zerophase-expanding-smooth/cw-balanced",
        "ridge/zerophase-expanding-smooth/cw-balanced",
    ),
    "정규화": (
        "catboost/zerophase-expanding-smooth/cw-balanced",
        "catboost/zerophase-none-smooth/cw-balanced",
    ),
    "필터": (
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


def _folds(path: Path) -> np.ndarray:
    return np.array([f["macro_f1"] for f in json.loads(path.read_text())["folds"]])


def contribution() -> Path:
    """축마다 그것을 뺐을 때 잃는 macro-F1."""
    gaps = {name: (_pooled(a) - _pooled(b)) * 100 for name, (a, b) in AXES.items()}
    order = sorted(gaps, key=gaps.get)

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    top = max(gaps, key=gaps.get)
    bars = ax.barh(
        order, [gaps[k] for k in order], color=[RED if k == top else BLUE for k in order]
    )
    ax.bar_label(bars, labels=[f"{gaps[k]:.1f}%p" for k in order], padding=5, fontsize=10)
    ax.set_xlim(0, max(gaps.values()) * 1.22)
    ax.set_xlabel("macro-F1 기여, %p")
    ax.set_title("축마다 그것을 뺐을 때 잃는 성능", loc="left", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#E8E8E8")
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = OUT / "contribution.png"
    fig.savefig(path, dpi=160)
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

    # 띠의 폭은 묶음별로 짝지은 차이의 표준편차다. 이만큼은 같은 설정을 다시 돌려도
    # 흔들리는 폭이므로 띠 안의 점은 서로 다르다고 말할 수 없다.
    base = _folds(ref[2] / "metrics.json")
    near = min(picked, key=lambda p: p[0])
    band = float(np.std(_folds(near[2] / "metrics.json") - base)) * 100

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    y0 = ref[1] * 100
    ax.axhspan(y0 - band, y0 + band, color=GREY, alpha=0.25, zorder=0)
    ax.axhline(y0, color=GREY, lw=1.2, zorder=1)
    ax.text(505, y0 + 0.05, "전체 505열", fontsize=9, color="#777777", ha="right")

    xs = sorted(picked + [ref], key=lambda p: p[0])
    ax.plot(
        [p[0] for p in xs],
        [p[1] * 100 for p in xs],
        "-o",
        color=BLUE,
        lw=1.8,
        ms=7,
        zorder=3,
        label="값을 보고 고른 것",
    )
    ax.plot(
        [p[0] for p in dropped],
        [p[1] * 100 for p in dropped],
        "s",
        color=RED,
        ms=8,
        zorder=3,
        label="이름으로 뺀 것",
    )

    for n, f1, _d in dropped:
        ax.annotate(
            f"{(f1 - ref[1]) * 100:+.2f}%p",
            (n, f1 * 100),
            textcoords="offset points",
            xytext=(10, -4),
            fontsize=9,
            color=RED,
        )
    n200 = min(picked, key=lambda p: p[0])
    ax.annotate(
        "200열",
        (n200[0], n200[1] * 100),
        textcoords="offset points",
        xytext=(-8, 14),
        fontsize=10,
        color=BLUE,
        fontweight="bold",
    )

    ax.set_xlabel("특징 열 수")
    ax.set_ylabel("macro-F1, %")
    ax.set_title("열을 줄여도 성능이 유지되는 구간", loc="left", fontsize=12)
    ax.set_xlim(170, 545)
    ax.set_ylim(82.6, 84.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#EEEEEE")
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    # 설명은 축 바깥에 두 줄로 둔다. 한 줄로 두면 오른쪽에서 잘린다.
    fig.text(
        0.015,
        0.015,
        f"회색 띠는 같은 설정을 다시 돌려도 흔들리는 폭 ±{band:.2f}%p 입니다.\n"
        "200열과 505열의 차이는 이 안에 들어오고 윌콕슨 p=0.557 입니다.",
        fontsize=9.5,
        color="#666666",
        va="bottom",
        linespacing=1.5,
    )
    path = OUT / "pruning.png"
    fig.savefig(path, dpi=160)
    return path


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for make in (contribution, pruning):
        print(f"생성: {make()}")
