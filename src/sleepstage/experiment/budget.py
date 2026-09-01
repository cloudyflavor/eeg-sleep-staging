"""판정을 얼마나 늦게 내려도 되는지에 따라 성능이 어떻게 달라지는지 잰다.

일부 특징은 뒤쪽 데이터가 있어야 계산할 수 있다. 즉시 답해야 하면 그런 특징을
쓸 수 없고, 몇 분 기다릴 수 있으면 쓸 수 있다.

허용 시간별로 쓸 수 있는 특징만 골라 학습해서, 기다리는 시간과 성능의 관계를
곡선으로 만든다.
"""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from sleepstage.config import Config
from sleepstage.experiment.split import fold_masks, load_folds
from sleepstage.experiment.train import _fit, _metrics, features_file, load_table, make_model

#: 에포크 하나가 몇 분인지. 허용 시간을 에포크 수로 바꿀 때 쓴다
EPOCH_MINUTES = 0.5


def lookahead_table(features_path: Path) -> dict[str, int]:
    side = features_path.with_suffix(".manifest.json")
    return json.loads(side.read_text(encoding="utf-8"))["lookahead_epochs"]


def columns_within(lookahead: dict[str, int], budget_min: float | None) -> list[str]:
    """허용 시간 안에 계산할 수 있는 열만 고른다. ``None`` 이면 제한을 두지 않는다."""
    if budget_min is None:
        return list(lookahead)
    allowed = budget_min / EPOCH_MINUTES
    return [c for c, look in lookahead.items() if look <= allowed]


def run_budgets(
    cfg: Config,
    budgets: list[float | None],
    out_name: str = "budget_curve.json",
) -> dict[str, Any]:
    """허용 시간별로 학습하고 평가해 곡선 데이터를 만든다."""
    p = cfg["train"]
    features_path = features_file(p)
    table, _ = load_table(features_path, p["drop_missing"])
    folds = load_folds(p["splits"])
    lookahead = lookahead_table(features_path)
    y = table["stage"].to_numpy()

    points = []
    for budget in budgets:
        cols = [c for c in columns_within(lookahead, budget) if c in table.columns]
        if not cols:
            raise ValueError(f"예산 {budget}분에 쓸 수 있는 열이 없습니다")
        X = table[cols].to_numpy(dtype=np.float32)

        t0 = time.perf_counter()
        trues, preds = [], []
        for fold in folds:
            train_mask, test_mask = fold_masks(table, fold)
            sw = None
            if p["class_weight"] == "balanced":
                from sklearn.utils.class_weight import compute_sample_weight

                sw = compute_sample_weight("balanced", y[train_mask])
            model = _fit(make_model(cfg), X[train_mask], y[train_mask], sw)
            trues.append(y[test_mask])
            preds.append(np.asarray(model.predict(X[test_mask])).ravel().astype(int))

        m = _metrics(np.concatenate(trues), np.concatenate(preds))
        label = "제한 없음" if budget is None else f"{budget}분"
        points.append(
            {
                "budget_min": budget,
                "n_features": len(cols),
                **m,
                "sec": round(time.perf_counter() - t0, 1),
            }
        )
        print(f"  예산 {label:>6}  열 {len(cols):>3}  macro-F1 {m['macro_f1']:.4f}", flush=True)

    out = Path(p["out_dir"]) / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"features_file": features_path.name, "points": points}, indent=2),
        encoding="utf-8",
    )
    return {"points": points, "out": str(out)}


def plot_curve(curve_json: str, out_png: str) -> str:
    """허용 시간과 성능의 관계를 그림으로 그린다."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = json.loads(Path(curve_json).read_text(encoding="utf-8"))["points"]
    finite = [p for p in pts if p["budget_min"] is not None]
    unlimited = [p for p in pts if p["budget_min"] is None]

    x = [p["budget_min"] for p in finite]
    y = [p["macro_f1"] for p in finite]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.plot(x, y, marker="o", color="black", linewidth=1.2, markersize=5)
    if unlimited:
        ax.axhline(unlimited[0]["macro_f1"], color="black", linestyle="--", linewidth=1)
        ax.text(x[-1], unlimited[0]["macro_f1"], " post hoc", va="bottom", ha="right", fontsize=9)
    for p in finite:
        ax.annotate(
            f"{p['n_features']}",
            (p["budget_min"], p["macro_f1"]),
            textcoords="offset points",
            xytext=(0, -14),
            ha="center",
            fontsize=8,
        )
    ax.set_xlabel("delay budget (min)")
    ax.set_ylabel("macro-F1")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, facecolor="white", bbox_inches="tight")
    return out_png
