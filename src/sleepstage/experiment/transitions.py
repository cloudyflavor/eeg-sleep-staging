"""앞뒤 조각을 이어서 판정을 다듬는다.

모델은 조각마다 따로 답을 낸다. 그래서 각성 한가운데에 깊은 잠 한 칸이 튀어나와도
그대로 내보낸다. 실제 라벨에서 각성에서 깊은 잠으로 바로 가는 일은 0.00 % 다.

단계가 어떻게 바뀌는지를 표로 만들어 두고 억지스러운 길에 벌점을 준다.

세 가지를 지켜야 한다.

**표는 학습 묶음의 라벨로만 만든다.** 평가 대상의 정답을 보면 점수가 부풀려지고
그건 실행해도 안 보인다.

**뒤를 얼마나 볼지 고를 수 있어야 한다.** 밤 전체를 보면 아침에야 답이 나온다.
``lag`` 를 0 으로 두면 과거만 보므로 지연이 전혀 늘지 않는다.

**세기도 평가 대상 없이 골라야 한다.** 표에는 단계별 흔한 정도가 함께 들어 있어서
그대로 믿으면 흔한 단계로 쏠려 드문 단계를 똑같이 보는 macro-F1 에서 손해를 본다.
좋은 세기를 평가 점수로 고르면 그 점수가 낙관적으로 부풀려지므로, 묶음마다
나머지 묶음 안에서 정하고 그 묶음에 적용한다.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from sleepstage.experiment.split import STAGE_NAMES

N_STAGES = len(STAGE_NAMES)

#: 한 번도 안 나온 전이에 0 을 주면 그 길이 영원히 막힌다. 아주 작은 값을 깐다.
SMOOTHING = 0.1

#: 세기 후보. 1.0 은 표를 그대로 믿는 것이고 실측에서 오히려 손해였다.
WEIGHT_GRID = (0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0)

#: 녹음을 가르는 키. 한 사람의 첫 조각이 다른 사람의 마지막 조각을 이으면 안 된다.
RECORDING = ["subject", "night"]


def transition_log_prob(labels_per_recording) -> np.ndarray:
    """단계가 어떻게 바뀌는지 세어 로그 확률 표로 만든다.

    녹음 하나씩 넘겨야 한 사람의 첫 조각이 다른 사람의 마지막 조각을 잇지 않는다.
    """
    counts = np.full((N_STAGES, N_STAGES), SMOOTHING)
    for labels in labels_per_recording:
        labels = np.asarray(labels)
        if len(labels) > 1:
            np.add.at(counts, (labels[:-1], labels[1:]), 1.0)
    return np.log(counts / counts.sum(axis=1, keepdims=True))


def decode(log_emission: np.ndarray, log_trans: np.ndarray, lag: int | None) -> np.ndarray:
    """조각마다 가장 그럴듯한 단계를 고른다.

    ``lag`` 는 판정하기 전에 볼 뒤쪽 조각 수다. 0 이면 과거만 본다. ``None`` 이면
    밤 전체를 본다. 실시간에서는 이미 기다리는 조각 수를 넘지 않아야 한다.
    """
    n = len(log_emission)
    alpha = np.empty((n, N_STAGES))
    alpha[0] = log_emission[0]
    for t in range(1, n):
        alpha[t] = log_emission[t] + _logsumexp(alpha[t - 1, :, None] + log_trans, axis=0)
    if lag == 0:
        return alpha.argmax(axis=1)

    # 뒤에서 앞으로 한 번 훑어 각 지점의 "여기서부터 끝까지" 를 모아 둔다.
    beta = np.zeros((n, N_STAGES))
    for t in range(n - 2, -1, -1):
        beta[t] = _logsumexp(log_trans + (log_emission[t + 1] + beta[t + 1])[None, :], axis=1)
    if lag is None:
        return (alpha + beta).argmax(axis=1)

    # 뒤를 lag 만큼만 본다. 그 지점의 beta 에서 그 뒤 몫을 빼는 대신 다시 짧게 센다.
    out = np.empty(n, dtype=np.int64)
    for t in range(n):
        tail = np.zeros(N_STAGES)
        for u in range(min(n - 1, t + lag), t, -1):
            tail = _logsumexp(log_trans + (log_emission[u] + tail)[None, :], axis=1)
        out[t] = (alpha[t] + tail).argmax()
    return out


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    peak = values.max(axis=axis, keepdims=True)
    return (peak + np.log(np.exp(values - peak).sum(axis=axis, keepdims=True))).squeeze(axis)


def _prepare(pred: pd.DataFrame):
    """확률을 로그로 바꾸고, 녹음마다 시간 순서인 행 번호를 뽑아 둔다."""
    pred = pred.sort_values([*RECORDING, "epoch_idx"]).reset_index(drop=True)
    proba = pred[[f"proba_{s}" for s in STAGE_NAMES]].to_numpy(dtype="float64")
    log_emission = np.log(np.clip(proba, 1e-12, None))
    groups = [g.index.to_numpy() for _, g in pred.groupby(RECORDING, sort=False)]
    fold_of = pred["fold"].to_numpy()
    return pred, log_emission, groups, np.array([fold_of[g[0]] for g in groups])


def _decode_groups(log_emission, groups, log_trans, weight, lag) -> np.ndarray:
    out = np.concatenate([decode(log_emission[g], weight * log_trans, lag) for g in groups])
    return out


def apply(
    pred: pd.DataFrame,
    lag: int | None = 0,
    weight: float | None = None,
    grid=WEIGHT_GRID,
) -> tuple[pd.DataFrame, dict]:
    """저장해 둔 확률에 후처리를 걸어 새 판정을 낸다.

    ``weight`` 를 주면 그 세기를 그대로 쓰고, 비우면 묶음마다 나머지 묶음 안에서
    고른다. 어느 쪽이든 표는 그 묶음의 평가 대상을 뺀 라벨로만 만든다.
    """
    pred, log_emission, groups, group_fold = _prepare(pred)
    y = pred["stage"].to_numpy()
    out = np.empty(len(pred), dtype=np.int64)
    chosen: dict[int, float] = {}

    for fold in sorted(set(group_fold.tolist())):
        inner = [g for g, f in zip(groups, group_fold, strict=True) if f != fold]
        outer = [g for g, f in zip(groups, group_fold, strict=True) if f == fold]
        log_trans = transition_log_prob(y[g] for g in inner)

        w = weight
        if w is None:
            truth = np.concatenate([y[g] for g in inner])
            scores = [
                f1_score(
                    truth,
                    _decode_groups(log_emission, inner, log_trans, cand, lag),
                    average="macro",
                    zero_division=0,
                )
                for cand in grid
            ]
            w = grid[int(np.argmax(scores))]
        chosen[int(fold)] = float(w)

        for g in outer:
            out[g] = decode(log_emission[g], w * log_trans, lag)

    pred = pred.copy()
    pred["pred_smoothed"] = out.astype(np.int8)
    return pred, {"lag": lag, "weights": chosen}


def score(pred: pd.DataFrame, column: str) -> dict[str, float]:
    """묶음별 macro-F1 과 전체 macro-F1."""
    per_fold = [
        float(f1_score(g["stage"], g[column], average="macro", zero_division=0))
        for _, g in pred.groupby("fold")
    ]
    return {
        "macro_f1": float(f1_score(pred["stage"], pred[column], average="macro", zero_division=0)),
        "folds": per_fold,
    }
