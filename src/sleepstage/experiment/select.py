"""학습 행만 보고 쓸 열을 고른다.

이름으로 거르는 ``SUBSETS`` 와 다르다. 저기는 열 이름만 보므로 어느 행을 보든
결과가 같다. 여기는 값을 보고 고르므로 **어느 행을 보느냐가 결과를 바꾼다.**

평가 대상의 값을 보고 고르면 그 점수가 부풀려지고, 실행해도 안 보인다. 그래서
고르는 일은 묶음마다 따로, 그 묶음의 학습 행만으로 한다. 묶음마다 고른 열이
달라지는 것이 정상이다.

방법 이름은 ``top:200`` 처럼 콜론 뒤에 값을 적는다.
"""

import numpy as np

#: 상관을 잴 때 쓸 최대 행 수. 20만 행을 다 쓰면 505x505 를 얻는 데만 오래 걸리고
#: 4만 행이면 상관 추정이 이미 소수점 셋째 자리에서 안정된다.
CORR_ROWS = 40000

#: 순위를 매길 때 쓰는 나무 수. 최종 모델이 아니라 순위만 뽑으므로 적게 둔다.
RANK_TREES = 120


def _rank(X: np.ndarray, y: np.ndarray, sw: np.ndarray | None, threads: int) -> np.ndarray:
    """좋은 열부터 순서대로. 부스팅이 나눌 때 얻은 이득으로 잰다.

    한 번 더 학습하는 값을 치르는 대신, 열 하나씩 따로 보는 방식이 못 잡는
    조합 효과를 잡는다. 수면 단계는 열 하나로는 안 갈리는 경우가 많다.
    """
    from lightgbm import LGBMClassifier

    ranker = LGBMClassifier(
        n_estimators=RANK_TREES,
        learning_rate=0.15,
        num_leaves=31,
        n_jobs=threads,
        random_state=0,
        verbose=-1,
    )
    ranker.fit(X, y, sample_weight=sw)
    gain = ranker.booster_.feature_importance(importance_type="gain")
    # 이득이 같은 열이 여럿이면 열 번호 순으로 끊어 실행마다 같은 답이 나오게 한다
    return np.lexsort((np.arange(len(gain)), -gain))


def _top(X, y, sw, arg: str, threads: int) -> np.ndarray:
    return np.sort(_rank(X, y, sw, threads)[: int(arg)])


def _decorr(X, y, sw, arg: str, threads: int) -> np.ndarray:
    """닮은 열은 하나만 남긴다. 좋은 열부터 자리를 잡고 그 뒤를 쳐낸다.

    순위가 높은 쪽이 먼저 자리를 잡으므로, 한 무리에서 가장 쓸모 있는 열이
    대표로 남는다.
    """
    threshold = float(arg)
    order = _rank(X, y, sw, threads)

    rows = X if len(X) <= CORR_ROWS else X[:: len(X) // CORR_ROWS][:CORR_ROWS]
    # 결측을 그대로 두면 그 열의 상관이 전부 NaN 이 되고, NaN 을 0 으로 바꾸면
    # 닮은 열이 안 닮은 것으로 보여 하나도 안 걷힌다. 가운데값으로 메우고 잰다.
    rows = np.where(np.isnan(rows), np.nanmedian(rows, axis=0), rows)
    corr = np.corrcoef(rows, rowvar=False)
    np.fill_diagonal(corr, 0.0)
    # 값이 하나뿐인 열은 상관이 정의되지 않는다. 그런 열은 아무도 안 막는다.
    corr = np.abs(np.nan_to_num(corr))

    keep, blocked = [], np.zeros(X.shape[1], dtype=bool)
    for i in order:
        if blocked[i]:
            continue
        keep.append(i)
        blocked |= corr[i] > threshold
    return np.sort(np.array(keep))


#: 방법 이름 -> 고르는 함수. 모르는 이름은 예외다.
SELECTORS = {"top": _top, "decorr": _decorr}


def choose(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    sw: np.ndarray | None = None,
    threads: int = 8,
) -> np.ndarray | None:
    """쓸 열의 번호. ``none`` 이면 ``None`` 을 돌려주고 전부 쓴다는 뜻이다."""
    if not name or name == "none":
        return None
    method, _, arg = name.partition(":")
    if method not in SELECTORS or not arg:
        raise ValueError(f"모르는 feature_select: {name!r} (top:200 / decorr:0.95 꼴)")
    return SELECTORS[method](X, y, sw, arg, threads)
