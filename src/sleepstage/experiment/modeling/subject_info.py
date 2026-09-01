"""피험자 자체의 정보를 학습에 넣는다.

나이와 성별은 뇌파에서 뽑은 값이 아니라 사람에 대해 원래 아는 값이다. 그래서
특징 표에 굽지 않고 학습할 때 붙인다. 500MB 짜리 표를 상수 두 열 때문에 다시
만들 이유가 없고, 신호에서 계산한 것과 밖에서 가져온 것을 섞지 않는 편이 낫다.

실측으로 성능이 나이와 크게 엮여 있다. 녹음별 macro-F1 과 나이의 순위상관이
-0.51 이고, 80세 이상은 0.711 인데 40세 미만은 0.868 이다. 깊은 잠 비율이
14.2%에서 2.7%로 줄어드는 것이 그 통로다. 실제 기기에서도 나이는 아는 값이라
특징으로 넣어도 배포 조건을 어기지 않는다.
"""

from pathlib import Path

import numpy as np
import pandas as pd

#: 넣을 수 있는 조합. 이름이 그대로 실행 폴더 이름이 된다.
CHOICES = ("none", "age", "age+sex")

#: 나이를 묶는 경계. 문헌이 쓰는 구간이고 깊은 잠 비율이 이 경계에서 꺾인다.
AGE_EDGES = (0, 40, 60, 80, 200)
AGE_BANDS = ("~40", "40-60", "60-80", "80+")


def load(path: str | Path) -> pd.DataFrame:
    """피험자 표를 읽어 녹음 키에 맞춘다.

    표의 subject 는 0~82 인 번호이고 우리 키는 `SC4XX` 다. 숫자를 두 자리로 채워
    맞춘다. 자리를 안 채우면 한 자리 번호가 통째로 어긋난다.
    """
    df = pd.read_excel(path).rename(columns={"sex (F=1)": "female"})
    df["subject"] = "SC4" + df["subject"].astype(str).str.zfill(2)
    return df[["subject", "night", "age", "female"]]


def band(age: np.ndarray) -> np.ndarray:
    """나이를 구간 번호로 바꾼다."""
    return np.asarray(pd.cut(age, AGE_EDGES, labels=False, right=False))


def attach(table: pd.DataFrame, path: str | Path, mode: str) -> tuple[pd.DataFrame, list[str]]:
    """표에 피험자 정보를 붙이고 그 열 이름을 함께 돌려준다.

    붙는 값이 하나라도 비면 멈춘다. 조용히 결측으로 두면 그 녹음만 다른 조건으로
    학습되고 실행해도 안 보인다.
    """
    if mode not in CHOICES:
        raise ValueError(f"모르는 피험자 정보 설정입니다: {mode!r} ({'/'.join(CHOICES)})")
    if mode == "none":
        return table, []

    info = load(path)
    merged = table.merge(info, on=["subject", "night"], how="left")
    if len(merged) != len(table):
        raise ValueError("피험자 표에 같은 녹음이 여러 줄 있습니다")
    cols = ["age"] if mode == "age" else ["age", "female"]
    missing = merged[cols].isna().any(axis=1)
    if missing.any():
        keys = merged.loc[missing, ["subject", "night"]].drop_duplicates()
        raise ValueError(f"피험자 정보가 없는 녹음이 있습니다: {keys.to_dict('records')[:5]}")
    return merged.astype({c: "float32" for c in cols}), cols


def age_band_weight(table: pd.DataFrame, train_mask: np.ndarray) -> np.ndarray:
    """나이 구간마다 같은 무게가 되도록 조각별 가중치를 만든다.

    지금은 조각 수가 많은 구간이 학습을 지배한다. 성적이 나쁜 고령 구간의 무게를
    올리면 그쪽에 맞춰 배운다. 대신 잘 되던 구간을 잃을 수 있어서 재봐야 한다.
    """
    bands = band(table.loc[train_mask, "age"].to_numpy())
    counts = np.bincount(bands, minlength=len(AGE_BANDS))
    weight = len(bands) / (np.count_nonzero(counts) * np.maximum(counts, 1))
    return weight[bands]
