"""실행 결과를 어디에 저장할지 정한다.

폴더 계층만 보고 무엇을 돌렸는지 알 수 있게 축마다 한 단계씩 나눈다.
같은 모델끼리, 같은 전처리끼리 한자리에 모여서 비교하기 쉽다.

    runs/catboost/zerophase-expanding-smooth/cw-none/fce74f/
    runs/lstm/zerophase-expanding-win15/cw-balanced/a91c02/fold0/

기준선에 무언가를 더해 보는 실험은 따로 모은다. 고정된 조건이 길어서 축으로 나누면
칸마다 같은 말이 반복되고, 정작 무엇을 더했는지가 묻힌다.

    runs/improve/mean+spindles/91c0aa/

마지막 칸은 설정 전체를 요약한 값이다. 난수 씨앗이나 학습률처럼 폴더 이름에
담기지 않는 설정만 바뀌었을 때 앞 결과를 덮어쓰지 않게 해준다.
"""

from pathlib import Path

#: 폴더 이름에 쓸 요약값 길이
ID_LENGTH = 6


#: 개선 단계 실행을 모으는 자리. 축을 훑던 탐색 단계와 섞이지 않게 한다.
IMPROVE = "improve"

#: 기본값에서 벗어났을 때만 이름에 쓰는 설정. 고정된 조건은 실행 기록에 적는다.
#: 이름에 조건을 다 적으면 길어지기만 하고 정작 무엇을 더했는지가 안 보인다.
ADDITIONS = (
    ("welch_average", "median", "mean"),
    ("window_extremes", False, "extremes"),
    ("spindle_events", False, "spindles"),
    ("channel_correlation", False, "corr"),
)


def run_dir(root, model: str, preprocess: str, options: str, run_id: str, added: str = "") -> Path:
    """탐색 단계는 축마다 한 칸씩, 개선 단계는 더한 것 한 칸으로 나눈다."""
    if added:
        return Path(root) / IMPROVE / added / run_id[:ID_LENGTH]
    return Path(root) / model / preprocess / options / run_id[:ID_LENGTH]


#: 학습 쪽에서 더하는 것. 값이 곧 이름인 경우는 tag 를 None 으로 둔다.
TRAIN_ADDITIONS = (
    ("subject_info", "none", None),
    ("age_weight", False, "ageweight"),
)


def addition_slug(settings: dict, train: dict | None = None) -> str:
    """기준선에서 더한 것만 이어 붙인다. 더한 게 없으면 빈 문자열."""
    parts = [tag for key, default, tag in ADDITIONS if settings.get(key, default) != default]
    for key, default, tag in TRAIN_ADDITIONS:
        value = (train or {}).get(key, default)
        if value != default:
            parts.append(tag or str(value))
    return "+".join(parts)


def label(run: Path) -> str:
    """폴더 계층을 한 줄로 되돌린다. 기록 도구는 이름 하나만 받는다."""
    return "/".join(run.parts[-4:])


def preprocess_slug(settings: dict, window: int | None = None) -> str:
    """전처리 설정을 한 칸으로 줄인다. 신경망은 맥락 자리에 창 크기를 쓴다."""
    context = f"win{window}" if window is not None else settings.get("context", "?")
    return f"{settings.get('filter', '?')}-{settings.get('normalize', '?')}-{context}"


def options_slug(class_weight: str, subset: str | None = None) -> str:
    slug = f"cw-{class_weight}"
    return slug if not subset or subset == "all" else f"{slug}-{subset}"
