"""특징 계산 코드가 내는 값의 지문을 만듭니다.

산출물 이름은 설정만 보고 정해집니다. 그래서 코드를 고쳐 값이 달라져도 이름이
같고, 이미 있는 파일을 그대로 재사용하게 됩니다. 고친 줄 알고 옛 값으로 계속
실험하는 사고가 여기서 납니다.

파일 내용을 해시하면 주석 한 줄, 상수 하나만 옮겨도 지문이 바뀌어 멀쩡한
산출물을 전부 의심하게 됩니다. 대신 정해진 가짜 신호를 실제 계산 경로에 통과시켜
나온 값을 해시합니다. 값이 같으면 코드를 어떻게 고쳤든 같은 지문이고, 값이
달라지면 반드시 다른 지문입니다.

**설정마다 따로 냅니다.** 안 쓰는 계산을 새로 더했을 때 그것을 안 쓰는 산출물까지
의심하게 되면, 경고가 잦아져서 결국 아무도 안 봅니다.
"""

import hashlib

import numpy as np

LENGTH = 6

#: 가짜 신호의 길이와 조각 수. 정규화와 문맥 창이 채워질 만큼은 있어야 한다.
_N_EPOCHS = 40
_SFREQ = 100.0
_CHANNELS = ["EEG Fpz-Cz", "EEG Pz-Oz"]

#: 설정이 안 주어졌을 때 쓰는 값. 지문은 설정마다 따로 난다.
_DEFAULT_SETTINGS = {
    "filter": "zerophase",
    "normalize": "expanding",
    "context": "smooth",
    "distance_features": True,
    "welch_average": "median",
    "window_extremes": False,
    "spindle_events": False,
    "channel_correlation": False,
}

#: 반올림 자릿수. 플랫폼마다 마지막 자리 부동소수점이 달라 그대로 해시하면
#: 같은 코드가 맥과 서버에서 다른 지문을 낸다.
_DECIMALS = 5


def _synthetic_epochs() -> np.ndarray:
    """수면 뇌파를 흉내 낸 고정 신호. 대역마다 성분이 있어야 모든 특징이 0 이 아니다."""
    rng = np.random.default_rng(20260827)
    t = np.arange(_N_EPOCHS * int(30 * _SFREQ)) / _SFREQ
    base = (
        40 * np.sin(2 * np.pi * 1.0 * t)
        + 15 * np.sin(2 * np.pi * 6.0 * t)
        + 10 * np.sin(2 * np.pi * 13.0 * t) * (np.sin(2 * np.pi * 0.05 * t) > 0.6)
        + 5 * np.sin(2 * np.pi * 22.0 * t)
    )
    sig = np.stack([base + rng.normal(0, 4, t.size), 0.6 * base + rng.normal(0, 4, t.size)])
    return sig.reshape(2, _N_EPOCHS, -1).transpose(1, 0, 2).astype("float32")


def code_fingerprint(settings: dict | None = None) -> str:
    """그 설정으로 가짜 신호를 계산해 나온 값을 짧은 문자열로 요약합니다."""
    from sleepstage.core import context, filters, normalize
    from sleepstage.experiment.extract import epoch_features

    s = dict(_DEFAULT_SETTINGS, **(settings or {}))
    epochs = _synthetic_epochs()
    n_ch = epochs.shape[1]
    joined = epochs.transpose(1, 0, 2).reshape(n_ch, -1)
    filtered = filters.apply_filter(joined, _SFREQ, s["filter"])
    epochs = filtered.reshape(n_ch, _N_EPOCHS, -1).transpose(1, 0, 2)

    raw = epoch_features(
        epochs,
        _SFREQ,
        _CHANNELS,
        with_distance=s["distance_features"],
        welch_average=s["welch_average"],
        window_extremes=s["window_extremes"],
        spindle_events=s["spindle_events"],
        channel_correlation=s["channel_correlation"],
    )
    frames = [raw]
    if s["normalize"] != "none":
        fn, _ = normalize.NORMALIZERS[s["normalize"]]
        frames.append(fn(raw).add_suffix("_norm"))
    # 문맥은 원본에만 건다. 실제 파이프라인은 원본과 정규화를 합친 것에 걸지만,
    # 문맥 함수는 열 종류를 가리지 않고 똑같이 다루므로 원본만으로도 그 함수가
    # 바뀌면 값이 바뀐다. 여기를 넓히면 이미 만든 산출물의 지문이 전부 어긋난다.
    builder = context.CONTEXTS[s["context"]]
    if builder is not None:
        frames.append(builder(raw, np.arange(_N_EPOCHS))[0])

    digest = hashlib.sha256()
    for frame in frames:
        digest.update(",".join(frame.columns).encode())
        values = np.nan_to_num(frame.to_numpy(dtype="float64"), nan=-1.0)
        digest.update(np.round(values, _DECIMALS).tobytes())
    return digest.hexdigest()[:LENGTH]


def check_or_explain(recorded: str | None, current: str, path) -> None:
    """기록된 지문과 지금 지문이 다르면 무엇을 해야 하는지 알려주고 멈춥니다."""
    if recorded == current:
        return
    # 지문을 넣기 전에 만든 산출물이다. 같은 코드인지 알 방법이 없으므로
    # 멈추지는 않되 조용히 넘어가지도 않는다.
    if recorded is None:
        print(f"주의 — 지문이 없는 산출물을 재사용합니다: {path}")
        return
    raise SystemExit(
        f"\n이미 있는 산출물이 지금과 다른 값을 내는 코드로 만들어졌습니다\n"
        f"  파일        {path}\n"
        f"  기록된 지문  {recorded}\n"
        f"  지금 지문    {current}\n"
        f"설정이 같아서 이름은 같지만 값은 다릅니다.\n"
        f"다시 만들려면 --overwrite 를 붙이세요.\n"
    )
