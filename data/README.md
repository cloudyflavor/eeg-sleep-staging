# Data

## Dataset

**Sleep-EDF Expanded 의 Sleep Cassette** 부분입니다. 집에서 24시간 연속으로 잰 건강한
성인의 수면다원검사입니다.

> [Sleep-EDF Database Expanded, version 1.0.0](https://physionet.org/content/sleep-edfx/1.0.0/)
> PhysioNet, Open Data Commons Attribution License v1.0

| 항목 | 값 |
|---|---|
| 버전 | 1.0.0 |
| 원본 크기 | 8.2 GB |
| 피험자 | 78명, 25세부터 101세 |
| 녹음 | 153개, 75명이 2박씩 |
| 파일 | PSG 153 + Hypnogram 153 |
| 채널 | EEG Fpz-Cz, EEG Pz-Oz 두 개만 사용 |
| 샘플링 | 100 Hz |
| 조각 | 30초 |
| 판독 | 30초마다 R&K 기준 7단계 |

같은 데이터셋의 Sleep Telemetry 부분은 약물 연구 설계라 쓰지 않습니다.

## Download

```bash
mkdir -p data/raw/edfx && cd data/raw/edfx
wget -r -N -c -np https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/
wget -N https://physionet.org/files/sleep-edfx/1.0.0/SC-subjects.xls \
     -P physionet.org/files/sleep-edfx/1.0.0/
```

받고 나면 이 모양이어야 합니다.

```
data/raw/edfx/physionet.org/files/sleep-edfx/1.0.0/
├── sleep-cassette/
│   ├── SC4001E0-PSG.edf
│   ├── SC4001EC-Hypnogram.edf
│   └── ... 306개
└── SC-subjects.xls          나이와 성별
```

**PSG 와 Hypnogram 은 앞 6자가 같고 일곱 번째 글자만 다릅니다.** `SC4001E` 까지가
공통이고 `0` 이면 신호, `C` 면 판독입니다. 코드가 이 규칙으로 짝을 찾습니다.

## Pipeline Outputs

받은 원본은 그대로 두고 아래 폴더가 차례로 생깁니다.

| 폴더 | 무엇 | 크기 |
|---|---|---|
| `data/interim/epochs/` | 30초 조각 npz, 녹음당 하나 | 9.3 GB |
| `data/features/` | 특징 표 parquet, 설정마다 하나 | 표당 300~800 MB |
| `data/splits/` | 사람을 10묶음으로 나눈 결과 | 24 KB |

## Label Mapping

판독은 7단계인데 학습은 4단계로 씁니다.

| 원본 | 합친 뒤 |
|---|---|
| W | **W** (각성) |
| Stage 1, Stage 2 | **LS** (얕은잠) |
| Stage 3, Stage 4 | **DS** (깊은잠) |
| R | **R** (렘수면) |
| `?`, Movement time | 버림 |

**이 프로젝트가 겨냥한 개입이 네 단계만 구분하면 되기 때문입니다.** 서파에 맞춘 청각
자극은 지금이 깊은잠인지만, 악몽 장애 기기와 렘수면 행동장애 감시는 렘수면인지만 알면
됩니다. N1 과 N2 를 갈라도 그 개입이 달라지지 않습니다.

**앞선 논문과 같은 기준이라는 점도 있습니다.** 성능을 나란히 놓으려면 단계 체계가
같아야 합니다.

**대신 N1 과 N2 를 합치면 이 분야에서 가장 어려운 경계가 사라집니다.** 그래서 여기
수치는 5단계 문헌과 직접 비교할 수 없습니다.

## Wake Cropping

집에서 24시간을 계속 녹음해서 **절반 이상이 낮에 깨어 있는 구간**입니다. 실제 기기는
자기 전에 쓰고 아침에 벗으므로 잠든 구간 앞뒤 30분만 남깁니다.

| | 자르기 전 | 자른 뒤 |
|---|---:|---:|
| 조각 수 | 414,961 | 195,479 |
| W | 68.8 % | 33.7 % |
| LS | 21.9 % | 46.4 % |
| DS | 3.1 % | 6.7 % |
| R | 6.2 % | 13.2 % |

**53%가 사라지고 전부 W 입니다.** 자른 뒤의 다수 클래스는 LS 의 46.4%이고, 한 클래스만
답할 때 macro-F1 은 0.126 입니다.

**자르기는 되돌릴 수 있게 해 뒀습니다.** 조각을 만드는 단계에서는 경계만 기록하고,
실제로 자르는 것은 특징을 만들 때입니다.

## Quality Exclusion

`sleepstage quality` 가 녹음마다 채널별 진폭을 재고, **한 채널의 조각 중 진폭 20 µV
미만이 20%를 넘으면 그 녹음을 배제 목록에 올립니다.** 153개 중 2개가 걸리고 조각
2,194개가 빠져 학습과 평가는 193,285개로 진행됩니다.
