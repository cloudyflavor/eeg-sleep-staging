# sleepstage

EEG 단독 4-class 수면 단계 분류. **재현 가능한 전처리·학습·평가 파이프라인**을 만드는 것이 목적이며,
모든 설계 선택에 근거와 트레이드오프를 남긴다.

## 문제 정의

30초 뇌파 조각(에포크)을 네 단계로 분류한다.

| 클래스 | 원본 (R&K) | 뜻 |
|---|---|---|
| **W** | W | 각성 |
| **LS** | Stage 1, 2 | 얕은 수면 |
| **DS** | Stage 3, 4 | 깊은 수면 |
| **R** | R | REM |

## 데이터

**Sleep-EDF Expanded — Sleep-Cassette** (PhysioNet 1.0.0)

| 항목 | 값 |
|---|---|
| 피험자 | 78명 (**75명이 2박** — 분할 시 피험자로 묶어야 함) |
| 녹음 | 153개 |
| 채널 | EEG Fpz-Cz, EEG Pz-Oz (100 Hz, 바이폴라) |
| 대상 | 건강한 성인, 25–101세 |

수면 구간 앞뒤 30분만 남기는 표준 절단 후 **195,479 에포크**.

| 클래스 | 비율 |
|---|---:|
| W | 33.7 % |
| LS | 46.4 % |
| DS | **6.7 %** |
| R | 13.2 % |

**다수클래스 베이스라인 46.4 %.** 모든 성능은 이 값과 비교해서 읽어야 한다.

## 설계 원칙

> **되돌릴 수 없는 처리를 중간 산출물에 굽지 않는다.**

```
[1단계] EDF ──→ 에포크 npz      정보를 잃지 않는 처리만
[2단계] npz ──→ 특징 parquet    필터·정규화 등 비교 대상은 전부 여기
```

비교 실험이 이 프로젝트의 결과물이므로, 비교 대상이 되는 처리는 재실행이 싼 뒤쪽에 둔다.

## 평가

- **피험자 단위 10-fold 교차검증.** 같은 사람의 두 밤이 학습·평가로 갈리지 않게 묶는다
- fold별 예측을 모아 **pooled**로 지표 계산
- 대표 지표는 **macro-F1**. 정확도는 다수클래스에 휘둘리므로 단독으로 쓰지 않는다
- fold 배정은 **파일로 고정**하고 런타임에 다시 유도하지 않는다

## 설치

[uv](https://docs.astral.sh/uv/) 를 쓴다.

```bash
uv venv --python 3.12
uv pip install -e ".[boosting-full,tracking,analysis,dev]"
uv run pytest
```

extras 는 목적별로 나눠 뒀다.

| | 내용 |
|---|---|
| `boosting` | catboost (macOS 에서 추가 설치 없이 동작) |
| `boosting-full` | + xgboost, lightgbm — **macOS 는 `brew install libomp` 필요** |
| `tracking` | mlflow |
| `analysis` | matplotlib, xlrd |
| `deep` | torch (딥러닝 단계) |
| `dev` | pytest, ruff |

## 사용

```bash
sleepstage config configs/preprocess/base.yaml   # 설정 확인 + 해시
sleepstage prepare --config configs/preprocess/base.yaml --jobs 8
```

`prepare` 는 설정 해시를 산출물에 저장하고, 재실행 시 해시가 같으면 건너뛴다.

## 구조

```
src/sleepstage/
├── cli.py           단일 진입점 (구현된 명령만 등록)
├── config.py        설정 로딩 + 해시 (재현성의 축)
├── io/edf.py        파일 짝짓기, EDF 읽기, 시작시각 검증
└── preprocess/      라벨 코드화, 에포크화, 절단 경계, 품질 지표
configs/             실험 하나 = YAML 하나
tests/
docs/                설계 근거와 조사 기록
```

2단계 이후(`features` / `split` / `train` / `evaluate`)는 아직 없다.
**만들 때 디렉터리를 만든다** — 빈 자리를 미리 잡아두지 않는다.

## 문서

| 문서 | 내용 |
|---|---|
| [`docs/00-glossary.md`](docs/00-glossary.md) | 용어·모델·데이터셋 이름 |
| [`docs/01-data.md`](docs/01-data.md) | 보유 데이터 실측 |
| [`docs/02-prior-art.md`](docs/02-prior-art.md) | 문헌·레포 분석 |
| [`docs/03-stage1.md`](docs/03-stage1.md) | **1단계 결정·근거·실행 결과** |
| [`docs/04-stage2.md`](docs/04-stage2.md) | 2단계 결정 (진행 중) |
| [`docs/05-experiments.md`](docs/05-experiments.md) | 실험 계획과 재현성 |

색인과 열려 있는 결정은 [`docs/README.md`](docs/README.md).

## 이 모델이 하는 일과 하지 않는 일

**만드는 것은 수면 단계 분류기 하나다. 질병 판단은 하지 않는다.**

```
EEG  →  [수면 단계 분류]  →  hypnogram  →  임상 지표  →  의사가 판단
```

그래서 질병 라벨이 붙은 데이터가 필요 없다. 질병별 양상(예: 기면증의 REM 잠복기 단축)은
지표 옆에 붙이는 **문헌 지식**이지 학습 대상이 아니다.

EEG 만 쓰는 게 단점만은 아니다. **1~2채널이면 헤드밴드로 잴 수 있다.** 병원 PSG 는
비싸고 하룻밤밖에 못 재는데, "집에서 여러 밤 재서 수면 구조의 추이를 본다" 는 건
PSG 가 구조적으로 못 하는 일이다.

## 한계

건강한 성인 코호트로만 학습·평가한다. **수면장애 환자에서의 성능은 검증되지 않았다.**
N1과 N2를 LS로 합쳤으므로 5-class 문헌 수치와 직접 비교할 수 없다.
