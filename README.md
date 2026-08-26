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
uv pip install -e ".[boosting,dev]"
```

동작 확인:

```bash
.venv/bin/sleepstage env      # 어떤 라이브러리가 실제로 import 되는지
.venv/bin/pytest
```

### macOS 주의

`xgboost` 와 `lightgbm` 은 OpenMP 런타임(`libomp`)이 있어야 import 된다. 없으면 실패한다.

```bash
brew install libomp            # 필요할 때만
uv pip install -e ".[boosting-full,dev]"
```

`scikit-learn` 의 `HistGradientBoosting` 과 `catboost` 는 **libomp 없이 동작한다.**

## 구조

```
src/sleepstage/
├── config.py        설정 로딩 + 해시 (재현성의 축)
├── cli.py           단일 진입점
├── io/              EDF 읽기, 파일 짝짓기
├── preprocess/      라벨 매핑, 에포크화, 절단, 품질 지표
├── features/        대역파워·시간영역 특징
├── models/          선형 / 부스팅 / 앙상블
└── evaluation/      교차검증, 지표, 예측 저장
configs/             실험 하나 = YAML 하나
tests/
docs/                설계 근거와 조사 기록
runs/                실험 산출물
```

## 문서

| 문서 | 내용 |
|---|---|
| `docs/00-glossary.md` | 용어·모델·데이터셋 이름 정리 |
| `docs/01-our-data.md` | 보유 데이터 실측 프로파일 |
| `docs/02-references.md` | 문헌 조사 |
| `docs/05-reference-repos.md` | YASA·sleep-linear 코드 분석 |
| `docs/06-pipeline-repos.md` | DeepSleepNet·U-Time·SLEEPYLAND 분석 |
| `docs/08-preprocessing-design.md` | 전처리 결정 지점과 선택지 |
| `docs/09-preprocessing-proposal.md` | 전처리 제안과 근거 |
| `docs/10-experiment-plan.md` | 실험 계획과 재현성 |

## 한계

건강한 성인 코호트로만 학습·평가한다. **수면장애 환자에서의 성능은 검증되지 않았다.**
N1과 N2를 LS로 합쳤으므로 5-class 문헌 수치와 직접 비교할 수 없다.
