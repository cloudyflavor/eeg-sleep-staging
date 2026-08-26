# 용어·이름 정리

조사 문서에 나오는 이름들이 뭔지 한 줄씩. 헷갈릴 때 여기 보면 된다.

---

## 도구 (실제로 설치해서 돌릴 수 있는 것)

**YASA** — "Yet Another Spindle Algorithm". 파이썬 패키지다. `pip install yasa`.
수면 단계 자동 분류 + 수면방추(spindle)·서파 검출 + 아티팩트 제거 + 스펙트럼 분석을 다 한다.
라이선스 BSD-3-Clause(상업 이용 가능), 현재도 관리되고 있다. 논문은 Vallat & Walker 2021, *eLife*.
**우리와 같은 계열이다** — 딥러닝이 아니라 수작업 특징 약 200개 + LightGBM(부스팅 트리).

**sleep-linear** — Van der Donckt 논문의 공개 코드. `github.com/predict-idlab/sleep-linear`.

**SLEEPYLAND** — 24개 데이터셋을 모아 여러 모델을 같은 조건에서 비교하는 벤치마크 도구.
`github.com/biomedical-signal-processing/sleepyland`.

---

## 분류기 라이브러리 (전부 같은 계열)

**XGBoost / LightGBM / CatBoost** — 셋 다 그래디언트 부스팅 트리 라이브러리다. 구현이 다를 뿐 같은 종류의 모델이다.
우리 기존 논문은 XGBoost, YASA는 LightGBM, Van der Donckt는 CatBoost를 썼다.

---

## 논문·모델 이름

**Van der Donckt et al. 2023** — 사람 이름이지 도구가 아니다. 벨기에 겐트대 연구팀.
"딥러닝 말고 전통 ML도 봐라"는 취지의 논문. 수작업 특징 1,048개 + CatBoost로 딥러닝 모델들과 동급을 냈다.

**DeepSleepNet / TinySleepNet** — CNN + LSTM 구조의 고전 딥러닝 모델. 대부분의 논문이 비교군으로 쓴다.

**SeqSleepNet → XSleepNet → SleepTransformer → L-SeqSleepNet** — Huy Phan이라는 연구자가 이어서 낸 딥러닝 모델 계보.
전부 TensorFlow 1.x이고 라이선스가 CC-BY-NC(비상업)라 실무에 쓰기 껄끄럽다.

**U-Sleep** — 여러 코호트를 한꺼번에 학습시킨 딥러닝 모델. 우리 EDFX↔HMC 구상과 가장 가까운데 **아직 조사 못 했다.**

**SleepFM / SleepGPT / LaBraM** — 대규모 사전학습 "파운데이션 모델". 수면 단계 분류만 놓고 보면 전용 모델을 못 이긴다.

---

## 데이터셋

**Sleep-EDF (EDFX)** — 우리가 가진 것. PhysioNet 공개. 건강한 성인.
- **SC-20 / Sleep-EDF-20** — 초기 20명 부분집합. 옛날 논문들이 쓴다.
- **SC-78 / Sleep-EDF-78** — 78명 전체. 요즘 표준. **우리가 가진 게 이거다.**

**HMC** — 우리가 가진 또 하나. 네덜란드 병원의 수면 클리닉 환자 151명.

**SHHS / MASS / Physio2018 / DOD-H·DOD-O / ISRUC** — 다른 논문들이 쓰는 데이터셋. 대부분 신청·승인이 필요하다.

**NSRR** — National Sleep Research Resource. 위 데이터셋 상당수가 여기서 배포된다.

---

## 지표

**ACC (정확도)** — 맞춘 비율. **이 분야에서는 단독으로 믿으면 안 된다.** 클래스가 심하게 불균형해서
다수 클래스만 찍어도 높게 나온다. (우리 EDFX는 crop 전 W만 찍어도 68.8%)

**macro-F1 (MF1)** — 클래스별 F1을 단순 평균한 값. 소수 클래스를 못 맞히면 바로 떨어진다.
**이 분야의 실질적 대표 지표다.**

**κ (Cohen's kappa)** — 우연히 맞을 확률을 빼고 계산한 일치도. 0~1. 0.8 넘으면 좋은 편.
인간 채점자끼리의 일치도도 이걸로 잰다.

**LOSO (Leave-One-Subject-Out)** — 피험자 한 명씩 빼고 학습, 뺀 사람으로 평가. 피험자 단위 평가의 가장 엄격한 형태.

---

## 수면 단계 표기

| R&K (구 기준, EDFX) | AASM (현 기준, HMC) | 우리 4-class |
|---|---|---|
| W | W | **W** |
| Stage 1, Stage 2 | N1, N2 | **LS** (얕은 수면) |
| Stage 3, Stage 4 | N3 | **DS** (깊은 수면) |
| R | R | **R** (REM) |
| ? | — | 제외 |

**N1** — 가장 얕은 수면. 문헌에서 "가장 어려운 클래스"로 계속 등장한다. 전문가끼리도 잘 안 맞는다.
우리는 N1+N2를 LS로 합쳤기 때문에 이 문제를 우회한 셈이다(그래서 5-class 논문과 직접 비교가 안 된다).
