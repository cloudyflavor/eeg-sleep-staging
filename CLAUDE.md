# Sleep EDF 프로젝트

EEG만으로 수면 단계를 자동 분류하는 모델 개발.
최종 산출물은 **재현 가능한 전처리·학습·평가 파이프라인**이며, 선택마다 근거와 트레이드오프를 남기는 것이 목적이다.

> **상세 설계 문서는 로컬 저장소 `docs/`에 있다** (`~/Documents/dev/eeg-sleep-staging/docs/`).
> 이 파일은 서버 데이터의 명세와 확정 사항만 다룬다.

---

## 프로젝트 결정사항 (2026-08-25 기준)

| 항목 | 결정 | 비고 |
|---|---|---|
| **데이터셋** | **EDFX Sleep-Cassette 단독** | HMC는 보류. Telemetry는 미사용 |
| **채널** | **EEG Fpz-Cz + EEG Pz-Oz** (2채널) | EOG/EMG/ECG 사용 안 함 |
| **클래스** | **4-class** — W / LS / DS / R | 5-class 문헌과 직접 비교 불가 |
| **모델** | 수작업 특징 + 부스팅 트리 → 이후 PyTorch 딥러닝 추가 | 같은 프로토콜로 비교하는 것이 목적 |
| **평가** | 피험자 단위 교차검증 | 아래 "분할 시 주의" 참고 |

**진행 상태**: 전처리를 처음부터 재설계 중. 이전에 작성한 특징 추출·학습 코드는 전부 삭제했다.
설계 선택지는 `docs/08-preprocessing-design.md` 참고.

---

## ⚠️ 분할 시 반드시 지킬 것

**EDFX는 78명이 153개 녹음을 만든다. 75명이 2박씩 녹음했다.**

| 녹음 수 | 피험자 수 |
|---|---|
| 2박 | 75 |
| 1박 | 3 |

파일 단위로 나누면 같은 사람의 1일차가 train, 2일차가 test로 들어간다.
같은 사람의 뇌파는 밤이 달라도 매우 비슷하므로 명백한 누수다.

**분할 키는 파일명이 아니라 `SC4[XX]`의 XX(피험자 번호)여야 한다.**

---

## 실측 클래스 분포 (4-class, 전수 계산)

wake 절단은 첫/마지막 비-W 에포크로부터 ±60 에포크(30분)를 남기는 표준 관행.

| 클래스 | 절단 전 | % | 절단 후 | % |
|---|---:|---:|---:|---:|
| W | 285,433 | **68.79** | 65,951 | 33.74 |
| LS | 90,654 | 21.85 | 90,654 | 46.38 |
| DS | 13,039 | 3.14 | **13,039** | **6.67** |
| R | 25,835 | 6.23 | 25,835 | 13.22 |
| **계** | **414,961** | | **195,479** | |

- **절단하면 에포크의 53%가 사라지고, 전부 W다.**
- **다수클래스 베이스라인**: 절단 전 68.8%(W), **절단 후 46.4%(LS)**. 성능 수치는 반드시 이것과 비교할 것.
- **DS가 절단 후에도 6.67%뿐이다.** 고령 피험자가 많아 서파수면이 적기 때문이며 데이터 오류가 아니다.

### 피험자 인구 구성 (`SC-subjects.xls`, 녹음 153건 기준)

| 항목 | 값 |
|---|---|
| 연령 | 평균 59.0, 표준편차 22.1, 범위 **25–101세** |
| 연령 분포 | ~40세 39 / 40–60세 43 / 60–80세 39 / **80세 초과 32** |
| 성별 | 여 82 / 남 71 |

`SC-subjects.xls`는 `subject / night / age / sex(F=1) / LightsOff` 컬럼을 가진다.
읽으려면 `xlrd`가 필요한데 서버 venv에 없다 (필요하면 venv 안에 설치).

---

## 데이터셋 개요

| 항목 | Sleep-EDF Expanded (EDFX) | HMC Sleep Staging |
|------|--------------------------|-------------------|
| 출처 | PhysioNet 1.0.0 | PhysioNet 1.1 |
| 피험자 수 | 100명 (Cassette 78 + Telemetry 22) | 151명 |
| 총 녹음 수 | 197개 (Cassette 153 + Telemetry 44) | 151개 |
| 대상 | 건강한 성인 | 수면 클리닉 환자 |
| 스코어링 기준 | R&K | AASM |
| EEG 샘플링 레이트 | 100 Hz | 256 Hz |
| **현재 사용 여부** | **Cassette만 사용** | **보류** |

---

## Dataset 1: Sleep-EDF Expanded (EDFX) — 사용 중

**경로:** `data/raw/edfx/physionet.org/files/sleep-edfx/1.0.0/`

### 1-A. Sleep-Cassette (SC) — 이것을 쓴다

- **파일 수:** PSG 153개 + Hypnogram 153개
- **대상:** 건강한 성인, 수면/정신 장애 없음, 연령 25–101세
- **녹음 시간:** 약 22시간 (취침 전후 포함 — 그래서 wake 절단이 필요하다)
- **파일명:** `SC4[XX][N]E0-PSG.edf` / `SC4[XX][N]EC-Hypnogram.edf`
  - XX: 피험자 번호(00–82), N: 녹음 차수(1 or 2)
  - **PSG와 Hypnogram은 앞 6자(`SC4XXN`)가 공통이고 7번째 문자만 다르다** (`0` vs `C`)
- **수면 단계 (7class):** `W`, `1`, `2`, `3`, `4`, `R`, `?` + `Movement time`

#### 채널 목록

| # | 채널명 | 신호 종류 | 샘플링 | 단위 | 하드웨어 필터 |
|---|--------|-----------|------|------|------------|
| **0** | **EEG Fpz-Cz** | EEG (전두–중심) | **100 Hz** | µV | HP 0.5Hz / LP 100Hz |
| **1** | **EEG Pz-Oz** | EEG (두정–후두) | **100 Hz** | µV | HP 0.5Hz / LP 100Hz |
| 2 | EOG horizontal | 수평 안구운동 | 100 Hz | µV | HP 0.5Hz / LP 100Hz |
| 3 | Resp oro-nasal | 구강–비강 호흡 | 1 Hz | — | HP 0.03Hz / LP 0.9Hz |
| 4 | EMG submental | 턱 근전도 | **1 Hz** | µV | HP 16Hz + 정류, LP 0.7Hz |
| 5 | Temp rectal | 직장 체온 | 1 Hz | °C | — |
| 6 | Event marker | 이벤트 마커 | 1 Hz | — | Hold 2s |

> **0번과 1번만 사용한다.** EMG가 1Hz라 사실상 못 쓰는 점이 채널 선택의 제약이었다.
> EEG는 유양돌기 기준 없이 **바이폴라 유도**(두 두피 지점의 전위차)다. EDF record duration = 30초.
> **하드웨어에서 이미 필터링돼 있다** — 추가 밴드패스가 꼭 필요한지는 실험으로 확인할 것.

### 1-B. Sleep-Telemetry (ST) — 미사용

22명 × 2박(temazepam vs 위약), 약 8시간. 약물 연구 설계라 현재 스코프에서 제외.

---

## Dataset 2: HMC Sleep Staging — 보류

**경로:** `data/raw/hmc/physionet.org/files/hmc-sleep-staging/1.1/`

- **출처:** Haaglanden Medisch Centrum (HMC), 네덜란드 헤이그 — 병원 수면센터
- **측정 시기:** 2018년 (PhysioNet 공개 2021-07-01, v1.1 2022-03-18)
- **원논문:** Alvarez-Estevez D, Rijsman RM (2021), *Inter-database validation of a deep learning
  approach for automatic sleep scoring*, PLoS ONE 16(8): e0256111
  → **데이터베이스 간 교차 검증 논문. 나중에 EDFX↔HMC 전이를 할 때 반드시 읽을 것.**
- **파일 수:** PSG 151개 + Annotation 151개 (SN001–SN154, SN014·SN064·SN135 누락)
- **피험자:** 151명 (남 85 / 여 66), 1인 1박 — **분할 시 피험자 그룹핑 불필요**
- **연령:** 평균 53.9 ± 15.4세. **개인별 나이·성별·진단명은 없다** (집계값만 제공)
- **임상 지표(집계):** TIB 7.6h, TST 6.3h, SE 83%, AHI 14.6

#### 채널 목록

| # | 채널명 | 신호 종류 | 샘플링 | 하드웨어 필터 |
|---|--------|-----------|------|------------|
| 0–3 | EEG F4-M1 / C4-M1 / O2-M1 / C3-M2 | EEG | 256 Hz | HP 0.2Hz / **LP 35Hz** |
| 4 | EMG chin | 턱 근전도 | 256 Hz | HP 1.0Hz / LP 150Hz |
| 5–6 | EOG E1-M2 / E2-M2 | 안구운동 | 256 Hz | HP 0.2Hz / LP 35Hz |
| 7 | ECG | 심전도 | 256 Hz | HP 1.0Hz / LP 150Hz |

> 10-20 배치, 반대편 유양돌기(M1/M2) 기준 **단극 유도**. EDFX의 바이폴라와 **전극 구성이 근본적으로 다르다**
> — 두 데이터셋에 공통 채널이 하나도 없다는 뜻이며, 이것이 HMC를 보류한 이유 중 하나다.

**HMC 주석 파일 내용 (151개 전수 확인):** 수면 단계 5종 + `Lights on`/`Lights off`뿐.
**무호흡 이벤트·각성·사지운동·진단명은 없다.** 단 `Lights off/on`이 있어서 wake 절단을 라벨 없이 할 수 있다(EDFX엔 없는 이점).

---

## 실측 클래스 분포 — HMC (참고)

| 클래스 | 절단 전 % | 절단 후 % |
|---|---:|---:|
| W | 17.26 | 16.51 |
| LS | 47.82 | 48.25 |
| DS | **19.43** | **19.61** |
| R | 15.49 | 15.63 |

병원에서 취침 시간에 맞춰 ~8시간만 녹음해 처음부터 균형이 잡혀 있다. 절단해도 0.9%p만 움직인다.
**DS가 EDFX의 3배다.**

---

## 수면 단계 매핑 (4클래스)

| EDFX (R&K) | HMC (AASM) | 통합 | int |
| --- | --- | --- | --- |
| W | W | **W** | 0 |
| Stage 1, 2 | N1, N2 | **LS** (Light Sleep) | 1 |
| Stage 3, 4 | N3 | **DS** (Deep Sleep) | 2 |
| R | R | **R** (REM) | 3 |
| `?`, `Movement time` | — | 제외 | — |

> **주의**: N1+N2를 LS로 합치면 이 분야에서 가장 어려운 경계(N1 vs N2)가 사라진다.
> 따라서 4-class 수치는 5-class 문헌과 직접 비교할 수 없다. 논문·발표에 반드시 명시할 것.

---

## 디렉토리 구조

**2026-08-26에 폴더를 하나로 합쳤다.** 예전 `sleep_edf/`는 삭제했다 (중복본 16GB +
구 전처리 산출물 46GB 회수). 코드·데이터·환경이 전부 이 아래에 있다.

```
sleepstage/
├── CLAUDE.md
├── pyproject.toml                    # uv + hatchling
├── .venv/                            # Python 3.12.12
├── src/sleepstage/
│   ├── cli.py                        # 단일 진입점: env / config / prepare / ...
│   ├── config.py                     # 설정 로딩 + 해시 (재현성의 축)
│   ├── io/edf.py                     # 파일 짝짓기, EDF 읽기 (µV 변환)
│   ├── preprocess/                   # labels.py  epochs.py  prepare.py
│   ├── features/  models/  evaluation/    # 비어 있음 (2단계 이후)
├── configs/preprocess/base.yaml      # 1단계 설정
├── tests/
└── data/
    ├── raw/
    │   ├── edfx/physionet.org/files/sleep-edfx/1.0.0/
    │   │   ├── sleep-cassette/       # SC4XXX-PSG.edf + Hypnogram.edf  ← 사용
    │   │   ├── sleep-telemetry/      # 미사용
    │   │   └── SC-subjects.xls       # 피험자 나이·성별 (읽으려면 xlrd 필요)
    │   └── hmc/physionet.org/...     # HMC 302 EDF. 현재 미사용
    └── interim/epochs/               # 1단계 산출물 9.3GB
        ├── SC400N1.npz ... (153개)
        └── manifest.json             # 녹음별 감사 기록
```

### ⚠️ 서버에서 코드를 편집하지 말 것

코드의 정본은 **로컬 저장소**이고 GitHub `cloudyflavor/eeg-sleep-staging` (비공개) 에 있다.
서버는 실행 환경일 뿐이고 git 저장소가 아니다 — **공용 서버라 자격증명을 두지 않는다.**

```bash
# 로컬에서 (정본 → 서버). --delete 라서 서버 쪽 수정은 조용히 사라진다
rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' \
      src configs tests pyproject.toml README.md biglab:~/documents/chaerin/sleepstage/
```

`docs/` 는 로컬에만, `data/ .venv/ CLAUDE.md` 는 서버에만 있다.
저장소가 공개로 바뀌면 rsync 대신 서버에서 `git clone` → `git pull` 로 전환할 수 있다
(공개 저장소는 읽기에 자격증명이 필요 없다).

### 1단계 산출물 (`data/interim/epochs/*.npz`)

| key | 내용 |
|---|---|
| `data` | (n_ep, 2, 3000) float32 **µV** — Fpz-Cz, Pz-Oz |
| `labels` | (n_ep,) int8 — 0=W 1=LS 2=DS 3=R |
| `epoch_idx` | (n_ep,) int32 — 원본 에포크 번호. 번호가 튀면 판독불가로 제거된 자리 |
| `crop_lo` / `crop_hi` | wake 절단 경계. **자르지 않고 기록만** 했다 |
| `ptp` / `std` / `flat_ratio` | (n_ep, 2) float32 — 아티팩트 지표. **제거하지 않았다** |
| `config_hash` | 멱등 재개용 |

실측: 153녹음 / 78피험자 / 414,961 에포크 (절단 적용 시 195,479).
절단 후 W 33.7 / LS 46.4 / DS 6.7 / R 13.2 %.

### 디스크

**여유 151GB (3.2TB 중 96% 사용).** 공용 서버다.
**원본을 리샘플링해 통째로 복사하는 설계는 하지 말 것** — 디스크 예산이 안 된다.

---

## 폐기된 구 전처리에서 배운 것

구 코드와 `data/processed/` 는 삭제했지만, 아래 실측은 재설계에 계속 유효하다.

- **고정 임계값 아티팩트 제거는 EDFX에서 작동하지 않는다.** `ptp > 500 µV` 기준의
  검출률이 EDFX 전 파일 평균 **0.00%** 였다. EDFX 최대 ptp 가 약 353 µV 라 절대 안 걸린다.
  HMC 는 13.99% 가 걸리는데 대부분 **±800 µV 하드 클리핑**(값 범위가 정확히
  `[-0.0008, 0.0008]` = 포화)이다.
  → **임계값 하나로 두 데이터셋을 다룰 수 없다.** 그래서 현재 설계는 제거 대신 지표만 저장한다.
- EDFX(SC)와 ST 의 진폭 차이는 나이가 아니라 **장비** 때문이다 (25–50세로 맞춰 비교했을 때도
  LS 기준 SC 15.3 µV vs ST 23.8 µV).
- DS 비율 감소는 **나이** 때문이 맞다 (25–39세 13.5% → 80세+ 2.8%).

---

## 작업 관행

**대용량 처리는 반드시 1–2개 파일로 먼저 테스트하고 정상 확인 후 전체 실행.**

**공용 서버다. `~/documents/chaerin/` 밖은 건드리지 말 것.**

작업 루프: 코드는 로컬(macOS)에서 작성 → rsync/scp 업로드 → `ssh biglab '...'` 실행 → 로그·결과만 회수.
대용량 데이터는 로컬로 통째 전송하지 않는다. 특징 행렬 수준(수백 MB)은 전송해도 된다.

### 환경

- `.venv/` — Python 3.12.12. **2026-08-26에 새로 만들었다.**
  `uv pip install -e ".[boosting-full,tracking,dev]"` 로 설치.
  numpy 2.5.2 / scipy 1.18.1 / mne 1.12.1 / pandas 2.3.3 / pyarrow 25.0.1 / sklearn 1.9.0 /
  **catboost 1.2.10 · xgboost 3.4.1 · lightgbm 4.7.0 · mlflow 3.15.1** / pytest · ruff
- uv 는 `~/.local/bin/uv` (PATH 에 없으면 직접 지정)
- **미설치: torch, pyedflib, xlrd** — 필요해지면 venv 안에 설치한다 (시스템 전역 설치는 금지)
- GPU: NVIDIA RTX 2080 Ti 11GB (Turing — **bf16 미지원, fp16만**)
- CPU 40코어, RAM 125GB
