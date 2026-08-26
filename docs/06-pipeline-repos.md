# 데이터 파이프라인·벤치마크 레포 분석

조사일 2026-08-25. 네 레포를 clone해서 소스를 읽은 결과.

| 레포 | 프레임워크 | 마지막 커밋 | 테스트 | CI | 지금 돌아가나 |
|---|---|---|---|---|---|
| DeepSleepNet | TF1 (Py 3.5) | 2017년 코드 | 없음 | 없음 | **불가** — 명세로만 읽을 것 |
| TinySleepNet | TF1 | 2023-08 | 없음 | 없음 | 전처리 스크립트만 거의 가능 |
| **U-Time / psg-utils** | TF ≥2.8 | **2026-06 (활발)** | psg-utils만 있음 | 없음 | 대체로 가능 |
| SLEEPYLAND | Docker 마이크로서비스 | 2026-04 | 없음 | 없음 | `docker compose up` |

**넷 다 CI가 없습니다. 의미 있는 테스트는 `psg-utils` 하나뿐입니다.**
→ 테스트와 CI를 붙이는 것만으로 이 분야의 기준선을 넘습니다.

---

## 1. Sleep-EDF 전처리의 원본 — DeepSleepNet `prepare_physionet.py`

이후 논문 대부분이 베낀 스크립트입니다. 217줄 전체를 읽은 결과.

### wake 절단 — 관행이 아니라 실제 코드

```python
w_edge_mins = 30
nw_idx = np.where(y != stage_dict["W"])[0]
start_idx = nw_idx[0]  - (w_edge_mins * 2)   # 60 에포크 = 30분
end_idx   = nw_idx[-1] + (w_edge_mins * 2)
```

**첫/마지막 비-W 에포크로부터 ±60 에포크**입니다. `*2`인 이유는 1분에 30초 에포크가 2개라서입니다.
→ **제가 짠 코드(`CROP_EPOCHS = 60`, 첫/마지막 비-W 기준)와 정확히 일치합니다.** ✅

### 레이블 매핑 — 모두가 복사하는 그 dict

```python
ann2label = {
    "Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
    "Sleep stage 3": 3, "Sleep stage 4": 3,   # R&K S3/S4 → AASM N3
    "Sleep stage R": 4,
    "Sleep stage ?": 5, "Movement time": 5,   # 둘 다 UNKNOWN
}
EPOCH_SEC_SIZE = 30
```

### 채널

`--select_ch` 기본값이 `"EEG Fpz-Cz"`. **한 번에 한 채널**입니다.
README는 Fpz-Cz와 Pz-Oz에 대해 스크립트를 두 번 돌려서 **별도 디렉터리**에 넣으라고 합니다.
→ 이 분야의 표준 관행이 애초에 **단일 채널**이라는 뜻입니다.

### 베끼면 안 되는 것

- **PSG와 Hypnogram을 정렬된 리스트의 인덱스로 짝짓습니다.** `glob().sort()` 두 개를 `i`로 매칭. 피험자 ID 대조가 없습니다. 안전장치는 EDF 헤더의 시작시각 assert 하나뿐입니다.
- **시작할 때 출력 디렉터리를 `shutil.rmtree`** 합니다. 캐시도 재개도 없이 매번 전체 재빌드입니다.
- `print`가 로깅입니다.

---

## 2. TinySleepNet — 같은 레시피, 나은 엔지니어링

베낀 게 아니라 다시 쓴 것입니다. 배울 점:

- **pyedflib로 레코드 단위 읽기** → `readSignal(ch).reshape(-1, epoch_samples)` 로 **에포크 자르기가 공짜**입니다. DeepSleepNet의 복잡한 인덱스 산수가 통째로 사라집니다.
- **assert가 훨씬 많습니다**: 시작시각 일치, 주석 연속성(`onset_sec == total_duration`), 에포크 수 일치.
- **파일명 정규식으로 피험자를 묶습니다** (`data.py::get_subject_files()`). Sleep-EDF의 2박이 같은 fold에 남습니다. DeepSleepNet엔 이게 없습니다.
- **녹음별 감사 로그**를 출력 디렉터리 안에 남깁니다. 어떤 에포크를 왜 뺐는지가 전부 기록됩니다. 싸고 훌륭합니다.
- 특정 파일의 EDF 레코드 길이가 60초인 예외(`SC4362F0`)를 문서화하고 처리합니다.

### 조용한 함정 하나

두 레포는 **레이블 처리 순서가 반대**입니다.

- DeepSleepNet: UNKNOWN을 **샘플 단위**로 제거 → wake 절단
- TinySleepNet: wake 절단 → MOVE/UNK를 **에포크 단위**로 제거

그리고 TinySleepNet은 `"Sleep stage ?"`(6)와 `"Movement time"`(5)을 **구분해서 남깁니다**.
→ **같은 데이터에서 서로 다른 에포크 집합이 나옵니다.** 논문 수치를 비교할 때 알고 있어야 할 차이입니다.

---

## 3. U-Time / psg-utils — 다중 코호트를 다루는 기계

**우리 EDFX+HMC 문제에 가장 직접적인 답입니다.** 데이터 계층이 별도 패키지 `perslev/psg-utils`로 분리돼 있습니다.

### 코호트마다 YAML 하나

```yaml
data_dir: ...
psg_regex: ...          # 정렬 인덱스가 아니라 정규식으로 짝짓기
hyp_regex: ...
period_length: 30
set_sample_rate: 128
channel_sampling_groups: ...
sleep_stage_annotations: {...}   # 데이터셋별 문자열→정수 매핑
quality_control_func: clip_noisy_values
min_max_times_global_iqr: 20
scaler: "RobustScaler"
```

U-Sleep은 이런 YAML **13개**로 13개 코호트를 학습시킵니다.
**코드에 `if dataset == ...` 분기가 없습니다.** 데이터셋의 특이사항은 전부 설정으로 빠집니다.

→ **우리의 EDFX/HMC 차이(전극·샘플링레이트·채점기준)가 20줄짜리 YAML 두 개가 됩니다.**

### 전극 배치가 다른 문제를 어떻게 푸나

- `ChannelMontage`가 `"EEG Fpz-Cz"`를 채널+기준전극으로 파싱합니다
- `channel_types.py`가 MNE의 `standard_1020` 몽타주로 만든 정규식으로 채널명을 EEG/EOG/EMG로 **자동 분류**합니다
- **`RandomChannelSelector`**: 학습할 때마다 각 그룹에서 **채널 하나를 무작위로 뽑습니다**. `[EEG Fpz-Cz, EEG Pz-Oz]` 중 하나, EOG 중 하나 식으로
- `ChannelDropout` 증강 (`drop_fraction: 0.5, apply_prob: 0.1`)

**이게 U-Sleep이 전극 배치에 강한 이유입니다** — 특정 채널에 의존하지 못하게 학습 중에 계속 바꿔 버립니다.

> 이건 딥러닝 학습 기법이라 부스팅 트리에 그대로 안 옵니다. 하지만 **설정 패턴은 그대로 가져올 수 있습니다.**

### 샘플링레이트

전부 로드 시점에 `set_sample_rate`(128)로 통일. `scipy.signal.resample_poly`(기본) 또는 `mne.filter.resample`.
→ **제가 `resample_poly`를 고른 것과 같습니다.** ✅ (목표 레이트는 128 vs 100으로 다름)

### R&K vs AASM

데이터셋별 `sleep_stage_annotations`가 문자열을 표준 정수로 매핑합니다. 지저분한 코호트를 위해
`stage_mapper.py`가 임의 문자열을 퍼지 매칭합니다 (`"WAKE"/"WK"/"W"`, `"R.E.M."`, `"MT"/"MOVEMENT"/"?"` → UNKNOWN).

**UNKNOWN을 버리지 않고 손실에서 마스킹**합니다 (`ignore_out_of_bounds_classes: true`).
→ 우리는 버리고 있습니다. 딥러닝이 아니면 마스킹이 어려우니 버리는 게 맞지만, 선택지로 알아둘 것.

### 메모리보다 큰 데이터

세 가지 큐가 있습니다.

| 큐 | 방식 | 용도 |
|---|---|---|
| `EagerQueue` | 전부 RAM | 작은 데이터 |
| `LazyQueue` | `with study.loaded_in_context():` 로드-사용-해제 | 검증 |
| `LimitationQueue` | 최대 25–40개만 유지, 32–50회 접근 후 축출, 백그라운드 로더 스레드 7개 | 대규모 학습 |

`H5SleepStudy`는 h5py로 **필요한 구간만 슬라이스**해서 읽습니다. 전체를 메모리에 올리지 않습니다.

> **우리에겐 과합니다.** 이건 딥러닝 미니배치 문제를 푸는 장치인데, 부스팅 트리는 특징 행렬이 RAM에 들어갑니다.

---

## 4. SLEEPYLAND — 벤치마크 하네스

Docker Compose 마이크로서비스 5개(`gui` → `manager-api` → `nsrr-download` / `wild-to-fancy` / `usleepyland`).

### 데이터셋 등록 = JSON 한 블록

`dataset_commands.json`에 데이터셋마다 `--channels`(그 코호트 EDF의 원본 채널명), `--rename_channels`(표준 이름으로 매핑), `--resample 128`, `--seed 123`을 적습니다.

### 모델마다 입력 포맷이 다른 문제를 어떻게 푸나

**푸는 게 아니라 없앱니다.** DeepResNet과 SleepTransformer를 U-Time 포크 안에 **다시 구현**해서 모든 신경망 모델이 동일한 128Hz 조화된 입력을 먹게 만들었습니다. 유일한 특징 기반 모델(YASA)만 `--model_external yasa`로 빠져나갑니다.

> **어댑터를 만드는 대신 상류에서 이질성을 제거하는 설계.** 우리가 EDFX/HMC를 다룰 때 참고할 방향입니다.

---

## 5. 훔칠 만한 패턴 정리

| 패턴 | 출처 | 왜 |
|---|---|---|
| **녹음당 중간 파일 하나** | 넷 다 | 병렬화가 자명, 부분 재빌드 가능, 피험자 단위 분할이 쉬움 |
| **데이터셋 특이사항을 설정으로** | U-Time YAML, SLEEPYLAND JSON | 코드에 `if dataset ==` 분기가 없어짐 |
| **정규식 짝짓기 + 강한 assert** | U-Time, TinySleepNet | 정렬 인덱스 짝짓기의 사고를 원천 차단 |
| **rmtree 대신 멱등 재개** | (없음 — 이게 안티패턴) | 설정 해시가 같고 출력이 있으면 건너뛰기. 15줄이면 됨 |
| **녹음별 감사 로그** | TinySleepNet | 무엇을 왜 뺐는지 추적 가능 |
| **QC를 이름 붙은 파라미터 함수로** | psg-utils | `clip_noisy_values` (임계값 = 채널별 전역 IQR × 20) |
| **시드를 설정에 박기** | SLEEPYLAND | 모든 명령에 `--seed 123` |
| **분할을 파일로 고정** | U-Time `fixed_split` | 런타임에 다시 유도하지 않음 |

**안티패턴**: 정렬 인덱스 짝짓기, 시작 시 `rmtree`, `print` 로깅, 모델 레지스트리 하드코딩, 동시 쓰기가 있는 단일 HDF5.

---

## 6. 우리 상황에 대한 권고

제약: 서버 디스크 여유 **103GB**(3.2TB 중 97% 사용), 원본 PSG 86GB, 부스팅 트리용 특징 행렬, 전극·레이트가 다른 데이터셋 2개, 포트폴리오.

### 6-1. 가장 중요한 디스크 결정

**원본 86GB → 녹음당 특징 parquet = 총 1–3GB.**

에포크당 수백 개 float × 약 10⁵–10⁶ 에포크면 이 정도입니다.
**리샘플링된 신호 사본을 통째로 만들면 안 됩니다** — 103GB 예산이 날아갑니다.

> ⚠️ **우리는 이미 그걸 했습니다.** `data/processed/`에 46GB의 npz가 있습니다.
> 이건 신호를 그대로 담고 있어서 원본과 크기가 비슷합니다.
> 특징만 뽑고 나면 이 중간 산출물이 필요한지 재검토해야 합니다.

권장 흐름: EDF 하나 읽기 → 특징 추출 → `SC4001E0.parquet` 쓰기 (키: `dataset, subject, night, epoch_idx, stage`) → 메모리 해제 → 다음 파일. 2단계는 단순 concat.

### 6-2. 병렬화

녹음 단위 `ProcessPoolExecutor`/joblib. 완전히 독립적이고, 파일이 분리돼 있어 동시 쓰기 문제가 없습니다.
U-Time의 큐/스트리밍 기계는 **건너뜁니다** — 딥러닝 미니배치 문제를 푸는 장치입니다.

### 6-3. 차별점

넷 다 CI가 없고 셋은 테스트가 없습니다.
합성 EDF로 stage-mapper·wake-trim·짝짓기 로직을 pytest하고 GitHub Actions를 붙이면 **이 분야 기준선을 쉽게 넘습니다.**
