#!/bin/sh
# 필터와 시간 문맥을 조합한 특징 표 9종을 같은 모델로 비교한다.
# 모델은 HistGB 로 고정해서 차이가 전처리에서만 오게 한다.
# 계산이 싼 조합부터 돌려서 빨리 확인하고, 하나라도 실패하면 멈춘다.
# 로그: logs/compare_preprocess.log
mkdir -p logs
exec > logs/compare_preprocess.log 2>&1
echo "=== 특징 표 9종 비교 시작 ==="
for ctx in none smooth shift; do
  for filt in causal none zerophase; do
    F=$(.venv/bin/sleepstage feature-path --config configs/features/base.yaml --set features.filter=$filt --set features.context=$ctx) || exit 1
    echo "=== filter=$filt context=$ctx $F ==="
    .venv/bin/sleepstage train --config configs/experiment/base.yaml \
      --set train.features="$F" || { echo "FAILED $filt/$ctx"; exit 1; }
  done
done
echo "=== 완료 ==="
