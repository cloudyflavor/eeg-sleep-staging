#!/bin/sh
# 특징을 하나씩 빼면서 성능이 얼마나 떨어지는지 잰다.
# 채널 하나만 쓰기, 거리 기반 특징 빼기, 정규화 빼기 세 가지를 본다.
# 모델은 catboost 로 고정해서 차이가 특징에서만 오게 한다.
# 로그: logs/ablate_features.log
mkdir -p logs
exec > logs/ablate_features.log 2>&1
FIXED="--set features.filter=zerophase --set features.context=smooth"
SMOOTH=$(.venv/bin/sleepstage feature-path --config configs/features/base.yaml $FIXED) || exit 1
T=".venv/bin/sleepstage train --config configs/experiment/base.yaml \
   --set train.model=catboost --set train.drop_missing=true"

for sub in fpz_only pz_only no_distance; do
  echo "=== subset=$sub ==="
  $T --set train.features="$SMOOTH" --set train.feature_subset=$sub \
    || { echo "FAILED $sub"; exit 1; }
done

echo "=== normalize=none ==="
.venv/bin/sleepstage features --config configs/features/base.yaml --jobs 8 \
  --set features.filter=zerophase --set features.context=smooth \
  --set features.normalize=none || exit 1
NONORM=$(.venv/bin/sleepstage feature-path --config configs/features/base.yaml $FIXED --set features.normalize=none) || exit 1
$T --set train.features="$NONORM" || { echo "FAILED nonorm"; exit 1; }
echo "=== 완료 ==="
