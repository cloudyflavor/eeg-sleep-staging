#!/bin/sh
# Phase 1.5 — catboost 고정 ablation: 채널 / MMD·Esis / 정규화
SMOOTH=data/features/b51ceabfb2b8.parquet
T=".venv/bin/sleepstage train --config configs/experiment/base.yaml    --set train.model=catboost --set train.drop_missing=true"
# 정규화 없는 특징 표 (있으면 건너뜀)
.venv/bin/sleepstage features --config configs/features/base.yaml   --set features.filter=zerophase --set features.context=smooth   --set features.normalize=none --jobs 8 || exit 1
for sub in fpz_only pz_only no_distance; do
  echo "=== subset=$sub ==="
  $T --set train.features=$SMOOTH --set train.feature_subset=$sub || { echo "FAILED $sub"; exit 1; }
done
echo "=== normalize=none ==="
$T --set train.features=data/features/f5f066c371ad.parquet || { echo "FAILED nonorm"; exit 1; }
echo "=== Phase 1.5 완료 ==="
