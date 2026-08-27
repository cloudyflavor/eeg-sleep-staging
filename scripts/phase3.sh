#!/bin/sh
# Phase 3 — catboost 튜닝·가중치·특징 수. 전처리는 zerophase-smooth 고정.
F=data/features/b51ceabfb2b8.parquet
BASE=".venv/bin/sleepstage"
COMMON="--set train.features=$F --set train.model=catboost --set train.drop_missing=true"

echo "=== 1. 무작위 탐색 8조합 × 3-fold ==="
$BASE tune --config configs/experiment/base.yaml --trials 8 --search-folds 3 $COMMON || exit 1

echo "=== 2. 클래스 가중치 비교 (balanced 는 Phase 2 결과 재사용) ==="
$BASE train --config configs/experiment/base.yaml $COMMON --set train.class_weight=none || exit 1

echo "=== 3. 특징 수 곡선 ==="
IMP=$(ls -t runs/*/importance.csv | head -1)
$BASE feature-curve --config configs/experiment/base.yaml --importance "$IMP" \
  --counts 20,50,100,200 $COMMON || exit 1

echo "=== Phase 3 완료 ==="
