#!/bin/sh
# Phase 2 — 모델 축: 전처리 상위 2개(zerophase-smooth / zerophase-shift) × 모델 7종.
# drop_missing=true 로 통일해 선형과 부스팅이 같은 행을 본다.
# 싼 모델부터. 앙상블은 재학습 없이 저장된 확률 평균으로 나중에 계산한다.
SMOOTH=data/features/b51ceabfb2b8.parquet
SHIFT=data/features/dfbb784c3c75.parquet
for feat in $SMOOTH $SHIFT; do
  for m in majority logreg_l2 logreg_l1 histgb lightgbm xgboost catboost; do
    echo "=== $feat $m ==="
    .venv/bin/sleepstage train --config configs/experiment/base.yaml \
      --set train.features=$feat --set train.model=$m --set train.drop_missing=true \
      || { echo "FAILED $m"; exit 1; }
  done
done
echo "=== Phase 2 완료 ==="
