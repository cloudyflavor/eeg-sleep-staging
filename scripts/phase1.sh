#!/bin/sh
# Phase 1 — 전처리 축: 9개 특징 조합 × 10-fold, 모델은 HistGB 고정.
# 러닝 커브가 돌고 있으면 끝날 때까지 기다렸다가 시작한다.
# 싼 조합(97특징)부터 돌려 빨리 피드백을 얻고, 첫 조합 실패 시 즉시 멈춘다.
while pgrep -f "[r]un_curve.sh" > /dev/null; do sleep 60; done
echo "=== Phase 1 시작: 9조합 × 10fold ==="
for h in 181306a1fc8f 33776098d42b fa8bfcb1da03 \
         d7707650c0dd 91ad452df0fd b51ceabfb2b8 \
         31da369722a3 b402c3e2c9bf dfbb784c3c75; do
  echo "=== features=$h ==="
  .venv/bin/sleepstage train --config configs/experiment/base.yaml \
    --set train.features=data/features/$h.parquet || { echo "FAILED $h"; exit 1; }
done
echo "=== Phase 1 완료 ==="
