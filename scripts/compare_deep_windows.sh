#!/bin/sh
# 앞뒤로 몇 조각까지 함께 볼 때 성능이 어떻게 달라지는지 잰다.
# 창 1 은 조각 하나만 보는 경우이고, 창 15 는 부스팅의 평활 창과 크기와 지연이 같다.
# 창을 겹치지 않게 잘라서 조각당 학습 노출을 창 크기와 무관하게 같게 둔다.
# 로그: logs/compare_deep_windows.log
mkdir -p logs
exec > logs/compare_deep_windows.log 2>&1
for w in 1 9 15 21 31; do
  [ "$w" = 1 ] && len=0 || len=$w
  for f in 0 1 2 3 4 5 6 7 8 9; do
    echo "=== window $w fold $f ==="
    .venv/bin/sleepstage deep --config configs/experiment/deep.yaml --fold "$f" \
      --set deep.sequence_length="$len" --set deep.sequence_stride="$w" \
      || { echo "실패: window $w fold $f"; exit 1; }
  done
done
echo "=== 완료 ==="
