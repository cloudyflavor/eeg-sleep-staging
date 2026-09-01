#!/bin/sh
# 비교에 쓰는 특징 표를 전부 만든다.
# 필터 3가지와 문맥 3가지를 조합한 9개, 그리고 정규화를 뺀 1개다.
# 코드가 바뀌어 값이 달라졌으면 --overwrite 를 붙여 다시 만들어야 한다.
# 설정 해시만 보고 건너뛰기 때문에 설정이 같으면 옛 파일을 그대로 쓴다.
# 로그: logs/build_features.log
mkdir -p logs
exec > logs/build_features.log 2>&1
for f in none causal zerophase; do
  for c in none shift smooth; do
    echo "=== filter=$f context=$c ==="
    .venv/bin/sleepstage features --config configs/features/base.yaml --jobs 8 --overwrite \
      --set features.filter="$f" --set features.context="$c" \
      || { echo "실패: $f $c"; exit 1; }
  done
done
echo "=== filter=zerophase context=smooth normalize=none ==="
.venv/bin/sleepstage features --config configs/features/base.yaml --jobs 8 --overwrite \
  --set features.filter=zerophase --set features.context=smooth --set features.normalize=none \
  || { echo "실패: nonorm"; exit 1; }
echo "=== 완료 ==="
