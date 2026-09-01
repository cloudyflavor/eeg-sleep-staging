#!/bin/sh
# 특징을 하나씩 더하면서 성능이 얼마나 오르는지 잰다.
# 한꺼번에 넣으면 어느 것이 얼마나 기여했는지 알 수 없다.
#
# 0  기준선
# 1  5초 조각을 합칠 때 중앙값 대신 평균을 쓴다
# 2  조각 6개의 대역 세기 최댓값과 최솟값을 더한다
# 3  수면방추를 사건으로 세어 개수와 지속시간을 더한다
#
# 전처리는 앞서 우승한 조합으로 고정하고 모델은 catboost 로 고정한다.
# 결과는 runs/improve/<더한 것>/ 아래에 모인다. 축을 훑던 탐색 단계와 섞이지 않는다.
# 고정한 조건은 실행마다 manifest.json 의 condition 에 적힌다.
# 로그: logs/compare_feature_additions.log
mkdir -p logs
exec > logs/compare_feature_additions.log 2>&1

FIXED="--set features.filter=zerophase --set features.context=smooth"
TRAIN=".venv/bin/sleepstage train --config configs/experiment/base.yaml \
  --set train.model=catboost --set train.drop_missing=true"

run() {
  name=$1
  shift
  echo "=== $name ==="
  .venv/bin/sleepstage features --config configs/features/base.yaml --jobs 8 $FIXED "$@" \
    || { echo "실패: 특징 표 $name"; exit 1; }
  path=$(.venv/bin/sleepstage feature-path --config configs/features/base.yaml $FIXED "$@") || exit 1
  echo "특징 표 $path"
  $TRAIN --set train.features="$path" || { echo "실패: 학습 $name"; exit 1; }
}

run "0 기준선"
run "1 평균으로 합치기" --set features.welch_average=mean
run "2 조각별 최댓값 최솟값" --set features.welch_average=mean --set features.window_extremes=true
run "3 수면방추" --set features.welch_average=mean --set features.window_extremes=true \
  --set features.spindle_events=true
echo "=== 완료 ==="
