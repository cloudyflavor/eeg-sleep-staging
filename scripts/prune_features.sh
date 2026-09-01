#!/bin/sh
# 열을 줄이면 성능이 오르는지 잰다.
#
# 505개까지 늘려 온 특징이 서로 많이 닮아 있다. 같은 밑동의 앞뒤 평균과 최근
# 평균은 중앙값 0.947 로 닮았고, 절대 상관 0.9 를 넘는 짝이 596개다. 닮은 열이
# 많으면 나무가 매번 비슷한 열 중 하나를 집어 중요도가 흩어지고, 어느 것도
# 충분히 깊게 쓰이지 않는다.
#
# 줄이는 방법을 두 갈래로 나눈다.
#
# 이름만 보고 거르는 쪽은 어느 행을 보든 결과가 같아 누수가 없다.
#   no_p2min       최근 2분 평균을 뺀다, 337열
#   no_extremes    조각 안 최댓값과 최솟값을 뺀다, 361열
#   smoothed_only  평활 안 한 원본을 뺀다, 337열
#
# 값을 보고 고르는 쪽은 반드시 묶음 안에서 학습 행만 본다. 평가 행을 보고
# 고르면 점수가 부풀고 실행해도 안 보인다.
#   top:200 top:300      부스팅 이득 순으로 앞에서 자른다
#   decorr:0.95 0.90     닮은 무리마다 가장 쓸모 있는 하나만 남긴다
#
# 기준선은 같은 표에 열을 다 쓴 improve/mean+extremes+spindles+corr 이고
# macro-F1 0.8462 다.
#
# 결과는 runs/improve/mean+extremes+spindles+corr-<줄인 것>/ 에 모인다.
# 로그: logs/prune_<줄인 것>.log
mkdir -p logs

WON="--set features.filter=zerophase --set features.context=smooth \
  --set features.welch_average=mean --set features.window_extremes=true \
  --set features.spindle_events=true --set features.channel_correlation=true"

F=$(.venv/bin/sleepstage feature-path --config configs/features/base.yaml $WON) || exit 1
echo "특징 표 $F"

# 40코어를 셋으로 나눈다. 공용 서버라 다 쓰지 않는다.
THREADS=13

run() {
    tag=$1
    shift
    .venv/bin/sleepstage train --config configs/experiment/base.yaml \
        --set train.features="$F" --set train.model=catboost \
        --set train.drop_missing=true --set train.threads=$THREADS \
        "$@" > "logs/prune_$tag.log" 2>&1
    echo "끝 $tag  $(grep -o 'macro-F1 [0-9.]*' "logs/prune_$tag.log" | tail -1)"
}

echo "=== 1차 ==="
run no_p2min --set train.feature_subset=no_p2min &
run no_extremes --set train.feature_subset=no_extremes &
run smoothed_only --set train.feature_subset=smoothed_only &
wait

echo "=== 2차 ==="
run top200 --set train.feature_select=top:200 &
run top300 --set train.feature_select=top:300 &
run decorr095 --set train.feature_select=decorr:0.95 &
wait

echo "=== 3차 ==="
run decorr090 --set train.feature_select=decorr:0.90 &
wait

echo "=== 완료 ==="
