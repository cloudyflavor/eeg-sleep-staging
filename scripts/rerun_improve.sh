#!/bin/sh
# 결측 행을 버리지 않고 개선 계열을 다시 잰다.
#
# 왜 다시 재는가. 방추 개수는 대부분 0 이고, 0 만 이어지면 사람별 기준으로 바꿀 때
# 위아래 폭이 0 이 되어 그 칸이 빈다. 설정이 빈칸 있는 줄을 통째로 버리게 돼 있어서
# 방추를 넣은 실행만 19만 5천 줄 중 1만 줄이 빠졌다.
#
# 빠진 줄이 어려운 줄이라 시험 범위가 쉬워졌다. 같은 행에서 다시 재보면 방추의
# 몫이 +0.30%p 에서 -0.04%p 가 된다. 즉 방추가 올린 것이 아니라 어려운 문제가
# 시험에서 빠진 것이었다.
#
# 나이와 성별을 넣은 실행은 다시 안 돌린다. 방추를 넣은 표에서 그대로 비교했으므로
# 평가 행이 이미 같았고, 효과가 없다는 결론이 그 결함과 무관하다.
#
# 부스팅은 결측을 그대로 받는다. drop_missing 을 끄면 모든 실행이 같은
# 195,479 줄로 학습하고 같은 줄에서 채점받는다.
#
# 결과는 runs/improve/<더한 것>/ 아래 새 이름으로 들어간다. 옛 실행은 남는다.
# 로그: logs/rerun_<이름>.log
mkdir -p logs

BASE="--set features.filter=zerophase --set features.context=smooth"
MEAN="$BASE --set features.welch_average=mean"
EXT="$MEAN --set features.window_extremes=true"
SPI="$EXT --set features.spindle_events=true"

path() {
    .venv/bin/sleepstage feature-path --config configs/features/base.yaml "$@"
}

# 공용 서버다. 40코어 중 12개까지만 쓰고 한 번에 하나씩 돌린다.
# 우선순위도 낮춰 다른 사람이 필요할 때 양보하게 한다.
THREADS=12

run() {
    tag=$1
    table=$2
    shift 2
    nice -n 15 .venv/bin/sleepstage train --config configs/experiment/base.yaml \
        --set train.features="$table" --set train.model=catboost \
        --set train.drop_missing=false --set train.threads=$THREADS \
        "$@" > "logs/rerun_$tag.log" 2>&1
    echo "끝 $tag  $(grep -m1 '^macro-F1' "logs/rerun_$tag.log")"
}

T_BASE=$(path $BASE) || exit 1
T_MEAN=$(path $MEAN) || exit 1
T_EXT=$(path $EXT) || exit 1
T_SPI=$(path $SPI) || exit 1
echo "표  기준선 $T_BASE  mean $T_MEAN  극단값 $T_EXT  방추 $T_SPI"

run base "$T_BASE"
run mean "$T_MEAN"
run extremes "$T_EXT"
run spindles "$T_SPI"

echo "=== 완료 ==="
