#!/bin/sh
# 사람에 대해 원래 아는 값을 넣으면 오르는지 잰다.
#
# 성능이 나이와 크게 엮여 있다. 녹음별 macro-F1 과 나이의 순위상관이 -0.51 이고
# 80세 이상이 0.711, 40세 미만이 0.868 이다. 그래서 두 방향을 본다.
#
#   1  나이를 특징으로 넣어 모델이 직접 쓰게 한다
#   2  나이와 성별을 함께 넣는다
#
# 나이 구간별 가중치는 뺐다. 구간마다 조각 수가 이미 비슷해서 무게가 ±15% 밖에
# 안 움직인다. 고령이 학습에서 밀려서 못 하는 것이 아니라 그냥 더 어렵다.
#
# 특징 표는 앞선 실험에서 우승한 것으로 고정한다. 표를 다시 만들지 않는다.
# 나이와 성별은 신호에서 뽑은 값이 아니라 밖에서 가져온 값이라 학습 단계에서 붙인다.
#
# 결과는 runs/improve/<더한 것>/ 아래에 모인다.
# 로그: logs/compare_subject_info.log
mkdir -p logs
exec > logs/compare_subject_info.log 2>&1

FIXED="--set features.filter=zerophase --set features.context=smooth \
  --set features.welch_average=mean --set features.window_extremes=true \
  --set features.spindle_events=true"
F=$(.venv/bin/sleepstage feature-path --config configs/features/base.yaml $FIXED) || exit 1
echo "특징 표 $F"
# 두 실험은 서로 독립이라 동시에 돌린다. 40코어 중 12개씩 나눠 쓴다.
# 공용 서버라 전부 쓰지 않고, 스레드 수는 결과를 바꾸지 않는다.
T=".venv/bin/sleepstage train --config configs/experiment/base.yaml \
   --set train.features=$F --set train.model=catboost --set train.drop_missing=true \
   --set train.threads=12"

echo "=== 0 기준선, 앞 실험의 우승 조합 ==="
$T || exit 1

echo "=== 1 나이 와 2 나이+성별 을 동시에 ==="
$T --set train.subject_info=age > logs/subject_age.log 2>&1 &
age=$!
$T --set train.subject_info=age+sex > logs/subject_age_sex.log 2>&1 &
sex=$!
wait $age || { echo "FAILED age"; exit 1; }
wait $sex || { echo "FAILED age+sex"; exit 1; }
tail -4 logs/subject_age.log logs/subject_age_sex.log
echo "=== 완료 ==="
