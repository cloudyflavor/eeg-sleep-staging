#!/bin/sh
# 두 채널이 얼마나 닮았는지를 더하면 REM 이 오르는지 잰다.
#
# 고령의 REM 이 병목이다. 80세 초과에서 REM 의 33.7 %가 얕은 잠으로 새고
# REM 확률 중앙값이 0.645 다. 40세 미만은 90.8 %를 맞히고 중앙값이 0.967 이다.
#
# REM 은 원래 안구운동으로 정의하는데 우리는 그 채널이 없다. 대신 눈이
# 전기적으로 건전지라 이마 전극에 흔적이 섞여 들어온다. 뒤통수에는 거의 안 섞인다.
#
# 지금 채널 비교 특징은 세기 비율뿐이라 "이마가 더 세다" 까지만 안다.
# 서파도 이마가 더 세고 안구운동도 이마가 더 세서 그것만으로는 안 갈린다.
# 파형이 닮았는지를 함께 보면 갈린다.
#
# 만들기 전에 판별력을 쟀다. R 대 LS 기준으로 시그마 0.686, 안구운동 대역 0.641,
# 저주파 0.631 이다. 함께 잰 안구운동 사건 세기와 톱니파 비대칭은 0.5 근처라 뺐다.
#
# 결과는 runs/improve/<더한 것>/ 아래에 모인다.
# 로그: logs/compare_rem_features.log
mkdir -p logs
exec > logs/compare_rem_features.log 2>&1

WON="--set features.filter=zerophase --set features.context=smooth \
  --set features.welch_average=mean --set features.window_extremes=true \
  --set features.spindle_events=true"

echo "=== 특징 표 만들기 ==="
.venv/bin/sleepstage features --config configs/features/base.yaml --jobs 8 \
  $WON --set features.channel_correlation=true || exit 1

F=$(.venv/bin/sleepstage feature-path --config configs/features/base.yaml \
  $WON --set features.channel_correlation=true) || exit 1
echo "특징 표 $F"

echo "=== 학습 ==="
.venv/bin/sleepstage train --config configs/experiment/base.yaml \
  --set train.features="$F" --set train.model=catboost --set train.drop_missing=true \
  --set train.threads=24 || exit 1
echo "=== 완료 ==="
