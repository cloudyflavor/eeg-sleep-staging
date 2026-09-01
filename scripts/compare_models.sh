#!/bin/sh
# 앞선 비교에서 우승한 특징 표에 모델 여섯 종을 붙여 비교한다.
# 결측 행 처리를 통일해서 선형 모델과 부스팅이 같은 행을 보게 한다.
# 계산이 싼 모델부터 돌린다. 앙상블은 저장된 확률을 평균해 나중에 계산한다.
# lasso 는 뺐다. 19만 행에서 좌표하강법이 2시간 넘게 걸리는데
# 선형 비교군은 ridge 하나로 충분하다.
#
# 앞뒤 값 붙이기 표도 뺐다. 전처리 축 실험에서 앞뒤 평균과 0.2%p 차이였고
# 열이 481개라 계산만 무겁다. 배포에 쓰는 표에서만 모델을 비교한다.
# 로그: logs/compare_models.log
mkdir -p logs
exec > logs/compare_models.log 2>&1
SMOOTH=$(.venv/bin/sleepstage feature-path --config configs/features/base.yaml \
  --set features.filter=zerophase --set features.context=smooth) || exit 1
echo "특징 표 $SMOOTH"
for m in majority ridge histgb lightgbm xgboost catboost; do
  echo "=== $m ==="
  .venv/bin/sleepstage train --config configs/experiment/base.yaml \
    --set train.features="$SMOOTH" --set train.model=$m --set train.drop_missing=true \
    || { echo "FAILED $m"; exit 1; }
done
echo "=== 완료 ==="
