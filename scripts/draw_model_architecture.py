"""모델 세 종류가 같은 조각에서 출발해 어디서 갈라지는지를 그립니다.

저장소 최상위에서 실행하면 assets/models.svg 를 다시 만듭니다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _diagram import (
    PALETTE,
    arrow,
    begin,
    glyph_clock,
    glyph_gauge,
    glyph_net,
    glyph_package,
    glyph_ratio,
    glyph_slices,
    glyph_split,
    glyph_stream,
    glyph_table,
    glyph_tree,
    glyph_wave,
    group,
    legend,
    node,
    save,
    title,
)

W, H = 2000, 1060

begin(W, H)
title(
    "EEG Sleep Staging, Model Architectures",
    "All three read the same 30-second epoch. They differ in what the model receives.",
)

LANE_A, LANE_B, LANE_C = 250, 560, 880

# ---------------------------------------------------------------- 공통 입력
node(140, 560, PALETTE["io"], glyph_wave, ["30-second epoch"], ["2 channels x 3000", "samples"])

# ---------------------------------------------------------------- A. 부스팅
group(300, 150, 1830, 370, "BOOSTING, HAND-DESIGNED FEATURES", PALETTE["exp"])
node(410, LANE_A, PALETTE["core"], glyph_gauge, ["epoch features"], ["about 50 per epoch"])
node(650, LANE_A, PALETTE["core"], glyph_clock, ["normalize"], ["expanding, past only"])
node(890, LANE_A, PALETTE["core"], glyph_net, ["context"], ["7.5-min average", "2-min trailing"])
node(1130, LANE_A, PALETTE["exp"], glyph_table, ["feature row"], ["289 columns"])
node(1400, LANE_A, PALETTE["exp"], glyph_tree, ["catboost"], ["gradient-boosted trees"])
node(1700, LANE_A, PALETTE["io"], glyph_slices, ["four probabilities"], ["W, LS, DS, R"])

arrow([(169, 545), (240, 545), (240, LANE_A), (381, LANE_A)])
arrow([(439, LANE_A), (621, LANE_A)])
arrow([(679, LANE_A), (861, LANE_A)])
arrow([(919, LANE_A), (1101, LANE_A)])
arrow([(1159, LANE_A), (1371, LANE_A)])
arrow([(1429, LANE_A), (1671, LANE_A)])

# ---------------------------------------------------------------- B. CNN
group(300, 430, 1830, 700, "CNN, RAW SIGNAL", PALETTE["prep"])
node(410, LANE_B, PALETTE["prep"], glyph_wave, ["raw signal"], ["no features computed"])
node(700, 510, PALETTE["prep"], glyph_ratio, ["small kernel"], ["0.5 s, spindles"])
node(700, 620, PALETTE["prep"], glyph_ratio, ["large kernel"], ["4 s, slow waves"])
node(990, LANE_B, PALETTE["prep"], glyph_package, ["concatenate"], ["2048 values"])
node(1260, LANE_B, PALETTE["prep"], glyph_split, ["dropout, linear"], ["one epoch decides"])
node(1700, LANE_B, PALETTE["io"], glyph_slices, ["four probabilities"], ["W, LS, DS, R"])

arrow([(169, LANE_B), (381, LANE_B)])
arrow([(439, LANE_B), (560, LANE_B), (560, 510), (671, 510)])
arrow([(439, LANE_B), (560, LANE_B), (560, 620), (671, 620)])
arrow([(729, 510), (910, 510), (910, LANE_B), (961, LANE_B)])
arrow([(729, 620), (910, 620), (910, LANE_B), (961, LANE_B)])
arrow([(1019, LANE_B), (1231, LANE_B)])
arrow([(1289, LANE_B), (1671, LANE_B)])

# ---------------------------------------------------------------- C. LSTM
group(300, 760, 1830, 1010, "CNN AND LSTM, SEQUENCE", PALETTE["serve"])
node(410, LANE_C, PALETTE["serve"], glyph_stream, ["nine epochs"], ["4.5 minutes"])
node(700, LANE_C, PALETTE["prep"], glyph_package, ["same CNN encoder"], ["applied per epoch"])
node(990, LANE_C, PALETTE["serve"], glyph_table, ["nine vectors"], ["2048 each"])
node(1260, LANE_C, PALETTE["serve"], glyph_net, ["bidirectional LSTM"], ["hidden 128"])
node(1520, LANE_C, PALETTE["serve"], glyph_split, ["linear per epoch"], ["middle one is scored"])
node(1700, LANE_C, PALETTE["io"], glyph_slices, ["four probabilities"], ["W, LS, DS, R"])

arrow([(169, 575), (240, 575), (240, LANE_C), (381, LANE_C)])
arrow([(439, LANE_C), (671, LANE_C)])
arrow([(729, LANE_C), (961, LANE_C)])
arrow([(1019, LANE_C), (1231, LANE_C)])
arrow([(1289, LANE_C), (1491, LANE_C)])
arrow([(1549, LANE_C), (1671, LANE_C)])

# 같은 인코더를 쓴다는 것을 세로로 잇는다
arrow(
    [(700, 706), (700, 851)],
    dashed=True,
    label_text="same encoder",
    label_at=(776, 782),
    label_dy=0,
)

legend(
    1560,
    56,
    [
        ("solid", "data flow"),
        ("dashed", "same code reused"),
    ],
)

save("assets/models.svg")
