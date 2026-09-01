"""시스템 아키텍처 그림을 그립니다. 저장소 최상위에서 실행하세요."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _diagram import (
    FONT,
    MUTED,
    PALETTE,
    add,
    arrow,
    begin,
    curve,
    glyph_check,
    glyph_clock,
    glyph_cloud,
    glyph_db,
    glyph_device,
    glyph_file,
    glyph_folder,
    glyph_gauge,
    glyph_net,
    glyph_package,
    glyph_ratio,
    glyph_replay,
    glyph_slices,
    glyph_split,
    glyph_stream,
    glyph_table,
    glyph_tree,
    glyph_wave,
    group,
    legend,
    mid_badge,
    node,
    save,
    title,
)

W, H = 2020, 900

begin(W, H)
title(
    "EEG Sleep Staging, System Architecture",
    "Two-channel EEG, 30-second epochs, four sleep stages. "
    "The training path and the serving path share one feature implementation.",
)

R1, R2 = 250, 590

group(44, 150, 264, 480, "DATA SOURCE", PALETTE["source"])
node(154, R1, PALETTE["source"], glyph_wave, ["PSG EDF"], ["2 channels, 100 Hz"])
node(154, 380, PALETTE["source"], glyph_file, ["Hypnogram EDF"], ["30-second labels"])

group(304, 150, 524, 480, "PREPROCESSING", PALETTE["prep"])
node(414, R1, PALETTE["prep"], glyph_slices, ["prepare"], ["30-second epochs"])
node(414, 380, PALETTE["prep"], glyph_check, ["verify"], ["cross-check vs EDF"])

group(564, 150, 1004, 480, "FEATURE CORE", PALETTE["core"], dashed=False)
node(694, 240, PALETTE["core"], glyph_gauge, ["band power"], ["6 bands, ratios"])
node(874, 240, PALETTE["core"], glyph_net, ["waveform stats"], ["Hjorth, entropy"])
node(694, 350, PALETTE["core"], glyph_ratio, ["channel ratios"], ["front against back"])
node(874, 350, PALETTE["core"], glyph_clock, ["context"], ["past-only scaling"])

group(1044, 150, 1624, 350, "EXPERIMENT", PALETTE["exp"])
node(1144, R1, PALETTE["exp"], glyph_table, ["features"], ["289 columns"])
node(1334, R1, PALETTE["exp"], glyph_split, ["split"], ["subject-wise, 10 folds"])
node(1524, R1, PALETTE["exp"], glyph_tree, ["train"], ["cross-validation"])

group(1664, 150, 1974, 350, "MODEL ARTIFACT", PALETTE["art"])
add(
    f'<text x="1960" y="190" font-family="{FONT}" font-size="11.5" fill="{MUTED}" '
    f'text-anchor="end">written together by export</text>'
)
node(1754, R1, PALETTE["art"], glyph_package, ["model.cbm"], ["2.7 MB"])
node(1884, R1, PALETTE["art"], glyph_file, ["metadata"], ["columns, config, card"])

group(1044, 490, 1974, 890, "DOCKER IMAGE, REALTIME SERVING", PALETTE["serve"], title_right=True)
node(1189, R2, PALETTE["serve"], glyph_cloud, ["api"], ["FastAPI, one session per device"])
node(1509, R2, PALETTE["serve"], glyph_stream, ["stream"], ["runs the FEATURE CORE code"])
node(1829, R2, PALETTE["io"], glyph_clock, ["stage and confidence"], ["4.5-minute delay"])
node(1189, 760, PALETTE["art"], glyph_db, ["predictions"], ["SQLite, with an audit trail"])
node(1509, 760, PALETTE["serve"], glyph_replay, ["replay"], ["reproduce and compare"])
node(1829, 760, PALETTE["art"], glyph_folder, ["feature tables"], ["parquet per session"])

node(900, R2, PALETTE["io"], glyph_device, ["device"], ["one epoch every 30 s"])

legend(
    1690,
    56,
    [
        ("solid", "data flow"),
        ("dashed", "verification, not part of the request path"),
        (1, "order of the steps"),
    ],
)

# ---------------------------------------------------------------- edges
arrow([(183, R1), (385, R1)])
arrow([(183, 380), (250, 380), (250, R1), (385, R1)])
curve("M 388 272 C 338 300 338 342 392 353")
arrow([(443, R1), (505, R1), (505, 192), (563, 192)])
arrow([(1005, 192), (1024, 192), (1024, R1), (1115, R1)])
arrow([(1173, R1), (1305, R1)])
arrow([(1363, R1), (1495, R1)])
arrow([(1553, R1), (1656, R1)])
arrow([(929, R2), (1160, R2)])
arrow([(1218, R2), (1480, R2)])
arrow([(1538, R2), (1800, R2)])
arrow([(1165, 612), (1100, 612), (1100, 760), (1160, 760)])
arrow([(1538, 612), (1600, 612), (1600, 760), (1800, 760)])
arrow([(1480, 612), (1400, 612), (1400, 760), (1480, 760)], dashed=True)
arrow([(1754, 351), (1754, 440), (1189, 440), (1189, 561)])

mid_badge(264, 304, R1, 1)
mid_badge(524, 564, R1, 2)
mid_badge(1004, 1044, R1, 3)
mid_badge(1173, 1305, R1, 4)
mid_badge(1363, 1495, R1, 5)
mid_badge(1553, 1656, R1, 6)
mid_badge(929, 1160, R2, 7)
mid_badge(1218, 1480, R2, 8)
mid_badge(1538, 1800, R2, 9)

save("assets/architecture.svg")
