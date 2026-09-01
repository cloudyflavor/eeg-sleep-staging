"""그림을 그리는 공통 부분입니다.

아키텍처 그림 두 장이 같은 아이콘, 같은 색, 같은 간격을 쓰도록 여기에 모읍니다.
좌표를 코드로 두면 상자 하나를 옮길 때 전체 간격이 어긋나지 않습니다.
"""

from pathlib import Path

ICON = 58
FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"

INK = "#1F2933"
MUTED = "#5C6B7A"
LINE = "#3E4C59"

PALETTE = {
    "source": "#7B8794",
    "prep": "#2E7D9A",
    "core": "#2F6FB5",
    "exp": "#B8562F",
    "art": "#B03A48",
    "serve": "#2E7D5B",
    "io": "#3E4C59",
}

out: list[str] = []


def begin(width, height):
    out.clear()
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    )
    out.append(f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>')
    out.append(
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" '
        f'fill="{LINE}"/></marker></defs>'
    )


def save(path):
    out.append("</svg>")
    Path(path).write_text("\n".join(out), encoding="utf-8")
    print("생성:", path)


def add(s):
    out.append(s)


def title(text, subtitle):
    add(
        f'<text x="44" y="52" font-family="{FONT}" font-size="21" font-weight="700" '
        f'fill="{INK}">{text}</text>'
    )
    add(f'<text x="44" y="76" font-family="{FONT}" font-size="13" fill="{MUTED}">{subtitle}</text>')


def icon_box(x, y, color):
    add(
        f'<rect x="{x - ICON / 2}" y="{y - ICON / 2}" width="{ICON}" height="{ICON}" rx="9" '
        f'fill="{color}" />'
    )


def label(x, y, lines, size=13, weight="500", color=INK, anchor="middle"):
    for i, text in enumerate(lines):
        add(
            f'<text x="{x}" y="{y + i * (size + 3)}" font-family="{FONT}" font-size="{size}" '
            f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{text}</text>'
        )


def glyph_wave(x, y):
    add(
        f'<path d="M {x - 17} {y} q 5.5 -13 11 0 q 5.5 13 11 0 q 5.5 -13 11 0" '
        f'fill="none" stroke="#FFFFFF" stroke-width="2.4" stroke-linecap="round"/>'
    )


def glyph_file(x, y):
    add(
        f'<path d="M {x - 11} {y - 15} h 15 l 7 7 v 23 h -22 z" fill="none" stroke="#FFFFFF" '
        f'stroke-width="2.2" stroke-linejoin="round"/>'
    )
    add(f'<path d="M {x + 4} {y - 15} v 7 h 7" fill="none" stroke="#FFFFFF" stroke-width="2.2"/>')


def glyph_slices(x, y):
    for dx in (-13, -3, 7):
        add(
            f'<rect x="{x + dx}" y="{y - 13}" width="7" height="26" rx="2" fill="none" '
            f'stroke="#FFFFFF" stroke-width="2.2"/>'
        )


def glyph_check(x, y):
    add(f'<circle cx="{x}" cy="{y}" r="15" fill="none" stroke="#FFFFFF" stroke-width="2.2"/>')
    add(
        f'<path d="M {x - 7} {y} l 5 5 l 9 -10" fill="none" stroke="#FFFFFF" stroke-width="2.6" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
    )


def glyph_table(x, y):
    add(
        f'<rect x="{x - 16}" y="{y - 13}" width="32" height="26" rx="3" fill="none" '
        f'stroke="#FFFFFF" stroke-width="2.2"/>'
    )
    add(
        f'<path d="M {x - 16} {y - 4} h 32 M {x - 16} {y + 5} h 32 '
        f'M {x - 5} {y - 13} v 26 M {x + 6} {y - 13} v 26" '
        f'stroke="#FFFFFF" stroke-width="1.6"/>'
    )


def glyph_split(x, y):
    add(
        f'<path d="M {x - 16} {y} h 10 M {x - 6} {y} l 10 -10 h 12 M {x - 6} {y} l 10 10 h 12" '
        f'fill="none" stroke="#FFFFFF" stroke-width="2.4" stroke-linecap="round" '
        f'stroke-linejoin="round"/>'
    )


def glyph_tree(x, y):
    add(f'<circle cx="{x}" cy="{y - 11}" r="4.5" fill="#FFFFFF"/>')
    add(f'<circle cx="{x - 11}" cy="{y + 11}" r="4.5" fill="#FFFFFF"/>')
    add(f'<circle cx="{x + 11}" cy="{y + 11}" r="4.5" fill="#FFFFFF"/>')
    add(
        f'<path d="M {x - 2} {y - 8} l -7 15 M {x + 2} {y - 8} l 7 15" stroke="#FFFFFF" '
        f'stroke-width="2.2" stroke-linecap="round"/>'
    )


def glyph_net(x, y):
    pts = [(-14, -10), (-14, 10), (0, 0), (14, -10), (14, 10)]
    add(
        f'<path d="M {x - 14} {y - 10} L {x} {y} L {x + 14} {y - 10} M {x - 14} {y + 10} L {x} {y} '
        f'L {x + 14} {y + 10}" stroke="#FFFFFF" stroke-width="2" fill="none"/>'
    )
    for dx, dy in pts:
        add(f'<circle cx="{x + dx}" cy="{y + dy}" r="4" fill="#FFFFFF"/>')


def glyph_gauge(x, y):
    add(
        f'<path d="M {x - 15} {y + 8} a 15 15 0 0 1 30 0" fill="none" stroke="#FFFFFF" '
        f'stroke-width="2.4"/>'
    )
    add(
        f'<path d="M {x} {y + 8} l 10 -12" stroke="#FFFFFF" stroke-width="2.4" '
        f'stroke-linecap="round"/>'
    )


def glyph_package(x, y):
    add(
        f'<path d="M {x - 15} {y - 7} l 15 -8 l 15 8 v 15 l -15 8 l -15 -8 z" fill="none" '
        f'stroke="#FFFFFF" stroke-width="2.2" stroke-linejoin="round"/>'
    )
    add(
        f'<path d="M {x - 15} {y - 7} l 15 8 l 15 -8" fill="none" stroke="#FFFFFF" '
        f'stroke-width="2" stroke-linejoin="round"/>'
    )


def glyph_device(x, y):
    add(
        f'<rect x="{x - 15}" y="{y - 13}" width="30" height="20" rx="3" fill="none" '
        f'stroke="#FFFFFF" stroke-width="2.2"/>'
    )
    add(
        f'<path d="M {x - 8} {y + 13} h 16 M {x} {y + 7} v 6" stroke="#FFFFFF" stroke-width="2.2" '
        f'stroke-linecap="round"/>'
    )


def glyph_stream(x, y):
    for dx in (-12, -1, 10):
        add(
            f'<path d="M {x + dx} {y - 9} l 7 9 l -7 9" fill="none" stroke="#FFFFFF" '
            f'stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>'
        )


def glyph_clock(x, y):
    add(f'<circle cx="{x}" cy="{y}" r="15" fill="none" stroke="#FFFFFF" stroke-width="2.2"/>')
    add(
        f'<path d="M {x} {y - 8} v 9 l 7 4" fill="none" stroke="#FFFFFF" stroke-width="2.4" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
    )


def glyph_replay(x, y):
    add(
        f'<path d="M {x + 13} {y - 2} a 14 14 0 1 1 -5 -10" fill="none" stroke="#FFFFFF" '
        f'stroke-width="2.4" stroke-linecap="round"/>'
    )
    add(
        f'<path d="M {x + 14} {y - 15} v 9 h -9" fill="none" stroke="#FFFFFF" stroke-width="2.4" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
    )


def node(x, y, color, glyph, lines, sub=None):
    icon_box(x, y, color)
    glyph(x, y)
    label(x, y + ICON / 2 + 19, lines, size=13, weight="600")
    if sub:
        label(x, y + ICON / 2 + 19 + len(lines) * 16 + 2, sub, size=11.5, weight="400", color=MUTED)


def group(x1, y1, x2, y2, title, color, dashed=True, title_right=False):
    dash = ' stroke-dasharray="7 5"' if dashed else ""
    add(
        f'<rect x="{x1}" y="{y1}" width="{x2 - x1}" height="{y2 - y1}" rx="10" fill="none" '
        f'stroke="{color}" stroke-width="1.8"{dash}/>'
    )
    tx, anchor = (x2 - 14, "end") if title_right else (x1 + 14, "start")
    add(
        f'<text x="{tx}" y="{y1 + 21}" font-family="{FONT}" font-size="12.5" font-weight="700" '
        f'fill="{color}" letter-spacing="0.6" text-anchor="{anchor}">{title}</text>'
    )


def arrow(pts, dashed=False, color=LINE, label_text=None, label_at=None, label_dy=-8):
    """직각으로 꺾이는 화살표. 글씨를 지나갈 자리에는 curve 를 쓴다."""
    d = f"M {pts[0][0]} {pts[0][1]} " + " ".join(f"L {x} {y}" for x, y in pts[1:])
    dash = ' stroke-dasharray="6 5"' if dashed else ""
    add(
        f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.9"{dash} '
        f'marker-end="url(#arrow)" stroke-linejoin="round"/>'
    )
    if label_text:
        lx, ly = label_at
        add(
            f'<text x="{lx}" y="{ly + label_dy}" font-family="{FONT}" font-size="11.5" '
            f'fill="{MUTED}" text-anchor="middle">{label_text}</text>'
        )


def badge(x, y, n):
    add(f'<circle cx="{x}" cy="{y}" r="12" fill="{INK}"/>')
    add(
        f'<text x="{x}" y="{y + 4.5}" font-family="{FONT}" font-size="12.5" font-weight="700" '
        f'fill="#FFFFFF" text-anchor="middle">{n}</text>'
    )


def curve(d, dashed=False):
    dash = ' stroke-dasharray="6 5"' if dashed else ""
    add(
        f'<path d="{d}" fill="none" stroke="{LINE}" stroke-width="1.9"{dash} '
        f'marker-end="url(#arrow)"/>'
    )


def mid_badge(x1, x2, y, n, above=25):
    badge((x1 + x2) / 2, y - above, n)


def glyph_ratio(x, y):
    """두 채널을 나눠 비교한다는 뜻으로 위아래 파형과 그 사이 가로선을 그린다."""
    add(
        f'<path d="M {x - 15} {y - 9} q 5 -7 10 0 q 5 7 10 0" fill="none" stroke="#FFFFFF" '
        f'stroke-width="2.2" stroke-linecap="round"/>'
    )
    add(
        f'<path d="M {x - 16} {y} h 32" stroke="#FFFFFF" stroke-width="2.4" '
        f'stroke-linecap="round"/>'
    )
    add(
        f'<path d="M {x - 15} {y + 11} q 5 -7 10 0 q 5 7 10 0" fill="none" stroke="#FFFFFF" '
        f'stroke-width="2.2" stroke-linecap="round"/>'
    )


def glyph_db(x, y):
    """원통. 데이터베이스를 뜻하는 관행적인 모양이다."""
    add(
        f'<path d="M {x - 13} {y - 11} v 22 a 13 5 0 0 0 26 0 v -22" fill="none" '
        f'stroke="#FFFFFF" stroke-width="2.2"/>'
    )
    add(
        f'<ellipse cx="{x}" cy="{y - 11}" rx="13" ry="5" fill="none" stroke="#FFFFFF" '
        f'stroke-width="2.2"/>'
    )
    add(
        f'<path d="M {x - 13} {y} a 13 5 0 0 0 26 0" fill="none" stroke="#FFFFFF" '
        f'stroke-width="1.6"/>'
    )


def glyph_folder(x, y):
    add(
        f'<path d="M {x - 16} {y + 12} v -22 h 11 l 4 5 h 17 v 17 z" fill="none" '
        f'stroke="#FFFFFF" stroke-width="2.2" stroke-linejoin="round"/>'
    )


def glyph_cloud(x, y):
    add(
        f'<path d="M {x - 16} {y + 6} a 9 9 0 0 1 2 -17 a 12 12 0 0 1 22 -3 a 8 8 0 0 1 8 20 z" '
        f'fill="none" stroke="#FFFFFF" stroke-width="2.2" stroke-linejoin="round"/>'
    )


def legend(x, y, items):
    """선 모양이 무엇을 뜻하는지 적어둔다. 없으면 점선을 각자 다르게 읽는다."""
    for i, (kind, text) in enumerate(items):
        yy = y + i * 22
        if kind == "solid":
            add(
                f'<path d="M {x} {yy} h 34" stroke="{LINE}" stroke-width="1.9" '
                f'marker-end="url(#arrow)"/>'
            )
        elif kind == "dashed":
            add(
                f'<path d="M {x} {yy} h 34" stroke="{LINE}" stroke-width="1.9" '
                f'stroke-dasharray="6 5" marker-end="url(#arrow)"/>'
            )
        else:
            badge(x + 12, yy, kind)
        add(
            f'<text x="{x + 46}" y="{yy + 4}" font-family="{FONT}" font-size="11.5" '
            f'fill="{MUTED}">{text}</text>'
        )
