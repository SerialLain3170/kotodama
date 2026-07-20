"""Pluggable VN textbox styles (PIL, for correct Japanese rendering).

Each style is a (prepare, draw) pair registered in STYLES:
  prepare(dialogue, canvas_size) -> a context dict the draw fn needs (wrapped
    lines, fonts, whatever) — computed once per shot, not per frame.
  draw(draw, canvas_size, scene, character_name, t, ctx) -> draws the style
    (or nothing, before scene.text_start) onto `draw`'s image for time t.

Add a new style by writing a prepare/draw pair and adding it to STYLES.

All per-style pixel constants (margins, font sizes, box heights...) are tuned
for REFERENCE_SIZE and scaled by canvas_size[0]/REFERENCE_SIZE[0] wherever they
are used, so a style can be rendered at a small "native" resolution and then
nearest-neighbor upscaled by the caller to get pixelated, retro-bitmap-style
text that matches a pixelated background's resolution instead of looking like
crisp modern type pasted on top of it (see illustration_shot.py's
pixelate_text_native_width).
"""
from PIL import ImageFont

SANS_REGULAR = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
SANS_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
SERIF_REGULAR = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"

REFERENCE_SIZE = (1280, 720)
CHARS_PER_SEC = 12.0


def _scale(canvas_size: tuple[int, int]) -> float:
    return canvas_size[0] / REFERENCE_SIZE[0]


def wrap_text(draw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    lines, current = [], ""
    for ch in text:
        trial = current + ch
        if draw.textlength(trial, font=font) > max_width and current:
            lines.append(current)
            current = ch
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def _draw_revealed_lines(draw, wrapped_lines: list[str], n_shown: int, x: int, y: int, line_height: int, font, fill) -> None:
    remaining = n_shown
    for line in wrapped_lines:
        if remaining <= 0:
            break
        draw.text((x, y), line[:remaining], font=font, fill=fill)
        remaining -= len(line)
        y += line_height


# ---------------------------------------------------------------------------
# "default": classic ADV-style opaque bottom box with a name label.
# ---------------------------------------------------------------------------
_DEFAULT_BOX_HEIGHT = 180
_DEFAULT_MARGIN = 24
_DEFAULT_NAME_SIZE = 28
_DEFAULT_TEXT_SIZE = 34


def _prepare_default(dialogue: str, canvas_size: tuple[int, int]) -> dict:
    from PIL import Image, ImageDraw

    s = _scale(canvas_size)
    w, _ = canvas_size
    margin = _DEFAULT_MARGIN * s
    name_font = ImageFont.truetype(SANS_REGULAR, round(_DEFAULT_NAME_SIZE * s))
    text_font = ImageFont.truetype(SANS_REGULAR, round(_DEFAULT_TEXT_SIZE * s))
    dummy_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    wrapped = wrap_text(dummy_draw, dialogue, text_font, w - 2 * margin - 20 * s)
    return {"wrapped": wrapped, "name_font": name_font, "text_font": text_font}


def _draw_default(draw, canvas_size, scene, character_name, t, ctx) -> None:
    if t < scene.text_start:
        return
    s = _scale(canvas_size)
    w, h = canvas_size
    margin = _DEFAULT_MARGIN * s
    box_top = h - _DEFAULT_BOX_HEIGHT * s
    draw.rounded_rectangle(
        [margin, box_top, w - margin, h - margin],
        radius=round(16 * s), fill=(10, 10, 20, 200),
    )
    draw.text((margin + 20 * s, box_top + 14 * s), character_name, font=ctx["name_font"], fill=(255, 210, 120, 255))
    n_shown = min(len(scene.dialogue), int(max(0.0, t - scene.text_start) * CHARS_PER_SEC))
    y = box_top + 14 * s + _DEFAULT_NAME_SIZE * s + 16 * s
    _draw_revealed_lines(draw, ctx["wrapped"], n_shown, margin + 20 * s, y, (_DEFAULT_TEXT_SIZE + 10) * s, ctx["text_font"], (255, 255, 255, 255))


# ---------------------------------------------------------------------------
# "leaf_nvl": Leaf-style (Shizuku/Kizuato, 1996) full-screen novel mode.
# No textbox chrome: the whole frame dims, dialogue is quoted with corner
# brackets and set in a gothic bitmap-era-style font, and the speaker name (if
# any) is a small unboxed, muted label rather than a UI tag — these games
# rarely showed a name plate at all, relying on the prose.
# ---------------------------------------------------------------------------
_NVL_MARGIN_X = 90
_NVL_MARGIN_TOP = 50
_NVL_TEXT_SIZE = 32
_NVL_NAME_SIZE = 22
_NVL_DIM_ALPHA = 150


def _prepare_leaf_nvl(dialogue: str, canvas_size: tuple[int, int]) -> dict:
    from PIL import Image, ImageDraw

    s = _scale(canvas_size)
    w, _ = canvas_size
    text_font = ImageFont.truetype(SANS_REGULAR, round(_NVL_TEXT_SIZE * s))
    name_font = ImageFont.truetype(SANS_REGULAR, round(_NVL_NAME_SIZE * s))
    quoted = dialogue if dialogue.startswith("「") else f"「{dialogue}」"
    dummy_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    wrapped = wrap_text(dummy_draw, quoted, text_font, w - 2 * _NVL_MARGIN_X * s)
    return {"wrapped": wrapped, "quoted": quoted, "text_font": text_font, "name_font": name_font}


def _draw_leaf_nvl(draw, canvas_size, scene, character_name, t, ctx) -> None:
    if t < scene.text_start:
        return
    s = _scale(canvas_size)
    w, h = canvas_size
    draw.rectangle([0, 0, w, h], fill=(0, 0, 0, _NVL_DIM_ALPHA))

    margin_x = _NVL_MARGIN_X * s
    line_height = (_NVL_TEXT_SIZE + 14) * s
    name_h = _NVL_NAME_SIZE * s + 16 * s
    y0 = _NVL_MARGIN_TOP * s + name_h

    draw.text((margin_x, _NVL_MARGIN_TOP * s), character_name, font=ctx["name_font"], fill=(190, 190, 190, 230))

    n_shown = min(len(ctx["quoted"]), int(max(0.0, t - scene.text_start) * CHARS_PER_SEC))
    _draw_revealed_lines(draw, ctx["wrapped"], n_shown, margin_x, y0, line_height, ctx["text_font"], (255, 255, 255, 255))


# ---------------------------------------------------------------------------
# "ddlc": Doki Doki Literature Club-style opaque cream box with a colored
# rounded name tag overlapping its top edge.
# ---------------------------------------------------------------------------
_DDLC_BOX_HEIGHT = 170
_DDLC_MARGIN = 40
_DDLC_TEXT_SIZE = 32
_DDLC_NAME_SIZE = 26
_DDLC_NAME_COLOR = (255, 158, 205)  # pastel pink
_DDLC_BOX_FILL = (250, 244, 234, 255)
_DDLC_TEXT_COLOR = (70, 55, 60, 255)


def _prepare_ddlc(dialogue: str, canvas_size: tuple[int, int]) -> dict:
    from PIL import Image, ImageDraw

    s = _scale(canvas_size)
    w, _ = canvas_size
    text_font = ImageFont.truetype(SANS_BOLD, round(_DDLC_TEXT_SIZE * s))
    name_font = ImageFont.truetype(SANS_BOLD, round(_DDLC_NAME_SIZE * s))
    dummy_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    wrapped = wrap_text(dummy_draw, dialogue, text_font, w - 2 * _DDLC_MARGIN * s - 40 * s)
    return {"wrapped": wrapped, "text_font": text_font, "name_font": name_font}


def _draw_ddlc(draw, canvas_size, scene, character_name, t, ctx) -> None:
    if t < scene.text_start:
        return
    s = _scale(canvas_size)
    w, h = canvas_size
    margin = _DDLC_MARGIN * s
    box_top = h - _DDLC_BOX_HEIGHT * s

    name_font = ctx["name_font"]
    name_w = draw.textlength(character_name, font=name_font)
    pill_pad_x, pill_h = 22 * s, _DDLC_NAME_SIZE * s + 20 * s
    pill_left = margin + 24 * s
    pill_top = box_top - pill_h + 14 * s

    draw.rounded_rectangle(
        [margin, box_top, w - margin, h - margin / 2],
        radius=round(14 * s), fill=_DDLC_BOX_FILL, outline=(225, 210, 215, 255), width=max(1, round(3 * s)),
    )
    draw.rounded_rectangle(
        [pill_left, pill_top, pill_left + name_w + 2 * pill_pad_x, pill_top + pill_h],
        radius=round(pill_h / 2), fill=_DDLC_NAME_COLOR,
    )
    draw.text((pill_left + pill_pad_x, pill_top + 8 * s), character_name, font=name_font, fill=(255, 255, 255, 255))

    n_shown = min(len(scene.dialogue), int(max(0.0, t - scene.text_start) * CHARS_PER_SEC))
    y = box_top + pill_h - 4 * s
    _draw_revealed_lines(draw, ctx["wrapped"], n_shown, margin + 24 * s, y, (_DDLC_TEXT_SIZE + 12) * s, ctx["text_font"], _DDLC_TEXT_COLOR)


STYLES = {
    "default": (_prepare_default, _draw_default),
    "leaf_nvl": (_prepare_leaf_nvl, _draw_leaf_nvl),
    "ddlc": (_prepare_ddlc, _draw_ddlc),
}


def make_text_context(style: str, dialogue: str, canvas_size: tuple[int, int]) -> dict:
    prepare_fn, _ = STYLES[style]
    return prepare_fn(dialogue, canvas_size)


def draw_textbox(style: str, draw, canvas_size: tuple[int, int], scene, character_name: str, t: float, ctx: dict) -> None:
    _, draw_fn = STYLES[style]
    draw_fn(draw, canvas_size, scene, character_name, t, ctx)
