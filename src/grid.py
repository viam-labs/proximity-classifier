"""Orthographic grid of a point cloud, for picking class coordinates.

The side view (yz) draws +Y to the left. In this cloud +Y is left when you
look along +X, so a plot that puts +Y on the right is a mirror of that view.
Color is the axis the picture collapses.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

WIDTH = 780
HEIGHT = 560

VIEWS = {
    "yz": ("y", "z", "x", True),
    "xz": ("x", "z", "y", False),
    "xy": ("x", "y", "z", False),
}
AXIS = {"x": 0, "y": 1, "z": 2}


@dataclass
class Layout:
    horizontal: str
    vertical: str
    color_axis: str
    positive_left: bool
    min_h: float
    max_h: float
    min_v: float
    max_v: float
    min_c: float
    max_c: float
    scale: float
    origin_x: int
    origin_y: int
    empty: bool


def layout_for(points: np.ndarray, view: str) -> Layout:
    horizontal, vertical, color_axis, positive_left = VIEWS[view]
    margin_left, margin_right, margin_top, margin_bottom = 52, 78, 36, 40
    plot_w = WIDTH - margin_left - margin_right
    plot_h = HEIGHT - margin_top - margin_bottom
    empty = points.size == 0
    if empty:
        min_h, max_h, min_v, max_v, min_c, max_c = -1.0, 1.0, -1.0, 1.0, 0.0, 1.0
    else:
        cols = points[:, [AXIS[horizontal], AXIS[vertical], AXIS[color_axis]]]
        min_h, min_v, min_c = cols.min(axis=0)
        max_h, max_v, max_c = cols.max(axis=0)
    span_h = float(max_h - min_h)
    span_v = float(max_v - min_v)
    if span_h < 1e-6:
        span_h = 1.0
        min_h -= 0.5
        max_h += 0.5
    if span_v < 1e-6:
        span_v = 1.0
        min_v -= 0.5
        max_v += 0.5
    # One scale keeps a class radius circular.
    scale = min(plot_w / span_h, plot_h / span_v)
    return Layout(
        horizontal=horizontal,
        vertical=vertical,
        color_axis=color_axis,
        positive_left=positive_left,
        min_h=float(min_h),
        max_h=float(max_h),
        min_v=float(min_v),
        max_v=float(max_v),
        min_c=float(min_c),
        max_c=float(max_c),
        scale=scale,
        origin_x=margin_left,
        origin_y=margin_top,
        empty=empty,
    )


def world_to_pixel(layout: Layout, point: np.ndarray) -> tuple[int, int]:
    h = float(point[AXIS[layout.horizontal]])
    v = float(point[AXIS[layout.vertical]])
    dx = (layout.max_h - h) * layout.scale if layout.positive_left else (h - layout.min_h) * layout.scale
    dy = (layout.max_v - v) * layout.scale
    return layout.origin_x + int(round(dx)), layout.origin_y + int(round(dy))


def render_grid(
    points: np.ndarray,
    classes: dict[str, tuple[float, float, float]],
    radius: float,
    view: str,
) -> bytes:
    layout = layout_for(points, view)
    image = Image.new("RGB", (WIDTH, HEIGHT), (248, 248, 248))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    title = f"{view} view: horizontal {layout.horizontal.upper()}, vertical {layout.vertical.upper()}, color is {layout.color_axis.upper()}"
    if layout.positive_left:
        title += f" (+{layout.horizontal.upper()} is left)"
    draw.text((8, 4), title, fill=(0, 0, 0), font=font)
    _draw_grid(draw, font, layout)
    _draw_color_bar(draw, font, layout)

    if not layout.empty:
        color_values = points[:, AXIS[layout.color_axis]]
        span = layout.max_c - layout.min_c
        for point, color_value in zip(points, color_values):
            x, y = world_to_pixel(layout, point)
            t = 0.0 if span <= 0 else (float(color_value) - layout.min_c) / span
            color = _color_for(t)
            draw.point((x, y), fill=color)
            if 0 <= x + 1 < WIDTH and 0 <= y + 1 < HEIGHT:
                draw.point((x + 1, y), fill=color)
                draw.point((x, y + 1), fill=color)
    else:
        draw.text((layout.origin_x, HEIGHT // 2), "no points in the cloud", fill=(80, 80, 80), font=font)

    for name, (x, y, z) in classes.items():
        px, py = world_to_pixel(layout, np.array([x, y, z], dtype=np.float64))
        r = max(4, int(round(radius * layout.scale)))
        draw.ellipse((px - r, py - r, px + r, py + r), outline=(220, 30, 50))
        draw.text((px + 6, py - 14), name, fill=(180, 20, 40), font=font)

    out = io.BytesIO()
    image.save(out, format="JPEG", quality=80)
    return out.getvalue()


def _draw_grid(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, layout: Layout) -> None:
    step_h = _nice_step(layout.max_h - layout.min_h)
    step_v = _nice_step(layout.max_v - layout.min_v)
    h = math.ceil(layout.min_h / step_h) * step_h
    while h <= layout.max_h + step_h * 0.01:
        x, y_top = world_to_pixel(layout, _on_axes(layout, h, layout.max_v))
        _, y_bot = world_to_pixel(layout, _on_axes(layout, h, layout.min_v))
        draw.line((x, y_top, x, y_bot), fill=(210, 210, 210))
        draw.text((x - 10, y_bot + 2), _format_tick(h, step_h), fill=(60, 60, 60), font=font)
        h += step_h
    v = math.ceil(layout.min_v / step_v) * step_v
    while v <= layout.max_v + step_v * 0.01:
        x_left, y = world_to_pixel(layout, _on_axes(layout, layout.min_h, v))
        x_right, _ = world_to_pixel(layout, _on_axes(layout, layout.max_h, v))
        draw.line((x_left, y, x_right, y), fill=(210, 210, 210))
        draw.text((8, y - 6), _format_tick(v, step_v), fill=(60, 60, 60), font=font)
        v += step_v


def _on_axes(layout: Layout, horizontal: float, vertical: float) -> np.ndarray:
    point = np.zeros(3, dtype=np.float64)
    point[AXIS[layout.horizontal]] = horizontal
    point[AXIS[layout.vertical]] = vertical
    return point


def _draw_color_bar(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, layout: Layout) -> None:
    x0 = WIDTH - 58
    y0 = layout.origin_y
    height = HEIGHT - 36 - 40
    for i in range(height):
        t = 1 - i / max(height - 1, 1)
        draw.line((x0, y0 + i, x0 + 14, y0 + i), fill=_color_for(t))
    step = _nice_step(layout.max_c - layout.min_c)
    draw.text((x0 - 4, y0 - 14), layout.color_axis.upper(), fill=(0, 0, 0), font=font)
    draw.text((x0 + 18, y0), _format_tick(layout.max_c, step), fill=(0, 0, 0), font=font)
    draw.text((x0 + 18, y0 + height - 12), _format_tick(layout.min_c, step), fill=(0, 0, 0), font=font)


def _color_for(t: float) -> tuple[int, int, int]:
    t = min(1.0, max(0.0, t))
    return (int(20 + 230 * t), int(50 + 160 * t), int(190 - 150 * t))


def _nice_step(span: float) -> float:
    if span <= 0:
        return 1.0
    raw = span / 8
    power = 10 ** math.floor(math.log10(raw))
    frac = raw / power
    if frac < 1.5:
        nice = 1
    elif frac < 3.5:
        nice = 2
    elif frac < 7.5:
        nice = 5
    else:
        nice = 10
    return nice * power


def _format_tick(value: float, step: float) -> str:
    if step >= 1:
        return f"{value:.0f}"
    return f"{value:.1f}"
