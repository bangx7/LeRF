"""``draw_reference_frame``: render a predicted object-centric frame onto an image (pure PIL)."""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageDraw, ImageFont, ImageOps

TOOL_NAME = "draw_reference_frame"

AXIS_KEYS = ("axis_left", "axis_up", "axis_front")
POINT_KEYS = ("origin", *AXIS_KEYS)
REQUIRED_PARAMS = frozenset(("reference_object", *POINT_KEYS))

AXIS_STYLES: dict[str, dict] = {
    "axis_left": {"color": (40, 210, 70, 255), "label": "L"},
    "axis_up": {"color": (55, 125, 255, 255), "label": "U"},
    "axis_front": {"color": (245, 55, 55, 255), "label": "F"},
}
ORIGIN_COLOR = (255, 220, 40, 255)

# An axis shorter than EPS_REL * min(W, H) px is treated as foreshortened (degenerate).
EPS_REL = 0.02
# Right-handed frame: each axis is the cross product of the other two (cyclic).
HANDEDNESS_PAIRS = {
    "axis_front": ("axis_left", "axis_up"),
    "axis_left": ("axis_up", "axis_front"),
    "axis_up": ("axis_front", "axis_left"),
}
# Below this |sin| the two visible axes are too collinear to infer the depth sign.
COLLINEAR_SIN_MIN = 0.15

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    "/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf",
    "/usr/share/fonts/google-droid-sans-fonts/DroidSans-Bold.ttf",
    "/usr/share/fonts/urw-base35/NimbusSans-Bold.otf",
)

# Fixed messages returned to the model when a call is rejected.
RETRY_MESSAGES: dict[str, str] = {
    "params": (
        "The call must contain exactly these five parameters, once each: "
        "reference_object, origin, axis_up, axis_front, axis_left. Reply again "
        "with all five."
    ),
    "empty_reference_object": ("The reference_object parameter must name the object. Reply again with it filled in."),
    "coord_shape": (
        "origin, axis_up, axis_front and axis_left must each hold exactly two "
        "numbers, written as [x, y]. Reply again in that format."
    ),
    "coord_not_int": ("Coordinates must be integers, not decimals or strings. Reply again with integer coordinates."),
    "coord_out_of_range": (
        "Coordinates are normalized to 0-1000 relative to the image and must stay "
        "within that range. Reply again with coordinates in 0-1000."
    ),
    "no_image": (
        "The image could not be loaded, so the frame was not drawn. Answer the question from the original image."
    ),
}


@dataclass
class ToolResult:
    image: Image.Image
    origin_px: tuple[int, int]
    endpoints_px: dict[str, tuple[int, int]]
    axis_len_px: dict[str, float]
    warnings: list[str] = field(default_factory=list)
    depth_marks: dict[str, str] = field(default_factory=dict)


def validate_arguments(parameters: dict[str, Any]) -> Optional[str]:
    """Return a ``RETRY_MESSAGES`` key if the call is invalid, else ``None``."""
    if not isinstance(parameters, dict) or set(parameters) != REQUIRED_PARAMS:
        return "params"
    ref = parameters["reference_object"]
    if not isinstance(ref, str) or not ref.strip():
        return "empty_reference_object"
    for key in POINT_KEYS:
        value = parameters[key]
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return "coord_shape"
        if not all(isinstance(c, int) and not isinstance(c, bool) for c in value):
            return "coord_not_int"
        if not all(0 <= c <= 1000 for c in value):
            return "coord_out_of_range"
    return None


def prepare_image(image: Image.Image | str | Path) -> Image.Image:
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    return ImageOps.exif_transpose(image).convert("RGB")


def load_image(path: str | Path, max_pixels: int = 1024 * 1024) -> Image.Image:
    """Load an image and downscale it to at most ``max_pixels`` (as in training)."""
    with Image.open(path) as im:
        w, h = im.size
        if max_pixels <= 0 or w * h <= max_pixels:
            return prepare_image(im)
        scale = math.sqrt(max_pixels / (w * h))
        size = (max(1, int(w * scale)), max(1, int(h * scale)))
        buf = io.BytesIO()
        im.convert("RGB").resize(size, Image.LANCZOS).save(buf, format="JPEG", quality=95)
        return prepare_image(Image.open(buf))


def to_pixel(point, width: int, height: int) -> tuple[int, int]:
    """Map a 0-1000 normalized point to pixels (per axis)."""
    x = min(width - 1, max(0, int(round(float(point[0]) * width / 1000.0))))
    y = min(height - 1, max(0, int(round(float(point[1]) * height / 1000.0))))
    return x, y


def _font(size: int):
    for path in _FONT_CANDIDATES:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_arrow(draw, start, end, color, width: int) -> None:
    draw.line((start, end), fill=color, width=width)
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    if length < 1.0:
        return

    ux, uy = dx / length, dy / length
    head_length = min(max(width * 4.0, 12.0), max(4.0, length * 0.45))
    head_half_width = head_length * 0.48
    base_x, base_y = end[0] - ux * head_length, end[1] - uy * head_length
    px, py = -uy, ux
    draw.polygon(
        [
            end,
            (int(round(base_x + px * head_half_width)), int(round(base_y + py * head_half_width))),
            (int(round(base_x - px * head_half_width)), int(round(base_y - py * head_half_width))),
        ],
        fill=color,
    )


def depth_marker_radius(origin_radius: int) -> int:
    return max(origin_radius + 3, int(round(origin_radius * 1.6)))


def _draw_depth_marker(draw, center, color, toward_camera: bool, origin_radius: int, line_width: int) -> None:
    """Circled dot = toward the camera, circled cross = away from it."""
    cx, cy = center
    radius = depth_marker_radius(origin_radius)
    lw = max(3, line_width)
    halo = (0, 0, 0, 230)
    draw.ellipse((cx - radius - 1, cy - radius - 1, cx + radius + 1, cy + radius + 1), outline=halo, width=lw + 2)
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=color, width=lw)
    if toward_camera:
        dot = max(2, lw // 2)
        draw.ellipse((cx - dot, cy - dot, cx + dot, cy + dot), fill=color)
    else:
        arm = (radius - lw) / math.sqrt(2.0)
        for sign in (1.0, -1.0):
            draw.line((cx - arm, cy - sign * arm, cx + arm, cy + sign * arm), fill=color, width=lw)


def execute(image: Image.Image | str | Path, arguments: dict) -> ToolResult:
    """Render the frame given by ``arguments`` (0-1000 ``[x, y]`` points) onto ``image``."""
    base = prepare_image(image)
    width, height = base.size

    points: dict[str, tuple[int, int]] = {}
    for key in POINT_KEYS:
        value = arguments.get(key)
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{key} must be a 2-element [x, y]; got {value!r}")
        points[key] = to_pixel(value, width, height)

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    scale = max(width, height) / 512.0
    line_width = max(3, int(round(4 * scale)))
    font = _font(max(13, int(round(15 * scale))))
    origin = points["origin"]

    warnings: list[str] = []
    axis_len_px: dict[str, float] = {}
    degenerate_threshold = EPS_REL * min(width, height)

    # origin disc first so short axes are drawn on top of it
    origin_radius = max(5, int(round(7 * scale)))
    draw.ellipse(
        (origin[0] - origin_radius, origin[1] - origin_radius, origin[0] + origin_radius, origin[1] + origin_radius),
        fill=ORIGIN_COLOR,
        outline=(0, 0, 0, 255),
        width=max(1, line_width // 2),
    )
    draw.text(
        (origin[0] + origin_radius + 2, origin[1]),
        "O",
        fill=ORIGIN_COLOR,
        font=font,
        stroke_width=max(1, line_width // 2),
        stroke_fill=(0, 0, 0, 230),
        anchor="lm",
    )

    for key in AXIS_KEYS:
        endpoint = points[key]
        axis_len_px[key] = math.hypot(endpoint[0] - origin[0], endpoint[1] - origin[1])
        if axis_len_px[key] < degenerate_threshold:
            warnings.append(f"degenerate:{key}")

    # a single foreshortened axis is drawn as a depth marker instead of an arrow
    depth_marks: dict[str, str] = {}
    marked_axis: str | None = None
    marked_toward = False
    degenerate = [key for key in AXIS_KEYS if axis_len_px[key] < degenerate_threshold]
    if len(degenerate) == 1:
        key = degenerate[0]
        a_key, b_key = HANDEDNESS_PAIRS[key]
        ax, ay = points[a_key][0] - origin[0], points[a_key][1] - origin[1]
        bx, by = points[b_key][0] - origin[0], points[b_key][1] - origin[1]
        sin_ab = (ax * by - ay * bx) / (math.hypot(ax, ay) * math.hypot(bx, by))
        if abs(sin_ab) < COLLINEAR_SIN_MIN:
            depth_marks[key] = "ambiguous"
            warnings.append(f"depth_ambiguous:{key}")
        else:
            marked_axis = key
            marked_toward = sin_ab < 0
            depth_marks[key] = "toward_camera" if marked_toward else "away_from_camera"

    # longest axis first, so shorter axes stay visible
    ring_radius = depth_marker_radius(origin_radius)
    for key in sorted(AXIS_KEYS, key=lambda k: (-axis_len_px[k], AXIS_KEYS.index(k))):
        style = AXIS_STYLES[key]
        endpoint = points[key]
        if key != marked_axis:
            draw_arrow(draw, origin, endpoint, style["color"], line_width)
        offset = ring_radius + 2 if key == marked_axis else line_width + 2
        label_x = min(width - 1, max(0, endpoint[0] + offset))
        label_y = min(height - 1, max(0, endpoint[1] + offset))
        draw.text(
            (label_x, label_y),
            style["label"],
            fill=style["color"],
            font=font,
            stroke_width=max(1, line_width // 2),
            stroke_fill=(0, 0, 0, 230),
            anchor="la",
        )

    if marked_axis is not None:
        _draw_depth_marker(
            draw, points[marked_axis], AXIS_STYLES[marked_axis]["color"], marked_toward, origin_radius, line_width
        )

    rendered = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
    return ToolResult(
        image=rendered,
        origin_px=origin,
        endpoints_px={key: points[key] for key in AXIS_KEYS},
        axis_len_px=axis_len_px,
        warnings=warnings,
        depth_marks=depth_marks,
    )
