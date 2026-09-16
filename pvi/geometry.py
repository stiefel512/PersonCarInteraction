"""Box geometry in normalized units.

Every spatial quantity the rule layer sees passes through here. The invariant
from design-plan.md s4 is that no rule ever compares raw pixels: the clip set
spans 352x288 to 3840x2160, so a pixel threshold tuned on one clip is
meaningless on another.

Boxes are xyxy in pixels, x2 >= x1 and y2 >= y1.
"""
from __future__ import annotations

import math

Box = tuple[float, float, float, float]


def width(b: Box) -> float:
    return max(0.0, b[2] - b[0])


def height(b: Box) -> float:
    return max(0.0, b[3] - b[1])


def diag(b: Box) -> float:
    return math.hypot(width(b), height(b))


def area(b: Box) -> float:
    return width(b) * height(b)


def gap_px(a: Box, b: Box) -> float:
    """Minimum separation between two axis-aligned boxes, 0 when they overlap.

    Axis gaps are independent, so the true corner-to-corner distance is their
    hypotenuse -- not the sum, and not the larger of the two. When the boxes
    overlap on an axis that axis contributes 0, which makes an edge-on-edge
    approach reduce to the single-axis distance, as it should.
    """
    dx = max(b[0] - a[2], a[0] - b[2], 0.0)
    dy = max(b[1] - a[3], a[1] - b[3], 0.0)
    return math.hypot(dx, dy)


def d_norm(person: Box, vehicle: Box) -> float:
    """Person-vehicle separation in units of the VEHICLE box diagonal.

    Normalizing by the vehicle rather than the person is deliberate: the vehicle
    is the stable reference (a parked car's apparent size is constant), whereas
    a person's box shrinks as they walk away from the camera, which would make
    the same physical gap read as a growing distance.

    Returns 0.0 for overlapping boxes. A degenerate vehicle box (zero diagonal)
    returns inf rather than raising -- a detector can emit one, and the caller
    should treat it as "no usable measurement", not crash.
    """
    dv = diag(vehicle)
    if dv <= 0.0:
        return math.inf
    return gap_px(person, vehicle) / dv


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = area(a) + area(b) - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def speed_bl_s(box_t0: Box, box_t1: Box, dt_s: float) -> float:
    """Centroid speed in body-lengths per second.

    Body-lengths rather than pixels for the same reason as d_norm: it is the
    only way a 10 px aerial person and a 400 px CCTV person can share a
    threshold. Uses the mean of the two person heights so the measure does not
    jump when the box resizes between frames.
    """
    if dt_s <= 0.0:
        return 0.0
    h = 0.5 * (height(box_t0) + height(box_t1))
    if h <= 0.0:
        return math.inf
    cx0, cy0 = 0.5 * (box_t0[0] + box_t0[2]), 0.5 * (box_t0[1] + box_t0[3])
    cx1, cy1 = 0.5 * (box_t1[0] + box_t1[2]), 0.5 * (box_t1[1] + box_t1[3])
    return math.hypot(cx1 - cx0, cy1 - cy0) / h / dt_s


def union_box(a: Box, b: Box) -> Box:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def dilate(b: Box, margin: float, bounds: tuple[int, int] | None = None) -> Box:
    """Expand a box by `margin` as a FRACTION of its own size, clipped to bounds.

    Fractional rather than absolute so `crop_margin` means the same thing on a
    352x288 clip and a 4K one.
    """
    dw, dh = width(b) * margin, height(b) * margin
    out = (b[0] - dw, b[1] - dh, b[2] + dw, b[3] + dh)
    if bounds is None:
        return out
    w, h = bounds
    return (max(0.0, out[0]), max(0.0, out[1]), min(float(w), out[2]), min(float(h), out[3]))
