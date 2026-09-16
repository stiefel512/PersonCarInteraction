"""Per-(person, vehicle) feature time series.

Everything here is computed on a shared frame axis spanning the clip, with NaN
wherever a track is absent. NaN rather than a sentinel number because the
downstream hysteresis treats NaN as "no measurement" and holds state across it,
which is what occlusion should do -- whereas a sentinel like 999 would read as
"very far away" and split one interaction into two.
"""
from __future__ import annotations

from dataclasses import dataclass

import math
import numpy as np

from ..config import smooth_frames
from ..geometry import Box, d_norm, iou, speed_bl_s
from ..schema import ClipMeta
from .spans import moving_average


@dataclass
class Track:
    """A tracked object over a contiguous-ish span of frames.

    `boxes` is indexed by position in `frames`, not by frame number -- the two
    are kept side by side rather than as a dict so the common case (iterating in
    order) stays cheap, and `box_at` handles lookup.
    """
    id: int
    cls: int
    frames: list[int]
    boxes: list[Box]
    interpolated: list[bool] | None = None

    @property
    def birth_frame(self) -> int:
        return self.frames[0]

    @property
    def death_frame(self) -> int:
        return self.frames[-1]

    @property
    def is_person(self) -> bool:
        return self.cls == 0

    def box_at(self, frame: int) -> Box | None:
        try:
            return self.boxes[self.frames.index(frame)]
        except ValueError:
            return None

    def centroid_motion(self) -> float:
        """Std-dev of the box centroid in units of the mean box diagonal.

        Used to decide `is_static`. Normalized so the threshold means the same
        thing for a 4K truck and a CIF hatchback.
        """
        if len(self.boxes) < 2:
            return 0.0
        cx = np.array([0.5 * (b[0] + b[2]) for b in self.boxes])
        cy = np.array([0.5 * (b[1] + b[3]) for b in self.boxes])
        diags = np.array([math.hypot(b[2] - b[0], b[3] - b[1]) for b in self.boxes])
        scale = float(diags.mean())
        if scale <= 0:
            return 0.0
        return float(math.hypot(cx.std(), cy.std()) / scale)

    def is_static(self, thresh: float = 0.05) -> bool:
        return self.centroid_motion() < thresh


@dataclass
class PairFeatures:
    """Aligned per-frame series for one (person, vehicle) pair.

    All arrays have length `meta.n_frames`. `visible` marks frames where BOTH
    tracks have a box; every other series is NaN there.
    """
    person_id: int
    vehicle_id: int
    vehicle_cls: int
    d_norm: np.ndarray
    iou: np.ndarray
    speed_bl_s: np.ndarray
    visible: np.ndarray        # bool
    door_delta: np.ndarray
    vehicle_is_static: bool

    @property
    def n_visible(self) -> int:
        return int(self.visible.sum())

    @property
    def min_d_norm(self) -> float:
        vals = self.d_norm[~np.isnan(self.d_norm)]
        return float(vals.min()) if vals.size else math.inf


def pair_features(person: Track, vehicle: Track, meta: ClipMeta,
                  smooth_window_s: float = 0.4,
                  door_delta: np.ndarray | None = None) -> PairFeatures:
    n = meta.n_frames
    dn = np.full(n, np.nan)
    iu = np.full(n, np.nan)
    sp = np.full(n, np.nan)
    vis = np.zeros(n, dtype=bool)

    pbox = {f: b for f, b in zip(person.frames, person.boxes)}
    vbox = {f: b for f, b in zip(vehicle.frames, vehicle.boxes)}

    dt = 1.0 / meta.fps
    prev_f, prev_b = None, None
    for f in range(n):
        pb, vb = pbox.get(f), vbox.get(f)
        if pb is None or vb is None:
            continue
        vis[f] = True
        dn[f] = d_norm(pb, vb)
        iu[f] = iou(pb, vb)
        if prev_b is not None and prev_f is not None:
            sp[f] = speed_bl_s(prev_b, pb, (f - prev_f) * dt)
        prev_f, prev_b = f, pb

    # Smoothing is applied to d_norm only. Speed is already a difference and
    # smoothing it twice over-damps the short events; iou is used for reporting,
    # not for triggering.
    w = smooth_frames(smooth_window_s, meta.fps)
    dn = np.array(moving_average(list(dn), w))

    dd = door_delta if door_delta is not None else np.full(n, np.nan)
    return PairFeatures(
        person_id=person.id, vehicle_id=vehicle.id, vehicle_cls=vehicle.cls,
        d_norm=dn, iou=iu, speed_bl_s=sp, visible=vis, door_delta=dd,
        vehicle_is_static=vehicle.is_static(),
    )


def door_delta_series(frames_gray: dict[int, np.ndarray], vehicle: Track,
                      n_frames: int, static_camera: bool,
                      scale: float = 1.0) -> np.ndarray:
    """Mean absolute deviation of the vehicle-box region from its temporal median.

    Valid ONLY when the camera is static and the vehicle is parked. Otherwise
    every pixel in the box changes for reasons that have nothing to do with a
    door, and the feature is worse than useless -- so it returns all-NaN and the
    rule that consumes it is skipped. This conditional is exactly what the
    open-vocabulary door cue (design-plan s6.7) would delete, which is why that
    probe is worth running.

    Normalized by 255 so the threshold is in intensity fractions, not levels.

    `scale` maps native-resolution boxes into the coordinate frame of
    `frames_gray`, which the caller may have downscaled to bound memory (600
    frames of 4K grayscale is ~5 GB). The measure is a coarse appearance
    statistic, so the downscale costs nothing that matters.
    """
    out = np.full(n_frames, np.nan)
    if not static_camera or not vehicle.is_static():
        return out

    # Fixed reference region: the median box over the track. Using the per-frame
    # box would move the window with the detector's jitter and register that
    # jitter as appearance change.
    boxes = np.array(vehicle.boxes, dtype=float) * scale
    x1, y1, x2, y2 = (int(round(v)) for v in np.median(boxes, axis=0))
    if x2 <= x1 or y2 <= y1:
        return out

    crops = {f: g[y1:y2, x1:x2].astype(np.float32)
             for f, g in frames_gray.items()
             if f in set(vehicle.frames) and g[y1:y2, x1:x2].size}
    if len(crops) < 3:
        return out

    stack = np.stack(list(crops.values()))
    median = np.median(stack, axis=0)
    for f, c in crops.items():
        if 0 <= f < n_frames:
            out[f] = float(np.abs(c - median).mean()) / 255.0
    return out
