"""Tiled ("sliced") inference: a decorator over any Detector.

Rationale and calibration (design-plan s6.2): SAHI reports **+5-7 AP
inference-only** on aerial data such as VisDrone. The larger published gains
require slicing-aided *fine-tuning*, which needs training data this project does
not have. So this is a useful gain, not a rescue, and whether it survives at all
is decided by the `gt1125_06` probe rather than by argument.

The mechanism is simple: a 10 px person in a 3840x2160 frame is ~0.3% of the
frame height, and a detector that internally resizes to 704 px sees it at ~3 px.
Running the same detector on 704 px tiles shows it that person at native scale.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..schema import ClipMeta
from .base import Detection, Detector, nms


def tile_origins(size: int, tile: int, overlap: float) -> list[int]:
    """Left (or top) edges of a 1-D tiling covering [0, size).

    The last tile is pulled back flush with the edge rather than allowed to run
    past it, so every pixel is covered exactly once or more and no tile contains
    out-of-frame padding. Padding would put a fake high-contrast border inside
    the receptive field, which detectors happily hallucinate objects against.
    """
    if tile >= size:
        return [0]
    step = max(1, int(round(tile * (1.0 - overlap))))
    origins = list(range(0, size - tile + 1, step))
    if origins[-1] != size - tile:
        origins.append(size - tile)
    return origins


@dataclass
class SlicedDetector(Detector):
    """Wraps another Detector, running it over overlapping tiles.

    Also runs the inner detector on the **full frame** and merges, because
    tiling alone loses objects larger than a tile -- and on `gt1125_06` the box
    truck is exactly that. Dropping the full-frame pass to save compute would
    trade the recall we are trying to buy for recall we already had.
    """
    inner: Detector
    tile: int = 704            # RF-DETR-L's native resolution
    overlap: float = 0.25
    merge_iou: float = 0.55
    merge_ios: float | None = 0.75
    include_full_frame: bool = True

    @property
    def revision(self) -> str:
        return self.inner.revision

    def detect(self, frames: Sequence[np.ndarray], meta: ClipMeta,
               frame_indices: Sequence[int], conf: float) -> list[list[Detection]]:
        xs = tile_origins(meta.width, self.tile, self.overlap)
        ys = tile_origins(meta.height, self.tile, self.overlap)

        out: list[list[Detection]] = []
        for frame, fidx in zip(frames, frame_indices):
            dets: list[Detection] = []

            for y0 in ys:
                for x0 in xs:
                    y1 = min(y0 + self.tile, meta.height)
                    x1 = min(x0 + self.tile, meta.width)
                    crop = frame[y0:y1, x0:x1]
                    tile_meta = ClipMeta(clip_id=meta.clip_id, width=x1 - x0,
                                         height=y1 - y0, fps=meta.fps,
                                         n_frames=meta.n_frames)
                    for d in self.inner.detect([crop], tile_meta, [fidx], conf)[0]:
                        dets.append(_shift(d, x0, y0))

            if self.include_full_frame:
                dets.extend(self.inner.detect([frame], meta, [fidx], conf)[0])

            out.append(nms(dets, self.merge_iou, self.merge_ios))
        return out


def _shift(d: Detection, dx: int, dy: int) -> Detection:
    x1, y1, x2, y2 = d.box
    return Detection(frame=d.frame, cls=d.cls, conf=d.conf,
                     box=(x1 + dx, y1 + dy, x2 + dx, y2 + dy))
