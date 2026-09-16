"""Detector interface.

An ABC here rather than a bare function because two implementations are actually
planned -- a plain detector and the sliced wrapper that composes over it -- and
because the design plan keeps D-FINE-X as a one-line fallback if RF-DETR's
newness causes friction. That is the whole justification; the tracker, proposer
and metrics stay flat modules for want of a second implementation.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..geometry import Box
from ..schema import ClipMeta

# --- canonical class vocabulary ---
#
# Class ids CANNOT be hard-coded against a checkpoint. The two detectors in play
# disagree on both indexing and naming:
#
#   Roboflow/rf-detr-large        91 labels, COCO "original" ids: person=1,
#                                 car=3, motorcycle=4, bus=6, truck=8
#   ustc-community/dfine-xlarge   80 labels, contiguous ids: person=0, car=2,
#                                 and Pascal-style names ("motorbike",
#                                 "aeroplane") rather than COCO's
#
# Filtering by a fixed id list against RF-DETR would have selected N/A, airplane
# and train while dropping motorcycle, bus and truck -- silently, as plausible
# detections of the wrong things. So ids are resolved by NAME per model, and
# every Detection leaving a detector carries a CANONICAL id from this table.
# Downstream code depends only on these.
COCO_PERSON = 0
COCO_VEHICLES = (1, 2, 3, 5, 7)   # bicycle, car, motorcycle, bus, truck
COCO_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
              5: "bus", 7: "truck"}

# Accepted spellings per canonical id, lowercased.
CLASS_SYNONYMS: dict[int, tuple[str, ...]] = {
    0: ("person",),
    1: ("bicycle", "bike"),
    2: ("car",),
    3: ("motorcycle", "motorbike"),
    5: ("bus",),
    7: ("truck", "lorry"),
}


def resolve_class_map(id2label: dict[int, str],
                      wanted: tuple[int, ...] = (0, 1, 2, 3, 5, 7)
                      ) -> dict[int, int]:
    """Map a model's own label ids onto canonical ids, by name.

    Raises if `person` cannot be resolved: a detector that cannot report people
    is useless here, and failing loudly at load beats an empty result set that
    looks like a clip with no people in it.
    """
    lower = {int(k): str(v).strip().lower() for k, v in id2label.items()}
    out: dict[int, int] = {}
    for canon in wanted:
        names = CLASS_SYNONYMS[canon]
        for mid, name in lower.items():
            if name in names:
                out[mid] = canon
                break
    if COCO_PERSON not in out.values():
        raise ValueError(
            "could not resolve a 'person' class from the model's id2label "
            f"({len(id2label)} labels). Add the spelling to CLASS_SYNONYMS."
        )
    return out


@dataclass(frozen=True)
class Detection:
    frame: int
    cls: int
    box: Box          # xyxy, pixels, in the clip's NATIVE resolution
    conf: float

    @property
    def is_person(self) -> bool:
        return self.cls == COCO_PERSON

    @property
    def name(self) -> str:
        return COCO_NAMES.get(self.cls, str(self.cls))


class Detector(abc.ABC):
    """Batched detection over frames of one clip.

    Batched rather than per-frame so the sliced wrapper can re-tile without
    needing a separate code path, and so GPU utilisation is not one frame at a
    time on the 4K clip.

    Implementations must return boxes in the clip's native pixel coordinates --
    any internal resize is the implementation's business and must be undone
    before returning. Every downstream spatial quantity is normalized by the
    vehicle diagonal, which is only meaningful if all boxes share one frame of
    reference.
    """

    @abc.abstractmethod
    def detect(self, frames: Sequence[np.ndarray], meta: ClipMeta,
               frame_indices: Sequence[int], conf: float) -> list[list[Detection]]:
        """Return one list of Detections per input frame, aligned by position.

        `frame_indices` gives the clip-level index of each frame so returned
        Detections carry the right `frame` value even when the caller passes a
        non-contiguous subset.
        """

    @property
    @abc.abstractmethod
    def revision(self) -> str:
        """Resolved model revision hash, recorded in the output for determinism."""


# Cap on detections kept per frame per class, after sorting by confidence.
# A surveillance frame holds tens of relevant objects, not hundreds; the tail
# below this is noise. The cap exists because lowering the tracker's detection
# floor to 0.10 multiplied per-frame counts, and on the 4K clip that runs 28
# tiles per frame, which exhausted system RAM.
MAX_DETS_PER_FRAME_PER_CLASS = 100


def cap_per_frame(dets: Sequence[Detection],
                  limit: int = MAX_DETS_PER_FRAME_PER_CLASS) -> list[Detection]:
    """Keep the `limit` highest-confidence detections of each class.

    Per class, not overall, so a frame full of parked cars cannot crowd out the
    people -- which is exactly the situation on the aerial clip, where vehicles
    outnumber persons five to one.
    """
    by_cls: dict[int, list[Detection]] = {}
    for d in dets:
        by_cls.setdefault(d.cls, []).append(d)
    out: list[Detection] = []
    for cls in sorted(by_cls):
        pool = sorted(by_cls[cls], key=lambda d: (-d.conf, d.box))
        out.extend(pool[:limit])
    return sorted(out, key=lambda d: (d.frame, d.cls, -d.conf, d.box))


def ios(a: Box, b: Box) -> float:
    """Intersection over the SMALLER box's area.

    Distinct from IoU and necessary for tiled inference: an object crossing a
    tile boundary is detected once as a truncated fragment inside the tile and
    once whole by the full-frame or neighbouring-tile pass. A fragment that is
    half the full box has IoU 0.5 with it and survives an IoU-0.55 NMS, leaving
    two boxes on one object. Its IoS against the full box is ~1.0, because the
    fragment is almost entirely contained.

    Measured on `gt1125_06`, IoU-only merging roughly doubled the vehicle count
    (22 -> 46 on one frame) almost entirely through such duplicates.
    """
    from ..geometry import area

    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    smaller = min(area(a), area(b))
    if smaller <= 0.0:
        return 0.0
    return inter / smaller


def nms(dets: Sequence[Detection], iou_thresh: float = 0.5,
        ios_thresh: float | None = 0.75) -> list[Detection]:
    """Class-wise greedy NMS, suppressing on IoU **or** containment.

    Kept here rather than in `sliced.py` so it can be tested without a GPU or a
    model. `ios_thresh=None` disables containment suppression and leaves plain
    IoU NMS.

    Suppression is class-wise, so a person standing against a car is never
    suppressed by the car's box however contained it is. Within a class the
    containment rule can in principle delete a genuinely nested object (a small
    car parked inside a truck's bounding box); at `ios_thresh=0.75` that is rare
    and speculative, whereas the tile-fragment duplicates it removes are
    measured and certain.

    Ordering: confidence first, then **larger area first**, then box
    coordinates. The area tie-break matters for exactly the case this exists
    for -- when a tile fragment and the whole object score equally, the whole
    object must be the survivor, not the fragment. The coordinate tie-break
    makes the result independent of input ordering, since determinism is graded.
    """
    from ..geometry import area, iou as box_iou

    keep: list[Detection] = []
    by_cls: dict[int, list[Detection]] = {}
    for d in dets:
        by_cls.setdefault(d.cls, []).append(d)

    for cls in sorted(by_cls):
        pool = sorted(by_cls[cls], key=lambda d: (-d.conf, -area(d.box), d.box))
        while pool:
            best = pool.pop(0)
            keep.append(best)
            pool = [d for d in pool
                    if box_iou(best.box, d.box) < iou_thresh
                    and (ios_thresh is None or ios(best.box, d.box) < ios_thresh)]
    return sorted(keep, key=lambda d: (d.frame, d.cls, -d.conf, d.box))
