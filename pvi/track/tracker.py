"""Multi-object tracking via `roboflow/trackers` (Apache-2.0).

**Supersedes design-plan s6.6.** That section says BoT-SORT's camera motion
compensation is an OpenCV C++ VideoStab module with no Python implementation
upstream, so `track/gmc.py` had to be hand-rolled. That was true of the original
BoT-SORT repo; it is **not** true of `roboflow/trackers` 2.6.0, which ships
`BoTSORTTracker(enable_cmc=True, cmc_method="orb"|"sift"|"sparseOptFlow"|"ecc")`
with a pure-Python CMC. So GMC on the tracking path is a library call after all,
and hand-rolled numerical code is off the critical path -- which the design plan
listed as one of its top risks.

`pvi/track/gmc.py` survives in a narrower role: deciding whether the *clip* has a
static camera, which gates the `door_delta` feature. The tracker uses CMC
internally but does not report how much the camera moved, and that number is what
the gate needs.

Two design choices worth stating:

**Separate tracker instances for persons and vehicles.** These trackers are
class-agnostic, so a single instance can hand a person's id to a car when their
boxes overlap -- which is exactly what happens when someone stands against a
vehicle, i.e. during every event we care about. Splitting them makes that
impossible.

**All vehicle classes share one tracker instance.** A car is detected as `car` in
one frame and `truck` in the next often enough that per-class tracking would
fragment a single vehicle into several tracks, and R2/R3 depend on track
continuity. Class is resolved per track afterwards by majority vote.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from ..detect.base import COCO_PERSON, Detection
from ..propose.features import Track
from ..schema import ClipMeta

# Vehicle track ids are offset so they cannot collide with person ids from the
# other tracker instance. Ids appear in the deliverable, so they must be unique
# within a clip.
VEHICLE_ID_OFFSET = 10_000


class MultiClassTracker:
    """Streaming tracker: fed frame by frame, yields Tracks at the end.

    Streaming rather than batch because the 4K clip cannot be held in memory --
    600 frames of 3840x2160 RGB is ~15 GB -- and CMC needs the actual frame.
    """

    def __init__(self, meta: ClipMeta, enable_cmc: bool = True,
                 cmc_method: str = "orb", min_consecutive: int = 2):
        from trackers import BoTSORTTracker

        def make():
            return BoTSORTTracker(
                frame_rate=meta.fps,
                enable_cmc=enable_cmc,
                cmc_method=cmc_method,
                minimum_consecutive_frames=min_consecutive,
                # lost_track_buffer in frames; 1 s of tolerance for an occlusion
                # behind a vehicle, expressed in seconds because the set spans
                # 6-30 fps and a fixed frame count would mean 5x different
                # things across clips.
                lost_track_buffer=max(5, int(round(meta.fps * 1.0))),
            )

        self.meta = meta
        self.person_tracker = make()
        self.vehicle_tracker = make()
        self._obs: dict[int, dict] = {}

    def update(self, frame_idx: int, frame_rgb: np.ndarray,
               dets: list[Detection]) -> None:
        import supervision as sv

        # The library expects BGR (it is OpenCV-facing); pvi.video yields RGB.
        # Getting this wrong degrades ORB feature matching silently rather than
        # raising, so the conversion is explicit here and nowhere else.
        # ascontiguousarray because a negative-stride view upsets OpenCV.
        frame_bgr = np.ascontiguousarray(frame_rgb[:, :, ::-1])

        persons = [d for d in dets if d.cls == COCO_PERSON]
        vehicles = [d for d in dets if d.cls != COCO_PERSON]

        self._run(self.person_tracker, persons, frame_bgr, 0, frame_idx, sv)
        self._run(self.vehicle_tracker, vehicles, frame_bgr, VEHICLE_ID_OFFSET,
                  frame_idx, sv)

    def _run(self, tracker, dets: list[Detection], frame_bgr, offset: int,
             frame_idx: int, sv) -> None:
        if dets:
            sv_dets = sv.Detections(
                xyxy=np.array([d.box for d in dets], dtype=np.float32),
                confidence=np.array([d.conf for d in dets], dtype=np.float32),
                class_id=np.array([d.cls for d in dets], dtype=int),
            )
        else:
            # Still update on empty frames: the tracker needs to age its lost
            # tracks and run CMC, or a gap in detections shifts every track.
            sv_dets = sv.Detections.empty()

        tracked = tracker.update(sv_dets, frame=frame_bgr)
        if tracked.tracker_id is None:
            return

        for i, tid in enumerate(tracked.tracker_id):
            if tid < 0:               # unconfirmed track -- not yet an identity
                continue
            box = tuple(float(v) for v in tracked.xyxy[i])
            cls = int(tracked.class_id[i]) if tracked.class_id is not None else COCO_PERSON
            rec = self._obs.setdefault(int(tid) + offset,
                                       {"frames": [], "boxes": [], "classes": []})
            rec["frames"].append(frame_idx)
            rec["boxes"].append(box)
            rec["classes"].append(cls)

    def finish(self) -> list[Track]:
        out: list[Track] = []
        for tid, rec in sorted(self._obs.items()):
            if not rec["frames"]:
                continue
            # Majority class over the track's life: a vehicle flipping between
            # `car` and `truck` across frames should still report one class.
            cls = Counter(rec["classes"]).most_common(1)[0][0]
            order = sorted(range(len(rec["frames"])), key=lambda i: rec["frames"][i])
            out.append(Track(
                id=tid, cls=cls,
                frames=[rec["frames"][i] for i in order],
                boxes=[rec["boxes"][i] for i in order],
            ))
        return out


def track_objects(detections: list[Detection], meta: ClipMeta,
                  frames: dict[int, np.ndarray], enable_cmc: bool = True,
                  cmc_method: str = "orb") -> list[Track]:
    """Batch convenience wrapper, for callers that already hold every frame.

    The streaming `MultiClassTracker` is what the CLI uses; this exists for
    tests and small clips.
    """
    tr = MultiClassTracker(meta, enable_cmc=enable_cmc, cmc_method=cmc_method)
    by_frame: dict[int, list[Detection]] = {}
    for d in detections:
        by_frame.setdefault(d.frame, []).append(d)
    for f in range(meta.n_frames):
        if f in frames:
            tr.update(f, frames[f], by_frame.get(f, []))
    return tr.finish()
