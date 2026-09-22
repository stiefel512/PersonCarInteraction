"""Entry point: one clip in, one JSON out.

Deliberately has no ground-truth dependency. Evaluation is a separate entry
point (`pvi.evaluate.run`), so the thing that produces the deliverable cannot
see the labels it will be scored against.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import config as C
from . import video
from .detect.base import COCO_PERSON, Detection, cap_per_frame
from .detect.hf import HFDetector
from .detect.sliced import SlicedDetector
from .judge.base import Verdict
from .judge.geometric import GeometricJudge
from .propose.features import Track, door_cue_series, pair_features
from .propose.rules import RULE_TYPE, Candidate, primary_rule, propose
from .schema import ClipMeta, Interaction, PersonRef, VehicleRef, write_output
from .track.tracker import MultiClassTracker


def seed_everything(seed: int) -> None:
    """Seed every RNG and ask torch for deterministic kernels.

    `warn_only=True` rather than hard-failing: a few ops have no deterministic
    implementation, and aborting the run would be worse than a warning that says
    exactly which one. The VLM response cache is the real reproducibility lever
    (design-plan s7); this is defence in depth.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


# Tiling is pointless below this width: at 704 px tiles a 1280-wide frame is
# already close to native scale through the detector's own resize, and for
# anything at or under the tile size `tile_origins` returns a single tile, so
# the tiled path becomes the plain path run twice. Only `gt1125_06` (3840) is
# above this in the current set.
TILE_MIN_WIDTH = 1920


def should_tile(meta: ClipMeta, requested: bool | None) -> bool:
    """Whether to tile this clip. `requested` overrides the resolution rule."""
    if requested is not None:
        return requested
    return meta.width >= TILE_MIN_WIDTH


# Loaded models, keyed by what distinguishes them. Process-lifetime, because
# `tools/run_all.py` runs 8 clips x 2 judges in ONE process and `run_clip` used
# to construct every model afresh each call -- 16 loads of a 16 GB VLM among
# them, which exhausted system RAM partway through the set. Caching also removes
# most of the wall-clock: loading Qwen costs far more than judging a clip.
_MODELS: dict[tuple, object] = {}


def clear_model_cache() -> None:
    """Drop cached models. For tests and for callers switching checkpoints."""
    _MODELS.clear()
    _free_gpu()


def build_detector(cfg: C.Config, tiled: bool):
    key = ("detector", cfg.detector.model_id, cfg.detector.revision, cfg.device)
    if key not in _MODELS:
        _MODELS[key] = HFDetector(
            cfg.detector.model_id, revision=cfg.detector.revision,
            device=cfg.device, classes=cfg.detector.classes)
    det = _MODELS[key]
    # The sliced wrapper is a thin decorator over the shared detector, so it is
    # rebuilt per clip while the weights are loaded once.
    return SlicedDetector(inner=det) if tiled else det


# Stored grayscale frames are downscaled to this max width. 600 frames of 4K
# grayscale is ~5 GB, and the one thing these frames feed -- the camera-motion
# decision -- loses nothing at 960 px.
GRAY_MAX_WIDTH = 960

# Memory budget for frames held while judging. `read_frames` materialises every
# frame it is given, and a 4K frame is 25 MB, so an unbounded request is a
# multi-gigabyte allocation. Candidates are judged in chunks that fit this.
MAX_JUDGE_FRAME_BYTES = 1_500_000_000


@dataclass
class ClipTracks:
    """Output of the expensive half of the pipeline.

    Decoding, detection and tracking depend only on `det_conf` (and the fixed
    detector settings), not on the proposal or judge thresholds. Holding this
    lets a sweep vary `tau_near`/`tau_far` without redoing the part that costs
    minutes per clip.
    """
    meta: ClipMeta
    tracks: list
    gray: dict
    gray_scale: float
    static_camera: bool
    tiled: bool
    detector_revision: str
    n_detections: int
    n_detections_above_det_conf: int


def track_clip(clip_path: Path, cfg: C.Config, tiled: bool | None = None,
               batch: int = 8) -> ClipTracks:
    """Decode, detect and track. The reusable half; see ClipTracks."""
    meta = video.probe(clip_path)
    t = cfg.tunable
    tiled = should_tile(meta, tiled)

    # Pass 1: decide whether the camera moves, before tracking, because the
    # answer decides whether the tracker runs CMC at all. Unconditional CMC is
    # NOT free -- when feature matching degenerates it corrupts association
    # rather than falling back to identity.
    gray, gray_scale = _decode_gray(clip_path, meta)
    static_camera = _camera_is_static(gray, meta)

    detector = build_detector(cfg, tiled)

    # Pass 2: detect and track in one streaming pass. Streaming because the 4K
    # clip cannot be held in memory and the tracker's CMC needs the real frame.
    # Detect down to the FLOOR, not to det_conf. ByteTrack's second association
    # pass needs the low-confidence detections; pre-filtering at det_conf
    # deletes them and splits one person into several sequential tracks on any
    # clip where the detector flickers. det_conf is passed to the tracker as its
    # high-confidence threshold instead.
    tracker = MultiClassTracker(meta, enable_cmc=not static_camera,
                                cmc_method=cfg.tracker.gmc,
                                det_conf=t.det_conf)
    # Counted, not accumulated. Holding every detection for the whole clip is
    # pure memory cost for a debug statistic, and at the 0.10 floor on the 4K
    # tiled clip that list is enormous.
    n_dets = 0
    n_dets_above_conf = 0
    buf, idxs = [], []

    def flush():
        nonlocal n_dets, n_dets_above_conf
        if not buf:
            return
        for per_frame, fi, fr in zip(
                detector.detect(buf, meta, idxs, cfg.tracker.det_floor),
                idxs, buf):
            per_frame = cap_per_frame(per_frame)
            n_dets += len(per_frame)
            n_dets_above_conf += sum(1 for d in per_frame if d.conf >= t.det_conf)
            tracker.update(fi, fr, per_frame)
        buf.clear()
        idxs.clear()

    for i, frame in video.iter_frames(clip_path, meta):
        buf.append(frame.copy())
        idxs.append(i)
        if len(buf) == batch:
            flush()
    flush()

    return ClipTracks(
        meta=meta, tracks=tracker.finish(), gray=gray, gray_scale=gray_scale,
        static_camera=static_camera, tiled=tiled,
        detector_revision=detector.revision,
        n_detections=n_dets, n_detections_above_det_conf=n_dets_above_conf,
    )


def judge_tracks(clip_path: Path, ct: ClipTracks, cfg: C.Config,
                 judge_name: str = "vlm", skip_door_cue: bool = False,
                 describe: bool = True
                 ) -> tuple[ClipMeta, list[Interaction], dict]:
    """Propose candidates from tracks and adjudicate them. The cheap half.

    `describe` runs the structured description pass (judge/describe.py) on
    accepted candidates. LOCO turns it off: it scores detection only, and a
    description call per accepted candidate per setting is pure cost there.
    """
    # Fail fast on a missing or truncated VLM checkpoint, before the door cue
    # spends a Grounding DINO pass. (This check used to sit in the tracking
    # half, where it referenced a judge name that no longer exists there -- a
    # NameError the geometric smoke run caught immediately.)
    if judge_name == "vlm":
        from .judge.vlm import check_weights_available
        check_weights_available(cfg.vlm.model_id, cfg.vlm.revision)

    t = cfg.tunable
    meta, tracks = ct.meta, ct.tracks
    gray, gray_scale, static_camera = ct.gray, ct.gray_scale, ct.static_camera
    persons = [tr for tr in tracks if tr.is_person]
    vehicles = [tr for tr in tracks if not tr.is_person]

    # --- propose, in two passes ---
    #
    # The door cue is an open-vocabulary detector pass, so it is far too
    # expensive to run on every frame. It is only ever consumed by R4, which
    # requires a person to already be near the vehicle -- so pass 1 computes the
    # near-spans without it, and the cue runs only on frames inside those spans.
    def build_pairs(door_by_vehicle):
        pairs, feats = [], {}
        for p in persons:
            for v in vehicles:
                pf = pair_features(p, v, meta, t.smooth_window_s,
                                   door_by_vehicle.get(v.id))
                if pf.n_visible == 0:
                    continue
                feats[(p.id, v.id)] = pf
                pairs.append((pf, p))
        return pairs, feats

    pairs, feats = build_pairs({})
    door_frames = _near_frames(pairs, t.tau_near, t.tau_far, meta)

    door_dets = {}
    if door_frames and not skip_door_cue:
        door_dets = _detect_doors(clip_path, meta, sorted(door_frames), cfg)
        door = {v.id: door_cue_series(door_dets, v, meta.n_frames)
                for v in vehicles}
        pairs, feats = build_pairs(door)

    cands = propose(pairs, meta, t.tau_near, t.tau_far, t.min_dwell_s,
                    t.door_conf_thresh)

    # --- judge ---
    detector_revision = ct.detector_revision
    _free_gpu()

    accepted, rejected = _judge_all(cands, meta, clip_path, cfg, judge_name,
                                    feats, persons, vehicles, describe)

    track_by_id = {tr.id: tr for tr in tracks}

    interactions = []
    for n, (cand, verdict) in enumerate(accepted, start=1):
        interactions.append(Interaction(
            interaction_id=f"{meta.clip_id}__i{n:03d}",
            event_group_id=f"{meta.clip_id}__g{n:03d}",
            type=verdict.type,
            frame_start=cand.frame_start, frame_end=cand.frame_end,
            time_start_s=round(meta.to_s(cand.frame_start), 3),
            time_end_s=round(meta.to_s(cand.frame_end), 3),
            person=PersonRef(track_id=cand.person_id,
                             description=verdict.person_desc,
                             attributes=verdict.person_attrs),
            vehicle=VehicleRef(track_id=cand.vehicle_id,
                               cls=_cls_name(cand.vehicle_cls),
                               description=verdict.vehicle_desc,
                               attributes=verdict.vehicle_attrs),
            note=verdict.note, confidence=round(verdict.confidence, 4),
            evidence={
                "rule": primary_rule(cand), **cand.evidence,
                # Normalized union boxes over the span. These let the evaluator
                # check that a prediction refers to the same actors as the GT
                # event it overlaps -- temporal overlap alone is not a match in
                # the multi-actor clips (evaluate/match.anchors_agree).
                "person_box_norm": _span_box_norm(
                    track_by_id.get(cand.person_id), cand.span, meta),
                "vehicle_box_norm": _span_box_norm(
                    track_by_id.get(cand.vehicle_id), cand.span, meta),
            },
        ))

    debug = {
        "clip_id": meta.clip_id,
        "judge": judge_name,
        "detector_revision": detector_revision,
        "static_camera": static_camera,
        "tiled": ct.tiled,
        "cmc_enabled": not static_camera,
        "n_detections": ct.n_detections,
        "n_detections_above_det_conf": ct.n_detections_above_det_conf,
        "det_floor": cfg.tracker.det_floor,
        "det_conf": t.det_conf,
        "n_person_tracks": len(persons),
        "n_vehicle_tracks": len(vehicles),
        "n_door_cue_frames": len(door_dets),
        "n_door_detections": sum(len(v) for v in door_dets.values()),
        "n_candidates": len(cands),
        "n_accepted": len(accepted),
        # Per-track summary. Detector failure and tracker fragmentation are both
        # SILENT here -- they show up as missing or duplicated candidates, never
        # as an error -- so the raw track inventory is a first-class diagnostic
        # (design-plan s7), not debug noise. `gap_frames` is the giveaway for
        # fragmentation: a track with many internal gaps is one identity the
        # tracker kept losing.
        "tracks": [
            {"id": tr.id, "cls": _cls_name(tr.cls),
             "birth": tr.birth_frame, "death": tr.death_frame,
             "n_observed": len(tr.frames),
             "span_frames": tr.death_frame - tr.birth_frame + 1,
             "gap_frames": (tr.death_frame - tr.birth_frame + 1) - len(tr.frames),
             "centroid_motion": round(tr.centroid_motion(), 4),
             "is_static": tr.is_static()}
            for tr in tracks
        ],
        "rejected": [
            {"person_id": c.person_id, "vehicle_id": c.vehicle_id,
             "frame_start": c.frame_start, "frame_end": c.frame_end,
             "rules": c.rules, "evidence": c.evidence,
             "verdict_type": (v.type if v else None),
             "verdict_confidence": (round(v.confidence, 4) if v else None),
             "reason": ("could not judge" if v is None else "rejected")}
            for c, v in rejected
        ],
    }
    return meta, interactions, debug


def _near_frames(pairs, tau_near: float, tau_far: float, meta: ClipMeta) -> set[int]:
    """Frames inside some person-vehicle near-span, padded by one second.

    These are the only frames R4 can possibly care about, so they are the only
    ones the open-vocabulary door detector is run on. Padding by a second
    catches a door opened just before contact or closed just after.
    """
    from .propose.rules import near_spans

    pad = max(1, meta.to_frames(1.0))
    out: set[int] = set()
    for pf, _person in pairs:
        for lo, hi in near_spans(pf, tau_near, tau_far):
            out.update(range(max(0, lo - pad), min(meta.n_frames, hi + pad + 1)))
    return out


def _detect_doors(clip_path: Path, meta: ClipMeta, frames: list[int],
                  cfg: C.Config) -> dict[int, list]:
    """Run the open-vocabulary detector for open doors on selected frames.

    Returns {frame: [(box, conf), ...]}. See
    experiments/2026-09-16_door-cue-ground-level/findings.md for why this
    replaced the pixel-change heuristic, and for the one clip it does not work
    on (gt1125_06, aerial).
    """
    from .detect.openvocab import OpenVocabDetector

    key = ("openvocab", cfg.openvocab.model_id, cfg.openvocab.revision, cfg.device)
    if key not in _MODELS:
        _MODELS[key] = OpenVocabDetector(
            cfg.openvocab.model_id, revision=cfg.openvocab.revision,
            device=cfg.device)
    ov = _MODELS[key]
    # Streamed one frame at a time rather than materialised. `read_frames`
    # holds every requested frame, and at 25 MB per 4K frame a few hundred of
    # them exhausts system RAM -- which is what happened once the lowered
    # detection floor multiplied the number of near-spans.
    wanted = set(frames)
    last = max(wanted)
    out: dict[int, list] = {}
    for f, frame in video.iter_frames(clip_path, meta):
        if f in wanted:
            dets = ov.detect([frame], [f], ["open car door"],
                             cfg.openvocab.box_threshold,
                             cfg.openvocab.text_threshold)[0]
        # Grounding DINO returns the matched token span, which for a weakly
        # grounded phrase can be a fragment such as "open" -- accept those too,
        # but they are a signal the grounding is poor (see the arm C findings).
            hits = [(d.box, d.conf) for d in dets
                    if "door" in d.label or d.label.strip() == "open"]
            if hits:
                out[f] = hits
        if f >= last:
            break
    return out


def _span_box_norm(track, span, meta: ClipMeta) -> list[float] | None:
    """Union of a track's boxes over a span, normalized to [0, 1].

    Union, not median. The evaluator compares this against a GT anchor placed on
    one specific frame, and a median box is only where the actor was *typically*
    -- on a panning camera that is nowhere in particular. Measured on
    `mKzCQKTHizw_1`, whose camera pans: the median person box missed the frame-118
    anchor by 0.004 in x and threw away a tIoU-0.81 match. The union covers
    everywhere the actor was during the event, so an anchor on any frame inside
    the span falls within it.

    Normalized because the anchors are, for the same reason everything else here
    is: the set spans 352x288 to 3840x2160.
    """
    if track is None:
        return None
    boxes = [b for f, b in zip(track.frames, track.boxes)
             if span[0] <= f <= span[1]]
    if not boxes:
        return None
    a = np.array(boxes, dtype=float)
    return [round(float(a[:, 0].min() / meta.width), 5),
            round(float(a[:, 1].min() / meta.height), 5),
            round(float(a[:, 2].max() / meta.width), 5),
            round(float(a[:, 3].max() / meta.height), 5)]


def run_clip(clip_path: Path, cfg: C.Config, judge_name: str = "vlm",
             tiled: bool | None = None, batch: int = 8,
             skip_door_cue: bool = False
             ) -> tuple[ClipMeta, list[Interaction], dict]:
    """Whole pipeline for one clip. Unchanged public API over the two halves."""
    ct = track_clip(clip_path, cfg, tiled=tiled, batch=batch)
    return judge_tracks(clip_path, ct, cfg, judge_name, skip_door_cue)


def _free_gpu() -> None:
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _cls_name(cls: int) -> str:
    from .detect.base import COCO_NAMES
    return COCO_NAMES.get(cls, str(cls))


def _decode_gray(clip_path: Path, meta: ClipMeta) -> tuple[dict[int, np.ndarray], float]:
    """Decode the clip once into downscaled grayscale frames.

    Returns the frames and the scale factor mapping native pixel coordinates
    into them, so callers holding native-resolution boxes can convert.
    """
    import cv2

    from .track.gmc import to_gray

    scale = min(1.0, GRAY_MAX_WIDTH / meta.width)
    out: dict[int, np.ndarray] = {}
    for i, frame in video.iter_frames(clip_path, meta):
        g = to_gray(frame)
        if scale < 1.0:
            g = cv2.resize(g, (int(meta.width * scale), int(meta.height * scale)),
                           interpolation=cv2.INTER_AREA)
        out[i] = g
    return out, scale


def _camera_is_static(gray: dict[int, np.ndarray], meta: ClipMeta,
                      sample: int = 12, thresh_frac: float = 0.004) -> bool:
    """Decide static vs moving camera from translation over a ONE-SECOND baseline.

    Measured rather than configured: a hard-coded clip id would silently be
    wrong on a ninth clip, and this is cheap on a handful of frame pairs.

    **The baseline is the crux.** A consecutive-frame baseline cannot separate
    these clips: measured on this set, `gt1125_06`'s drone drift is ~0.24
    px/frame while the static `mKzCQKTHizw_*` clips show ~0.83 px/frame of ORB
    estimation jitter -- so the drone reads as *less* mobile than a tripod. Over
    a second the two separate cleanly, because drift accumulates and zero-mean
    jitter does not. Evidence: `experiments/2026-09-16_camera-motion/`.

    The threshold is a FRACTION of frame width, not pixels: the set spans 352 px
    to 3840 px wide, and a fixed pixel threshold would call the same physical
    drift static on one clip and moving on another.
    """
    from .track.gmc import estimate

    keys = sorted(gray)
    gap = max(1, int(round(meta.fps)))
    if len(keys) < gap + 2:
        return True
    width = gray[keys[0]].shape[1]
    step = max(1, (len(keys) - gap) // sample)
    mags = []
    for a in keys[:-gap:step]:
        b = a + gap
        if b not in gray:
            continue
        r = estimate(gray[a], gray[b])
        if r.ok:
            mags.append(float(np.hypot(r.H[0, 2], r.H[1, 2])) / width)
    if not mags:
        return True
    return float(np.median(mags)) < thresh_frac


def _judge_all(cands, meta, clip_path, cfg, judge_name, feats, persons, vehicles,
               describe: bool = True):
    accepted: list[tuple[Candidate, Verdict]] = []
    rejected: list[tuple[Candidate, Verdict | None]] = []

    if judge_name == "geometric":
        judge = GeometricJudge(feats)
        for c in cands:
            v = judge.judge(c, meta)
            (accepted if v and v.is_interaction else rejected).append((c, v))
        return accepted, rejected

    from .judge.vlm import VLMJudge
    t = cfg.tunable
    key = ("vlm", cfg.vlm.model_id, cfg.vlm.revision, cfg.device,
           cfg.vlm.max_pixels_per_frame)
    if key not in _MODELS:
        _MODELS[key] = VLMJudge(
            cfg.vlm.model_id, revision=cfg.vlm.revision, device=cfg.device,
            cache_dir=cfg.paths.vlm_cache,
            max_new_tokens=cfg.vlm.max_new_tokens,
            max_pixels_per_frame=cfg.vlm.max_pixels_per_frame)
    judge = _MODELS[key]
    # Tunables can change between clips and between LOCO settings while the
    # weights stay put, so they are applied to the cached instance. Anything in
    # the cache KEY changes what the model is; these only change how it is used.
    judge.n_frames = t.n_vlm_frames
    judge.context_pad_s = t.context_pad_s
    judge.crop_margin = t.crop_margin
    judge.conf_thresh = t.vlm_conf_thresh

    pbox = {p.id: dict(zip(p.frames, p.boxes)) for p in persons}
    vbox = {v.id: dict(zip(v.frames, v.boxes)) for v in vehicles}

    # Decode only the frames the judge will look at, and only for a few
    # candidates at a time. Holding every candidate's frames at once is what
    # exhausted RAM on the 4K clip: 25 MB per frame, and the lowered detection
    # floor raised the candidate count.
    from .propose.spans import dilate_span, sample_frames

    def frames_for(c):
        span = dilate_span(c.span, meta.to_frames(t.context_pad_s), meta.n_frames)
        return sample_frames(span, t.n_vlm_frames)

    frame_bytes = meta.width * meta.height * 3
    per_chunk = max(1, int(MAX_JUDGE_FRAME_BYTES // max(1, frame_bytes)))

    pending: list[Candidate] = []
    pending_frames: set[int] = set()

    def run_chunk():
        if not pending:
            return
        frames = video.read_frames(clip_path, sorted(pending_frames), meta)
        for c in pending:
            pb, vb = pbox.get(c.person_id, {}), vbox.get(c.vehicle_id, {})
            v = judge.judge(c, meta, frames=frames,
                            person_boxes=pb, vehicle_boxes=vb)
            if v and v.is_interaction and describe:
                _attach_description(v, judge.describe(c, meta, frames, pb, vb))
            (accepted if v and v.is_interaction else rejected).append((c, v))
        pending.clear()
        pending_frames.clear()
        _free_gpu()

    for c in cands:
        need = set(frames_for(c))
        if pending and len(pending_frames | need) > per_chunk:
            run_chunk()
        pending.append(c)
        pending_frames.update(need)
    run_chunk()
    return accepted, rejected


def _attach_description(v: Verdict, desc: dict | None) -> None:
    """Replace the judge's free-text descriptions with slot-derived ones.

    On a failed parse the judge's free text stays and `*_attrs` stays None, so
    the record is still complete and the evaluator sees it as unscorable rather
    than as wrong.
    """
    if desc is None:
        return
    from .judge import describe as D
    v.person_attrs, v.vehicle_attrs = desc["person"], desc["vehicle"]
    v.person_desc = D.render_person(desc["person"])
    v.vehicle_desc = D.render_vehicle(desc["vehicle"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    ap.add_argument("--judge", choices=("vlm", "geometric"), default="vlm")
    tile_grp = ap.add_mutually_exclusive_group()
    tile_grp.add_argument("--tiled", dest="tiled", action="store_true",
                          default=None,
                          help="force tiled inference (default: by resolution)")
    tile_grp.add_argument("--no-tiled", dest="tiled", action="store_false",
                          help="force plain inference")
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()

    cfg = C.load(args.config)
    seed_everything(cfg.seed)

    outdir = args.outdir or cfg.paths.outputs
    meta, interactions, debug = run_clip(args.clip, cfg, args.judge, args.tiled)

    chash = C.config_hash(cfg)
    write_output(Path(outdir) / f"{meta.clip_id}.json", meta, interactions, chash)
    Path(outdir, f"{meta.clip_id}.debug.json").write_text(
        json.dumps(debug, indent=2) + "\n")
    C.dump_resolved(cfg, Path(outdir) / f"{meta.clip_id}.config.yaml")

    print(f"{meta.clip_id}: {debug['n_candidates']} candidates -> "
          f"{len(interactions)} interactions ({args.judge} judge)")
    for i in interactions:
        print(f"  {i.type:16s} [{i.frame_start:4d}-{i.frame_end:4d}] "
              f"p{i.person.track_id} v{i.vehicle.track_id} conf={i.confidence:.2f}")


if __name__ == "__main__":
    main()
