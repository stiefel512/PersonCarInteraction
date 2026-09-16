"""Bounded three-arm detection probe on `gt1125_06` (design-plan s6.2).

The clip is the moving-camera, oblique-overhead outlier of the set. This probe
decides two things before any pipeline code depends on them:

  arm A  RF-DETR-L, plain, native 3840x2160
  arm B  RF-DETR-L + tiled inference
  arm C  Grounding DINO prompted "open car door." (+ "person.", "car." as a
         sanity control -- a text-prompted detector that cannot find a car is
         not evidence about doors)

Decision rules, fixed here BEFORE looking at the numbers so the probe cannot be
read to suit whatever comes out:

  * If arm A person recall is near zero and arm B does not rescue it, drop
    `detect/sliced.py`, document the clip as a known limitation, and exclude its
    2 positives from the tiered metrics -- stated in the write-up, not silent.
  * If arm B beats arm A materially, keep tiling.
  * If arm B costs much and adds little, drop it; the corrected scale analysis
    (problem-definition.md, "Correction: gt1125_06 person scale") makes this a
    live outcome.
  * Arm C adopts the open-vocab door cue only if it finds doors on THIS clip.
    Finding persons and cars is necessary but not sufficient.

Ground truth available here is thin and the probe says so: the frozen GT has no
per-frame boxes, only 2 person anchors (frames 222, 239). Anchor coverage is
therefore a 2-sample recall check, not a recall number. The detection counts and
the annotated frames are what carry the decision, and they are read by eye.

Usage:
    .venv/bin/python tools/probe_gt1125.py [--arms A,B,C] [--frames 0,120,...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml
from PIL import Image, ImageDraw

from pvi.detect.base import COCO_PERSON, Detection
from pvi.detect.hf import HFDetector
from pvi.detect.sliced import SlicedDetector
from pvi.geometry import Box
from pvi.video import probe, read_frames

CLIP = "gt1125_06"
VIDEO = Path("Videos") / f"{CLIP}.mp4"

# Spread across the clip, and deliberately including the two GT anchor frames.
DEFAULT_FRAMES = [0, 120, 222, 239, 360, 480, 599]

# From data/ground_truth.json (frozen). Normalized (x, y) of the person at the
# event's start frame.
ANCHORS = {222: (0.4973, 0.5619), 239: (0.5768, 0.5651)}

DET_CONF = 0.35            # config/default.yaml
TILE = 704                 # RF-DETR-L native resolution
OVERLAP = 0.25

OV_PHRASES = ["open car door", "person", "car"]


def covers(box: Box, pt: tuple[float, float]) -> bool:
    x, y = pt
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def anchor_check(dets: list[Detection], frame: int, w: int, h: int) -> dict | None:
    if frame not in ANCHORS:
        return None
    ax, ay = ANCHORS[frame]
    pt = (ax * w, ay * h)
    hits = [d for d in dets if d.is_person and covers(d.box, pt)]
    return {
        "anchor_px": [round(pt[0], 1), round(pt[1], 1)],
        "covered_by_person_box": bool(hits),
        "best_conf": round(max((d.conf for d in hits), default=0.0), 4),
    }


def summarize(dets: list[Detection]) -> dict:
    persons = [d for d in dets if d.is_person]
    vehicles = [d for d in dets if not d.is_person]
    heights = [d.box[3] - d.box[1] for d in persons]
    return {
        "n_person": len(persons),
        "n_vehicle": len(vehicles),
        "max_person_conf": round(max((d.conf for d in persons), default=0.0), 4),
        "median_person_height_px": round(float(np.median(heights)), 1) if heights else None,
        "vehicle_classes": sorted({d.name for d in vehicles}),
    }


def annotate(frame: np.ndarray, dets: list[Detection], out: Path,
             anchor: tuple[float, float] | None = None) -> None:
    img = Image.fromarray(frame)
    dr = ImageDraw.Draw(img)
    for d in dets:
        colour = (255, 40, 40) if d.is_person else (40, 120, 255)
        dr.rectangle(d.box, outline=colour, width=4)
        dr.text((d.box[0] + 5, max(0, d.box[1] - 22)),
                f"{d.name} {d.conf:.2f}", fill=colour)
    if anchor:
        x, y = anchor
        dr.ellipse((x - 18, y - 18, x + 18, y + 18), outline=(0, 255, 0), width=5)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Downscale for review: a 4K PNG per frame per arm is unreadable and large.
    img.resize((img.width // 3, img.height // 3), Image.LANCZOS).save(out)


def run_closed_set(det, frames_by_idx, indices, meta, tag, outdir) -> dict:
    per_frame = {}
    t0 = time.perf_counter()
    for i in indices:
        dets = det.detect([frames_by_idx[i]], meta, [i], DET_CONF)[0]
        per_frame[str(i)] = summarize(dets)
        ac = anchor_check(dets, i, meta.width, meta.height)
        if ac:
            per_frame[str(i)]["anchor"] = ac
        anchor_pt = None
        if i in ANCHORS:
            anchor_pt = (ANCHORS[i][0] * meta.width, ANCHORS[i][1] * meta.height)
        annotate(frames_by_idx[i], dets, outdir / f"{tag}_f{i:06d}.png", anchor_pt)
    elapsed = time.perf_counter() - t0
    return {
        "per_frame": per_frame,
        "total_person_detections": sum(v["n_person"] for v in per_frame.values()),
        "total_vehicle_detections": sum(v["n_vehicle"] for v in per_frame.values()),
        "anchors_covered": sum(1 for v in per_frame.values()
                               if v.get("anchor", {}).get("covered_by_person_box")),
        "anchors_total": sum(1 for i in indices if i in ANCHORS),
        "seconds_total": round(elapsed, 2),
        "seconds_per_frame": round(elapsed / max(1, len(indices)), 2),
        "revision": det.revision,
    }


def run_openvocab(frames_by_idx, indices, meta, outdir, device) -> dict:
    from pvi.detect.openvocab import OpenVocabDetector, format_prompt

    ov = OpenVocabDetector(device=device)
    per_frame = {}
    t0 = time.perf_counter()
    for i in indices:
        dets = ov.detect([frames_by_idx[i]], [i], OV_PHRASES)[0]
        by_label: dict[str, int] = {}
        for d in dets:
            by_label[d.label] = by_label.get(d.label, 0) + 1
        per_frame[str(i)] = {
            "n_total": len(dets),
            "by_label": by_label,
            "top": [{"label": d.label, "conf": round(d.conf, 4),
                     "box": [round(v, 1) for v in d.box]} for d in dets[:8]],
        }
        img = Image.fromarray(frames_by_idx[i])
        dr = ImageDraw.Draw(img)
        for d in dets:
            dr.rectangle(d.box, outline=(255, 200, 0), width=4)
            dr.text((d.box[0] + 5, max(0, d.box[1] - 22)),
                    f"{d.label} {d.conf:.2f}", fill=(255, 200, 0))
        outdir.mkdir(parents=True, exist_ok=True)
        img.resize((img.width // 3, img.height // 3), Image.LANCZOS).save(
            outdir / f"C_openvocab_f{i:06d}.png")
    return {
        "prompt": format_prompt(OV_PHRASES),
        "per_frame": per_frame,
        "seconds_total": round(time.perf_counter() - t0, 2),
        "revision": ov.revision,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A,B,C")
    ap.add_argument("--frames", default=",".join(str(f) for f in DEFAULT_FRAMES))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="Roboflow/rf-detr-large")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    arms = [a.strip().upper() for a in args.arms.split(",") if a.strip()]
    indices = [int(f) for f in args.frames.split(",")]

    outdir = Path(args.outdir or f"experiments/{date.today():%Y-%m-%d}_gt1125-probe")
    outdir.mkdir(parents=True, exist_ok=True)
    frames_dir = outdir / "frames"

    meta = probe(VIDEO)
    print(f"{CLIP}: {meta.width}x{meta.height} @ {meta.fps:.2f} fps, "
          f"{meta.n_frames} frames")
    print(f"decoding {len(indices)} frames: {indices}")
    frames_by_idx = read_frames(VIDEO, indices, meta)

    results: dict = {}
    detector = None
    if "A" in arms or "B" in arms:
        detector = HFDetector(args.model, device=args.device,
                              classes=(0, 1, 2, 3, 5, 7))
        print(f"detector revision: {detector.revision}")

    if "A" in arms:
        print("arm A: plain, native resolution ...")
        results["A_plain"] = run_closed_set(detector, frames_by_idx, indices, meta,
                                            "A_plain", frames_dir)
    if "B" in arms:
        print(f"arm B: tiled {TILE}px @ {OVERLAP:.0%} overlap ...")
        sliced = SlicedDetector(inner=detector, tile=TILE, overlap=OVERLAP)
        results["B_tiled"] = run_closed_set(sliced, frames_by_idx, indices, meta,
                                            "B_tiled", frames_dir)
    if "C" in arms:
        print("arm C: Grounding DINO 'open car door.' ...")
        results["C_openvocab"] = run_openvocab(frames_by_idx, indices, meta,
                                               frames_dir, args.device)

    cfg = {
        "experiment": "gt1125_06 three-arm detection probe",
        "date": str(date.today()),
        "clip": CLIP,
        "seed": 0,
        "frames": indices,
        "det_conf": DET_CONF,
        "tile": TILE,
        "overlap": OVERLAP,
        "detector_model_id": args.model,
        "openvocab_model_id": "IDEA-Research/grounding-dino-base",
        "openvocab_phrases": OV_PHRASES,
        "device": args.device,
        "decodes_from": "Videos/gt1125_06.mp4 at native 3840x2160 "
                        "(NOT data/frames/, which is downscaled to 1600 wide)",
    }
    (outdir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (outdir / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    print(f"\nwrote {outdir}/results.json and {outdir}/config.yaml")
    for arm, r in results.items():
        if "total_person_detections" in r:
            print(f"  {arm}: {r['total_person_detections']} person, "
                  f"{r['total_vehicle_detections']} vehicle dets over "
                  f"{len(indices)} frames; anchors "
                  f"{r['anchors_covered']}/{r['anchors_total']}; "
                  f"{r['seconds_per_frame']}s/frame")


if __name__ == "__main__":
    main()
