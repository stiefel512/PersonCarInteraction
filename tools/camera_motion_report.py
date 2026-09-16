"""Measure per-clip camera motion, to decide where CMC should run.

The design plan assumed GMC could run unconditionally because it would collapse
to identity on a static camera. That is not what happens: when feature matching
degenerates, CMC returns a wrong warp rather than identity, and a wrong warp
corrupts data association silently. So the tracker enables CMC per clip, and
this tool produces the evidence for that switch.

Reports the median consecutive-frame translation as a fraction of frame width.
A fraction, not pixels: the set spans 352 to 3840 px wide, and the same physical
drift is 10x the pixel count on one clip versus another.

Usage:
    .venv/bin/python tools/camera_motion_report.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import yaml

from pvi.track.gmc import estimate, to_gray
from pvi.video import iter_frames, probe

GRAY_MAX_WIDTH = 960      # matches pvi.cli.GRAY_MAX_WIDTH
N_PAIRS = 15
THRESH_FRAC = 0.004       # matches pvi.cli._camera_is_static


def sample_pairs(clip: Path, n_pairs: int, gap: int
                 ) -> tuple[object, list[tuple[np.ndarray, np.ndarray]]]:
    """Frame pairs `gap` frames apart, spread across the clip.

    The gap is the whole point. Consecutive frames cannot separate a drifting
    camera from a static one: measured on this set, `gt1125_06`'s drone drift is
    ~0.24 px/frame while the static `mKzCQKTHizw_*` clips show ~0.83 px/frame of
    ORB estimation jitter. The drift is *below the noise* at a one-frame
    baseline.

    Over a longer baseline they separate cleanly, because drift accumulates and
    zero-mean jitter does not.
    """
    meta = probe(clip)
    scale = min(1.0, GRAY_MAX_WIDTH / meta.width)
    starts = list(range(0, max(1, meta.n_frames - gap),
                        max(1, (meta.n_frames - gap) // n_pairs)))
    wanted: set[int] = set()
    for a in starts:
        wanted.update((a, a + gap))

    got: dict[int, np.ndarray] = {}
    for i, frame in iter_frames(clip, meta):
        if i in wanted:
            g = to_gray(frame)
            if scale < 1.0:
                g = cv2.resize(g, (int(meta.width * scale), int(meta.height * scale)),
                               interpolation=cv2.INTER_AREA)
            got[i] = g
        if i >= max(wanted):
            break
    return meta, [(got[a], got[a + gap]) for a in starts
                  if a in got and a + gap in got]


def main() -> None:
    outdir = Path(f"experiments/{date.today():%Y-%m-%d}_camera-motion")
    outdir.mkdir(parents=True, exist_ok=True)

    results = {}
    print(f"{'clip':22s}{'resolution':12s}{'1-frame':>10s}{'1-second':>11s}  verdict")
    for clip in sorted(Path("Videos").glob("*.mp4")):
        row = {}
        for label in ("one_frame", "one_second"):
            meta = probe(clip)
            gap = 1 if label == "one_frame" else max(1, int(round(meta.fps)))
            meta, pairs = sample_pairs(clip, N_PAIRS, gap)
            mags, oks, reasons = [], 0, []
            for a, b in pairs:
                r = estimate(a, b)
                if r.ok:
                    oks += 1
                    mags.append(float(np.hypot(r.H[0, 2], r.H[1, 2])) / a.shape[1])
                else:
                    reasons.append(r.reason)
            row[label] = {
                "gap_frames": gap,
                "n_pairs": len(pairs),
                "n_estimates_ok": oks,
                "median_translation_frac_width":
                    round(float(np.median(mags)), 6) if mags else None,
                "max_translation_frac_width":
                    round(float(max(mags)), 6) if mags else None,
                "rejected_reasons": sorted(set(reasons)),
            }
        one_s = row["one_second"]["median_translation_frac_width"]
        static = (one_s is None) or (one_s < THRESH_FRAC)
        row["resolution"] = f"{meta.width}x{meta.height}"
        row["verdict"] = "static" if static else "moving"
        results[meta.clip_id] = row
        print(f"{meta.clip_id:22s}{row['resolution']:12s}"
              f"{row['one_frame']['median_translation_frac_width']:>10.5f}"
              f"{one_s:>11.5f}  {'STATIC' if static else 'MOVING'}")

    cfg = {"experiment": "per-clip camera motion", "date": str(date.today()),
           "n_pairs_per_clip": N_PAIRS, "gray_max_width": GRAY_MAX_WIDTH,
           "static_threshold_frac_width": THRESH_FRAC, "seed": 0,
           "method": "ORB + RANSAC homography; median translation as a fraction "
                     "of frame width, at a 1-frame and a 1-second baseline",
           "note": "the 1-second baseline is the decisive one: drift accumulates, "
                   "zero-mean estimation jitter does not"}
    (outdir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (outdir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {outdir}/results.json")


if __name__ == "__main__":
    main()
