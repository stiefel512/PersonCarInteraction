"""Evaluation entry point: compare outputs/ against the frozen ground truth.

Separate from the clip pipeline on purpose. The deliverable script takes a clip
and emits interactions with no ground-truth dependency; this reads what it
produced and scores it. Keeping them apart is what stops the GT from leaking
into the thing being evaluated.

Reports per-clip and pooled. Per-clip is mandatory rather than nice-to-have:
with n=8 a single mean hides everything, and two clips are named stress cases
(`mKzCQKTHizw_0` for precision, `gt1125_06` for recall).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..schema import (GTEvent, Interaction, PersonRef, VehicleRef,
                      load_ground_truth)
from .metrics import full_report, full_report_groups

# Named in problem-definition.md s3 -- reported individually, never only pooled.
STRESS_CLIPS = {"mKzCQKTHizw_0": "precision (runners near parked cars)",
                "gt1125_06": "recall (moving camera, oblique overhead)"}


def load_predictions(path: Path) -> tuple[str, list[Interaction]]:
    raw = json.loads(path.read_text())
    out = []
    for r in raw.get("interactions", []):
        v = dict(r["vehicle"])
        out.append(Interaction(
            interaction_id=r["interaction_id"],
            event_group_id=r.get("event_group_id", r["interaction_id"]),
            type=r["type"],
            frame_start=r["frame_start"], frame_end=r["frame_end"],
            time_start_s=r["time_start_s"], time_end_s=r["time_end_s"],
            person=PersonRef(track_id=r["person"]["track_id"],
                             description=r["person"].get("description", "")),
            vehicle=VehicleRef(track_id=v["track_id"], cls=v.get("class", ""),
                               description=v.get("description", "")),
            note=r.get("note", ""), confidence=r.get("confidence", 0.0),
            evidence=r.get("evidence", {}),
        ))
    return raw["clip_id"], out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", type=Path, default=Path("outputs"))
    ap.add_argument("--ground-truth", type=Path, default=Path("data/ground_truth.json"))
    ap.add_argument("--report", type=Path, default=Path("outputs/metrics.json"))
    args = ap.parse_args()

    gts = load_ground_truth(args.ground_truth)
    by_clip_gt: dict[str, list[GTEvent]] = {}
    for g in gts:
        by_clip_gt.setdefault(g.clip_id, []).append(g)

    pred_files = sorted(p for p in args.outputs.glob("*.json")
                        if not p.name.endswith((".debug.json", "metrics.json")))
    if not pred_files:
        raise SystemExit(f"no prediction files in {args.outputs}/")

    groups: list[tuple[list[Interaction], list[GTEvent]]] = []
    per_clip: dict[str, dict] = {}
    covered: set[str] = set()
    for p in pred_files:
        clip_id, preds = load_predictions(p)
        covered.add(clip_id)
        # Per-clip GROUPS, never a flat pool: frame spans are clip-local, so a
        # flat matching lets a prediction from one clip satisfy another clip's
        # event. See metrics.report_groups.
        groups.append((preds, by_clip_gt.get(clip_id, [])))
        per_clip[clip_id] = full_report(preds, by_clip_gt.get(clip_id, []))
        if clip_id in STRESS_CLIPS:
            per_clip[clip_id]["stress_case"] = STRESS_CLIPS[clip_id]

    # A clip with no prediction file is not a clip with no events -- it is a
    # clip that was never run, and silently scoring it as all-misses would
    # understate recall for a reason that has nothing to do with the method.
    missing = sorted(set(by_clip_gt) - covered)

    report = {
        "n_clips_scored": len(covered),
        "clips_without_predictions": missing,
        "n_gt_events_total": len(gts),
        "n_gt_positives_total": sum(1 for g in gts if g.is_positive),
        "pooled": full_report_groups(groups),
        "per_clip": per_clip,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")

    pooled = report["pooled"]["primary"]["tier1_detection"]
    pb = report["pooled"]["primary"]["tier1_pass_by"]
    print(f"scored {len(covered)} clip(s)"
          + (f"; NOT RUN: {missing}" if missing else ""))
    print(f"pooled tIoU>=0.3  P={pooled['precision']:.3f} R={pooled['recall']:.3f} "
          f"F1={pooled['f1']:.3f}  (tp={pooled['tp']} fp={pooled['fp']} fn={pooled['fn']}, "
          f"n_gt={pooled['n_gt']})")
    print(f"pass_by false fires: {pb['n_pass_by_falsely_fired']}"
          f"/{pb['n_pass_by_labeled']}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
