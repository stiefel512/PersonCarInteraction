"""Run every clip under one or both judges and emit a single comparison table.

The geometric-vs-VLM delta is the headline ablation (design-plan §2): the
geometric arm accepts every proposal, so whatever precision it lacks is exactly
what the semantic layer has to buy back. Reporting them side by side in one
table, rather than as two scattered runs, is the point.

Per-clip rows are mandatory, not a convenience. With n=8 a single mean hides
everything, and two clips are named stress cases -- `mKzCQKTHizw_0` for
precision, `gt1125_06` for recall.

Usage:
    .venv/bin/python tools/run_all.py --judges geometric,vlm
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from pvi import config as C
from pvi.cli import run_clip, seed_everything
from pvi.evaluate.metrics import full_report
from pvi.schema import GTEvent, load_ground_truth, write_output

STRESS = {"mKzCQKTHizw_0": "precision stress (runners near parked cars)",
          "gt1125_06": "recall stress (moving camera, oblique overhead)"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    ap.add_argument("--judges", default="geometric,vlm")
    ap.add_argument("--videos", type=Path, default=Path("Videos"))
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()

    judges = [j.strip() for j in args.judges.split(",") if j.strip()]
    cfg = C.load(args.config)
    seed_everything(cfg.seed)

    outdir = args.outdir or Path(f"experiments/{date.today():%Y-%m-%d}_run-all")
    outdir.mkdir(parents=True, exist_ok=True)

    gts = load_ground_truth(cfg.paths.ground_truth)
    gt_by_clip: dict[str, list[GTEvent]] = {}
    for g in gts:
        gt_by_clip.setdefault(g.clip_id, []).append(g)

    clips = sorted(args.videos.glob("*.mp4"))
    rows: list[dict] = []
    reports: dict = {}

    for judge in judges:
        pooled_preds = []
        for clip in clips:
            t0 = time.perf_counter()
            meta, interactions, debug = run_clip(clip, cfg, judge)
            elapsed = time.perf_counter() - t0
            pooled_preds.extend(interactions)

            jd = outdir / judge
            write_output(jd / f"{meta.clip_id}.json", meta, interactions,
                         C.config_hash(cfg))
            (jd / f"{meta.clip_id}.debug.json").write_text(
                json.dumps(debug, indent=2) + "\n")

            rep = full_report(interactions, gt_by_clip.get(meta.clip_id, []))
            reports[f"{judge}/{meta.clip_id}"] = rep
            d = rep["primary"]["tier1_detection"]
            pb = rep["primary"]["tier1_pass_by"]
            rows.append({
                "judge": judge, "clip": meta.clip_id,
                "n_gt_positives": d["n_gt"], "n_pred": d["n_pred"],
                "tp": d["tp"], "fp": d["fp"], "fn": d["fn"],
                "precision": d["precision"], "recall": d["recall"], "f1": d["f1"],
                "pass_by_labeled": pb["n_pass_by_labeled"],
                "pass_by_fired": pb["n_pass_by_falsely_fired"],
                "n_candidates": debug["n_candidates"],
                "static_camera": debug["static_camera"],
                "tiled": debug["tiled"],
                "seconds": round(elapsed, 1),
                "note": STRESS.get(meta.clip_id, ""),
            })
            print(f"  {judge:10s} {meta.clip_id:20s} "
                  f"P={d['precision']:.3f} R={d['recall']:.3f} F1={d['f1']:.3f} "
                  f"({d['tp']}/{d['fp']}/{d['fn']})  {elapsed:.0f}s")

        rep = full_report(pooled_preds, gts)
        reports[f"{judge}/POOLED"] = rep
        d = rep["primary"]["tier1_detection"]
        pb = rep["primary"]["tier1_pass_by"]
        rows.append({
            "judge": judge, "clip": "POOLED",
            "n_gt_positives": d["n_gt"], "n_pred": d["n_pred"],
            "tp": d["tp"], "fp": d["fp"], "fn": d["fn"],
            "precision": d["precision"], "recall": d["recall"], "f1": d["f1"],
            "pass_by_labeled": pb["n_pass_by_labeled"],
            "pass_by_fired": pb["n_pass_by_falsely_fired"],
            "n_candidates": "", "static_camera": "", "tiled": "", "seconds": "",
            "note": "all 8 clips",
        })
        print(f"  {judge:10s} {'POOLED':20s} "
              f"P={d['precision']:.3f} R={d['recall']:.3f} F1={d['f1']:.3f}")

    with (outdir / "comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (outdir / "results.json").write_text(json.dumps(reports, indent=2) + "\n")
    C.dump_resolved(cfg, outdir / "config.yaml")

    print(f"\nwrote {outdir}/comparison.csv")


if __name__ == "__main__":
    main()
