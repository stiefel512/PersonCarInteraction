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
from pvi.video import probe

STRESS = {"mKzCQKTHizw_0": "precision stress (runners near parked cars)",
          "gt1125_06": "recall stress (moving camera, oblique overhead)"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    ap.add_argument("--judges", default="geometric,vlm")
    ap.add_argument("--videos", type=Path, default=Path("Videos"))
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--clips", default=None,
                    help="comma-separated clip ids; default all. Running one "
                         "clip per process bounds page-cache growth, which on "
                         "this machine tripped the OOM watchdog partway through "
                         "the set even with ~50 GB genuinely available.")
    ap.add_argument("--rescore", action="store_true",
                    help="re-score existing outputs in --outdir without "
                         "re-running inference; use after a change to matching "
                         "or metrics, which must not cost GPU time to evaluate")
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
    if args.clips:
        want = {c.strip() for c in args.clips.split(",") if c.strip()}
        clips = [c for c in clips if c.stem in want]
    rows: list[dict] = []
    reports: dict = {}

    for judge in judges:
        pooled_preds = []
        for clip in clips:
            jd = outdir / judge
            if args.rescore:
                from pvi.evaluate.run import load_predictions
                out_json = jd / f"{clip.stem}.json"
                if not out_json.exists():
                    continue
                _, interactions = load_predictions(out_json)
                debug = json.loads((jd / f"{clip.stem}.debug.json").read_text())
                meta = probe(clip)
                elapsed = float("nan")
            else:
                t0 = time.perf_counter()
                meta, interactions, debug = run_clip(clip, cfg, judge)
                elapsed = time.perf_counter() - t0
                write_output(jd / f"{meta.clip_id}.json", meta, interactions,
                             C.config_hash(cfg))
                (jd / f"{meta.clip_id}.debug.json").write_text(
                    json.dumps(debug, indent=2) + "\n")
            pooled_preds.extend(interactions)

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
                "seconds": ("" if elapsed != elapsed else round(elapsed, 1)),
                "note": STRESS.get(meta.clip_id, ""),
            })
            print(f"  {judge:10s} {meta.clip_id:20s} "
                  f"P={d['precision']:.3f} R={d['recall']:.3f} F1={d['f1']:.3f} "
                  f"({d['tp']}/{d['fp']}/{d['fn']})"
                  + ("" if elapsed != elapsed else f"  {elapsed:.0f}s"))

        if args.clips:
            continue          # partial run: a pooled row here would be a lie
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
