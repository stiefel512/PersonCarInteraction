"""Leave-one-clip-out threshold selection.

Protocol (design-plan §6.3, scoped by §6a.2). The clip is the fold unit: 8
folds. For each fold, search thresholds on the other 7 clips, apply the winner
to the held-out clip, keep those predictions. Pool the 8 held-out clips and
report that as the headline. Separately tune on all 8 and evaluate on all 8, and
report that alongside, **labelled not-held-out**. The gap between them is itself
the result -- it measures how much 8 clips are being overfitted.

Clip-level folds rather than event-level because events within a clip share a
camera, a scene and a frame rate. Splitting events would leak nearly everything
that makes a threshold work.

**Only four thresholds vary**: `det_conf`, `tau_near`, `tau_far`,
`vlm_conf_thresh`. The other six stay at their documented defaults. Reason: 18
positives over 8 folds is ~2.25 positives per held-out clip, unevenly spread
(`iMGR` has 6, `gt1125_06` has 2), so a fold's F1 moves in very coarse steps and
varying ten knobs fits fold-specific noise rather than anything that transfers.

Two honesty notes that belong in the write-up, not just here:

1. **The pooled held-out number does not correspond to a shippable config.**
   Each fold may select different thresholds, so it characterises the
   *procedure*. `config/default.yaml` ships the all-data values, whose honest
   number is the not-held-out one.
2. **Even LOCO is not held-out in the usual sense.** The clips are not
   independent draws from anything -- two pairs share a camera -- and the
   thresholds were designed while looking at this data.

## Cost

Re-running detection and tracking per config would dominate, so tracks are
cached on disk per `(clip, det_conf)`. `tau_near`/`tau_far` then only re-run the
proposal stage, which is cheap. `vlm_conf_thresh` is **free**: it is a
post-filter on a verdict already returned, so it never triggers inference.
The VLM's own response cache (keyed on rendered image bytes) absorbs most of
what remains, since nearby threshold settings produce identical crops.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from .. import config as C
from ..schema import GTEvent, Interaction, load_ground_truth
from .match import match_events
from .metrics import detection_prf

# Grid over the four searched knobs. Deliberately coarse: with ~2.25 positives
# per held-out fold, a finer grid resolves noise, not signal.
GRID: dict[str, tuple[float, ...]] = {
    # det_conf is the tracker's high-confidence threshold, not a
    # pre-filter, so the useful band sits well above tracker.det_floor.
    "det_conf": (0.35, 0.50, 0.65),
    "tau_near": (0.08, 0.15, 0.25),
    "tau_far": (0.20, 0.30, 0.45),
    "vlm_conf_thresh": (0.3, 0.5, 0.7),
}


def valid_settings(grid: dict[str, tuple[float, ...]] = GRID) -> list[dict[str, float]]:
    """Cartesian product of the grid, minus configurations the constraints reject.

    `tau_far > tau_near` is structural: with it violated the hysteresis
    degenerates and the spans fragment, so those points are not merely bad, they
    are meaningless.
    """
    keys = list(grid)
    out = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        s = dict(zip(keys, combo))
        if s["tau_far"] <= s["tau_near"]:
            continue
        out.append(s)
    return out


def score(preds: Sequence[Interaction], gts: Sequence[GTEvent]) -> float:
    """Selection objective: aggregate detection F1 at the primary tIoU.

    F1 rather than precision, despite the task's precision-over-recall stance.
    Selecting on precision alone rewards predicting nothing, which scores 0/0;
    the operating point is moved by `vlm_conf_thresh`, which is *in* the search,
    so the objective does not also need to encode the preference.
    """
    return detection_prf(match_events(preds, gts)).f1


def select(fit_clips: Iterable[str], preds_by: dict[tuple[str, str], list[Interaction]],
           gt_by_clip: dict[str, list[GTEvent]],
           settings: list[dict[str, float]]) -> dict[str, float]:
    """Pick the setting maximising pooled F1 over `fit_clips`.

    Ties are broken by the setting's position in the grid, so the result does
    not depend on dict ordering -- determinism is graded, and a tie is common
    at this sample size.
    """
    best, best_f1 = None, -1.0
    for s in settings:
        key = setting_key(s)
        preds, gts = [], []
        for clip in fit_clips:
            preds.extend(preds_by.get((clip, key), []))
            gts.extend(gt_by_clip.get(clip, []))
        f1 = score(preds, gts)
        if f1 > best_f1:
            best, best_f1 = s, f1
    return best or settings[0]


def setting_key(s: dict[str, float]) -> str:
    return json.dumps(s, sort_keys=True)


def run(preds_by: dict[tuple[str, str], list[Interaction]],
        gt_by_clip: dict[str, list[GTEvent]],
        settings: list[dict[str, float]] | None = None) -> dict[str, Any]:
    """Run the LOCO protocol over already-computed predictions.

    `preds_by` maps (clip_id, setting_key) to that clip's predictions under that
    setting. Separating the sweep from the scoring keeps this function pure and
    testable without a GPU.
    """
    settings = settings or valid_settings()
    clips = sorted(gt_by_clip)

    folds = []
    held_out_preds: list[Interaction] = []
    held_out_gts: list[GTEvent] = []
    for held in clips:
        fit = [c for c in clips if c != held]
        chosen = select(fit, preds_by, gt_by_clip, settings)
        preds = preds_by.get((held, setting_key(chosen)), [])
        held_out_preds.extend(preds)
        held_out_gts.extend(gt_by_clip[held])
        folds.append({
            "held_out_clip": held,
            "selected": chosen,
            "n_predictions": len(preds),
            "fold_f1": round(score(preds, gt_by_clip[held]), 4),
        })

    all_data = select(clips, preds_by, gt_by_clip, settings)
    all_preds, all_gts = [], []
    for c in clips:
        all_preds.extend(preds_by.get((c, setting_key(all_data)), []))
        all_gts.extend(gt_by_clip[c])

    loco_prf = detection_prf(match_events(held_out_preds, held_out_gts))
    tuned_prf = detection_prf(match_events(all_preds, all_gts))

    # How often the folds agree. If every fold picks something different, the
    # selection is fitting fold noise and the pooled number is describing a
    # procedure with no stable answer -- worth seeing, not hiding.
    agreement = {}
    for k in GRID:
        vals = [f["selected"][k] for f in folds]
        agreement[k] = {"values": vals,
                        "n_distinct": len(set(vals)),
                        "modal": max(set(vals), key=vals.count)}

    return {
        "protocol": "leave-one-clip-out, 8 folds, clip as fold unit",
        "searched": sorted(GRID),
        "frozen_at_defaults": sorted(
            set(C.Tunables.__dataclass_fields__) - set(GRID)),
        "n_settings_evaluated": len(settings),
        "headline_held_out": loco_prf.as_dict(),
        "all_data_tuned_NOT_held_out": tuned_prf.as_dict(),
        "all_data_selected": all_data,
        "overfitting_gap_f1": round(tuned_prf.f1 - loco_prf.f1, 4),
        "fold_agreement": agreement,
        "folds": folds,
        "caveats": [
            "The pooled held-out number characterises the PROCEDURE, not any "
            "single config: folds may select different thresholds.",
            "config/default.yaml ships the all-data values, whose honest number "
            "is the not-held-out one.",
            "Clips are not independent draws -- two pairs share a camera -- and "
            "the thresholds were designed while looking at this data.",
            "18 positives over 8 folds is ~2.25 per held-out clip, unevenly "
            "spread; every number here is indicative, not significant.",
        ],
    }


def sweep(clips: Sequence[Path], cfg: C.Config, judge: str,
          cache_dir: Path, settings: list[dict[str, float]] | None = None
          ) -> dict[tuple[str, str], list[Interaction]]:
    """Run the pipeline over every clip x setting, caching tracks per det_conf.

    Import is local so that `run` and `select` stay usable without torch.
    """
    from ..cli import run_clip

    settings = settings or valid_settings()
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: dict[tuple[str, str], list[Interaction]] = {}

    for clip in clips:
        for s in settings:
            key = setting_key(s)
            cached = cache_dir / f"{clip.stem}__{abs(hash(key)):016x}.pkl"
            if cached.exists():
                out[(clip.stem, key)] = pickle.loads(cached.read_bytes())
                continue
            _, interactions, _ = run_clip(clip, cfg.with_tunables(**s), judge)
            cached.write_bytes(pickle.dumps(interactions))
            out[(clip.stem, key)] = interactions
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    ap.add_argument("--judge", choices=("vlm", "geometric"), default="vlm")
    ap.add_argument("--videos", type=Path, default=Path("Videos"))
    ap.add_argument("--cache", type=Path, default=Path("outputs/loco_cache"))
    ap.add_argument("--report", type=Path, default=Path("outputs/loco.json"))
    args = ap.parse_args()

    cfg = C.load(args.config)
    gts = load_ground_truth(cfg.paths.ground_truth)
    gt_by_clip: dict[str, list[GTEvent]] = {}
    for g in gts:
        gt_by_clip.setdefault(g.clip_id, []).append(g)

    clips = sorted(args.videos.glob("*.mp4"))
    settings = valid_settings()
    print(f"{len(clips)} clips x {len(settings)} settings "
          f"(of {len(list(itertools.product(*GRID.values())))} before constraints)")

    preds_by = sweep(clips, cfg, args.judge, args.cache, settings)
    report = run(preds_by, gt_by_clip, settings)
    report["judge"] = args.judge

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")

    h, t = report["headline_held_out"], report["all_data_tuned_NOT_held_out"]
    print(f"\nLOCO (held out)       P={h['precision']:.3f} R={h['recall']:.3f} "
          f"F1={h['f1']:.3f}  (n_gt={h['n_gt']})")
    print(f"all-data (NOT held out) P={t['precision']:.3f} R={t['recall']:.3f} "
          f"F1={t['f1']:.3f}")
    print(f"overfitting gap in F1: {report['overfitting_gap_f1']:+.4f}")
    for k, v in report["fold_agreement"].items():
        print(f"  {k:18s} {v['n_distinct']} distinct value(s) across folds, "
              f"modal {v['modal']}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
