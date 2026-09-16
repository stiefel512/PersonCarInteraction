# `gt1125_06` detection probe — findings

Design-plan §6.2. Decision rules were fixed in `tools/probe_gt1125.py` **before**
the numbers were seen.

Run: `.venv/bin/python tools/probe_gt1125.py --arms A,B`
Config: `config.yaml`. Raw numbers: `results.json`. Annotated frames: `frames/`.
7 frames spread across the clip, including both GT anchor frames (222, 239).
Detector `Roboflow/rf-detr-large` @ `f62f7dd5252b61097cbace33886045816dadbde9`,
`det_conf=0.35`, decoded from `Videos/gt1125_06.mp4` at native 3840×2160.

## Result

| | arm A (plain) | arm B (tiled 704 px / 25%) |
|---|---|---|
| person detections (7 frames) | 37 | 42 |
| vehicle detections (7 frames) | 155 | 207 |
| GT anchors covered | **2/2** | **2/2** |
| anchor confidence (f222 / f239) | 0.66 / 0.84 | **0.92 / 0.93** |
| median person height | 93 px | 84 px |
| seconds per frame | 0.53 | 1.21 |

## Decisions

**1. The clip is NOT dropped.** The "recall near zero → document as a known
limitation and exclude its 2 positives" branch is off the table. Plain detection
already finds ~5 persons and ~22 vehicles per frame, covers both GT anchors, and
detects the box truck. This follows the earlier correction that persons here are
~110 px at native resolution, not ~10 px.

**2. Tiling is kept, for this clip only.** It is enabled by resolution
(`pvi.cli.TILE_MIN_WIDTH = 1920`), so in the current set only `gt1125_06` uses
it. Below the tile size `tile_origins` returns a single tile and the tiled path
is just the plain path run twice.

What tiling buys, in order of how much it convinced me:

- **Confidence on the two GT persons rises from 0.66/0.84 to 0.92/0.93.** This
  matters more than the count: `det_conf` is a tunable the LOCO search will move,
  and a detection at 0.66 is one threshold step from disappearing.
- **It finds persons plain detection misses** — visible in `frames/`: the figure
  by the green bin and the one against the wall appear in B and not in A.
- Raw person count +14%, vehicle count +34%.

Cost is 2.3× runtime (0.53 → 1.21 s/frame; ~12 min for the full 600-frame clip
against ~5 min). The task's runtime budget is soft, so this is affordable.

**3. Tile merging needed fixing, and the first version was wrong.** The initial
run reported 336 vehicle detections against plain's 155 — roughly double, almost
entirely **duplicates**, visible as stacked boxes on the minivan and several
parked cars. Cause: an object crossing a tile boundary is detected once as a
truncated fragment and once whole; a fragment that is half the full box has
IoU 0.5 with it and survives IoU-0.55 NMS.

Fixed by suppressing on **intersection-over-smaller** (containment, `ios ≥ 0.75`)
as well as IoU: the fragment's IoS against the whole box is ~1.0. Vehicle count
dropped 336 → 207 and the visible duplicates are largely gone. Suppression is
class-wise, so a person contained in a car's box is never removed.

A second bug surfaced while fixing this: with tied confidence, greedy NMS was
keeping the *fragment* and suppressing the *whole* box. Ordering is now
confidence, then **larger area**, then coordinates. A truncated box in place of
a whole vehicle would have mis-placed every downstream `d_norm`, since `d_norm`
is normalised by the vehicle diagonal.

## Caveats

- **7 frames, 2 anchors.** The frozen GT has no per-frame boxes for this clip, so
  "anchors covered" is a 2-sample check, not a recall figure. The decision rests
  on the annotated frames read by eye plus the confidence shift, and is stated
  that way rather than dressed up as a measurement.
- **Detection counts are not precision.** More boxes is not better; the duplicate
  problem above is exactly that failure. The counts are only interpretable
  alongside `frames/`.
- **Viewpoint, not scale, is the residual risk.** This is an oblique overhead
  view and COCO is overwhelmingly ground-level. Tiling does not address that at
  all. Detection is better here than expected, but nothing in the probe measures
  how much accuracy the viewpoint still costs.

## Arm C

Run separately — see `../2026-09-16_gt1125-probe-armC/`.
