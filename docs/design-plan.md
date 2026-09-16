# Design Plan — Person-Vehicle Interaction Detection

Reads from `docs/problem-definition.md` (scope, metrics, track decision) and
`docs/labeling-protocol.md` (GT conventions). **Open questions resolved 2026-09-14; detector and tracker
re-decided 2026-09-16 after a literature review (§6). APPROVED by the user
2026-09-16 — implementation is unblocked. Amendments made at approval time are
recorded in §6a.**

Method selection is backed by
`~/research/literature/object-detection-heterogeneous-surveillance/summary.md`.

## 1. Method summary

Hybrid **tracker-proposes / VLM-judges**. Detection + multi-object tracking give
a spatio-temporal skeleton of every person and vehicle. A deterministic geometric
layer reasons over track pairs to *propose* candidate interaction spans, tuned for
recall. A zero-shot vision-language model then *adjudicates* each candidate,
supplying the closed-vocabulary type, the appearance descriptions, and a free-text
note, tuned for precision.

The split exists because the two halves of this problem have different characters.
"Was this person in sustained contact with this vehicle between t1 and t2" is a
geometry question with a deterministic answer. "Was that contact an interaction or
did they just walk past" is a semantic question that geometry answers badly — and
per `problem-definition.md` §2 the dataset is deliberately seeded with 7 labeled
near-misses to punish exactly that confusion. Neither half is load-bearing alone.

No training. Every learned component is off-the-shelf and pinned by revision.

## 2. Architecture

```
pvi/
  config.py          Config dataclasses, YAML load, resolved-config dump
  schema.py          Output/GT dataclasses; the JSON contract
  video.py           Deterministic decode; ClipMeta (fps, size, n_frames)
  detect/
    base.py          Detector ABC
    rfdetr.py        RF-DETR-L via HF transformers (§6.1)
    sliced.py        Tiled-inference wrapper, composes over any Detector
    openvocab.py     Grounding DINO, probe-only until §6.2 resolves
  track/
    tracker.py       roboflow/trackers ByteTrack wrapper -> Track objects
    gmc.py           Global motion compensation — hand-rolled ORB+RANSAC (§6.6)
  propose/
    features.py      Per-pair per-frame feature time series
    rules.py         Feature series -> Candidate spans
  judge/
    base.py          Judge ABC
    vlm.py           VLM adjudicator
    geometric.py     Rule-only judge — the ablation arm
    prompt.py        Frame sampling, cropping, visual prompting, prompt text
  evaluate/
    match.py         Prediction <-> GT temporal matching
    metrics.py       Tiered reporting per problem-definition.md §3
  viz.py             Annotated-frame output (optional deliverable)
  cli.py             Entry point: one clip in, one JSON out
```

**Where abstraction is and isn't warranted.** ABCs only where a second
implementation is actually planned:

- `Detector` — ABC. Two real implementations (a plain detector and the sliced
  wrapper), and the open question in §6 may swap the backbone.
- `Judge` — ABC. Two real implementations, and the second one *is* the headline
  ablation. `geometric.py` is not a stub; it is the control arm.
- `Tracker`, the proposer, and the metrics are **flat modules, no ABC.** One
  implementation each is planned and inventing a base class for a single subclass
  is ceremony. Flagging this explicitly per the implementation skill's instruction
  rather than choosing silently.

## 3. Data flow

```
Videos/*.mp4
  -> video.decode           frames + ClipMeta
  -> detect                 per-frame person/vehicle boxes
  -> track                  Track[] with ids, per-frame boxes, birth/death
  -> propose.features       per (person,vehicle) pair: d_norm(t), speed(t),
                            occluded(t), visible(t), door_delta(t)
  -> propose.rules          Candidate[] (span + triggering rule + features)
  -> judge                  Verdict[] (is_interaction, type, descriptions)
  -> schema                 outputs/<clip_id>.json   [the deliverable]
  -> evaluate               vs data/ground_truth.json -> metrics.json
```

`evaluate` is a separate entry point, not part of the clip pipeline. The
deliverable script takes a clip and emits interactions with no GT dependency.

## 4. Two invariants that drive everything

These follow from `problem-definition.md` §2 and are the reason most of the
design looks the way it does.

**Spatial: never compare pixels.** The set spans 352x288 to 3840x2160. Every
spatial quantity is normalized:

- `d_norm(t) = gap_px(person_box, vehicle_box) / diag_px(vehicle_box)`, where
  `gap_px` is the box-to-box separation, 0 when they overlap.
- Person speed in **body-lengths per second**: `|dx|/person_height/dt`.
- Anchors and any reported coordinate normalized to `[0,1]`.

**Temporal: never count frames.** Frame rate spans 6 to 30 fps. Every threshold
is declared in seconds and converted per clip via `ClipMeta.fps`. The binding
constraint is the **shortest positive event, 0.75 s** — which at `iMGR`'s 6.67 fps
is **5 frames**. Any smoothing window must therefore stay at or below ~0.5 s and
is additionally floored at 3 frames, or it will erase the shortest true events.
(The *median* 4.5 s event is ~27 frames even at 6 fps, so the constraint binds
only at the short tail.)

## 5. Component contracts

**`Detector.detect(frames: list[ndarray], meta: ClipMeta) -> list[list[Detection]]`**
`Detection(frame, cls, box_xyxy, conf)`, classes restricted to person + the COCO
vehicle set. Batched so the sliced wrapper can re-tile without a separate path.

**`SlicedDetector(inner: Detector, tile, overlap)`** — decorator. Tiles each
frame, runs `inner` per tile, maps boxes back, merges by NMS. Used for
`gt1125_06` only (see §6).

**`track(detections, meta) -> list[Track]`**
`Track(id, cls, frames, boxes, birth_frame, death_frame, interpolated_mask)`.
GMC runs unconditionally — identity homography on the 7 static clips costs
almost nothing and means `gt1125_06` needs no separate code path.
Vehicles additionally get `is_static: bool` from bbox-centroid variance over
the track, which gates the `door_delta` feature.

**`features(person: Track, vehicle: Track, meta) -> PairFeatures`**
Per-frame aligned series: `d_norm`, `iou`, `speed_bl_s`, `visible`, `occluded`,
`door_delta`. `door_delta` is mean absolute deviation of the vehicle-box region
from its temporal median, valid **only** when `vehicle.is_static` and the clip is
static-camera; otherwise `NaN` and the rule that uses it is skipped.

**`propose(pairs, meta) -> list[Candidate]`** — four recall-oriented rules,
OR-combined, overlapping spans merged per pair:

| rule | trigger | targets |
|---|---|---|
| R1 dwell | contiguous `near` for >= `min_dwell_s` | `attend`, `load_unload` |
| R2 death-near | person track ends while `near` | `enter_vehicle` |
| R3 birth-near | person track starts while `near` | `exit_vehicle` |
| R4 door-change | `door_delta` spike while any person `near` | `open_close_door` |

`near` uses **hysteresis** — enter at `d_norm < tau_near`, leave at
`d_norm > tau_far` — so a person hovering at the threshold produces one span, not
a burst of fragments. R2/R3 are the rules that structurally separate enter/exit
from pass-by: a pass-by track is born and dies far from the vehicle.

**`Judge.judge(candidate, clip) -> Verdict | None`**
`Verdict(is_interaction, type, person_desc, vehicle_desc, note, confidence)`.
`None` or `is_interaction=False` drops the candidate to the debug artifact.

- `vlm.py`: samples `n_frames` uniformly across the span padded by
  `context_pad_s`, crops to `union(person, vehicle)` dilated by `crop_margin`,
  draws the person box in red and vehicle box in blue (visual prompting — without
  it the model has no way to know *which* person in a crowded frame). Constrained
  JSON decoding, greedy, `temperature=0`.
- `geometric.py`: `is_interaction=True` for every candidate; type from the
  triggering rule (R2 -> enter, R3 -> exit, R4 -> open_close_door, R1 -> attend);
  descriptions from track class and mean box colour. Deliberately crude — its job
  is to show what the VLM adds.

## 6. Resolved decisions

Settled 2026-09-14. Recorded here because each one has a consequence downstream.

1. **Detector: RF-DETR-L** (`Roboflow/rf-detr-large`, Apache-2.0, HF-native).
   56.5 COCO AP at 33.9M params — the best permissively-licensed accuracy found,
   and the only finalist with published AP_S (32.0 at S, 36.1 at M). Backbone is
   DINOv2, which Meta relicensed to Apache-2.0.
   *Supersedes the earlier RT-DETRv2 choice*, which the review found to be the
   weakest of the three permissive finalists (54.3 AP at 76.8M params). The
   previous justification cited "small-object recall"; **no AP_S evidence for
   RT-DETR was found in either direction**, so that claim was unsupported.
   **Fallback: D-FINE-X** (`ustc-community/dfine-xlarge-coco`) — more mature, a
   one-line swap behind the `Detector` ABC. Note RF-DETR XL/2XL are **PML 1.0,
   not Apache**, so the widely-quoted 60.1 AP variant is out of scope.

2. **`gt1125_06`: bounded probe before building, now three arms.**
   (a) RF-DETR-L plain, (b) RF-DETR-L + tiling, (c) Grounding DINO
   (`IDEA-Research/grounding-dino-base`) prompted `"open car door."`.
   Arms (a)/(b): if person recall is near zero, `detect/sliced.py` is dropped as
   wasted complexity and the clip is documented as a known limitation with its 2
   positives excluded from tiered metrics — the exclusion stated, not silent.
   Calibration: SAHI reports **+5-7 AP inference-only** on VisDrone; the larger
   published gains need slicing-aided fine-tuning, which is unavailable to us.
   Expect a useful gain, not a rescue.
   Arm (c) decides whether R4 stays a pixel-change heuristic or becomes a text
   prompt — see §6.7. This remains the one decision deferred to evidence.

3. **Thresholds: leave-one-clip-out.** Tune on 7 clips, evaluate on the held-out
   8th, rotate through all 8 folds. **LOCO is the headline number**; the
   all-data-tuned number is reported alongside and clearly labeled as
   not-held-out. The gap between them is itself a reportable quantity — it
   measures how much the thresholds are overfitting 8 clips.

4. **VLM: build on Qwen2.5-VL-7B-Instruct, report 7B and 32B.** Develop against
   7B for iteration speed; run final numbers on both. Both Apache-2.0, both fit
   in 48 GB. The 7B-vs-32B delta on `attend_vehicle` vs `pass_by` is the
   interesting part.

5. **Accepted risk, not a decision:** `attend_vehicle` vs `pass_by` is the hard
   discrimination and where precision will actually be lost. At n=2 for
   `attend_vehicle` this boundary cannot be tuned — it has to come from the
   prompt's type definitions. Expected, not surprising.

6. **Tracker: `roboflow/trackers`** (Apache-2.0). The AGPL problem cascades past
   the detector: Ultralytics, BoxMOT and motcpp are all AGPL-3.0, so avoiding
   Ultralytics is pointless if the tracker reintroduces the same obligation. The
   original ByteTrack and BoT-SORT repos are MIT but heavyweight and
   detector-coupled.
   **Consequence the earlier plan got wrong:** BoT-SORT's GMC is an OpenCV C++
   VideoStab module with **no Python implementation upstream** — its own authors
   recommend ORB or ECC instead. §5's "GMC runs unconditionally, nearly free" is
   wrong; `track/gmc.py` is hand-rolled ORB+RANSAC homography estimation. Small,
   but real work with its own correctness tests.

7. **Open-vocabulary door cue: probe first, do not design in yet.** R4 currently
   measures appearance change inside the vehicle box against a temporal median,
   which works **only on a static camera with a parked vehicle** — so it is dead
   on `gt1125_06` and on the moving sedan in `iMGR`. A text-prompted detector
   would make the cue scene-independent and delete a conditional branch.
   Against that: Grounding DINO's documented weaknesses are small objects and
   aerial imagery, precisely where the heuristic already fails. Hence arm (c)
   above rather than adoption on the strength of the argument.

## 6a. Amendments at approval (2026-09-16)

Three changes agreed when the plan was approved. They amend the sections above
rather than replacing them.

1. **`min_dwell_s` default corrected to 0.5, range narrowed to [0.3, 0.7].**
   The plan shipped a default of 1.0 against its own validated constraint
   `min_dwell_s < 0.75`, so config load would have rejected `config/default.yaml`.
   The old range [0.3, 3.0] was contradictory for the same reason. The constraint
   is the binding statement; the default and range were wrong.

2. **LOCO searches four thresholds, not ten.** `det_conf`, `tau_near`, `tau_far`
   and `vlm_conf_thresh` vary; the other six stay at their documented defaults.
   Reason: 18 positives over 8 folds is ~2.25 positives per held-out clip, and
   they are unevenly spread (`iMGR` has 6, `gt1125_06` has 2). A fold's F1 moves
   in very coarse steps, so varying ten knobs fits fold-specific noise rather than
   anything that transfers. The frozen six stay declared tunable in
   `config/schema.md` — a later sweep may search them — but nothing reported here
   is selected over them, and the write-up says so.

3. **`paths.frames` is labeling-only and the pipeline must not read it.**
   `tools/extract_frames.py` downscales to max width 1600, so
   `data/frames/gt1125_06/` is 1600x900 — a 2.4x downscale of a 3840x2160 clip,
   and the only clip affected. Feeding that to the detector would have silently
   answered the §6.2 probe on the wrong pixels. `pvi.video` decodes from
   `paths.videos` at native resolution; `data/frames/` serves the labeler and
   contact sheets only.

## 6c. Findings during implementation (2026-09-16)

Things that turned out differently once code met the actual libraries and data.
Each one corrects a statement above.

1. **§6.6 is superseded: GMC is a library call after all.** The plan says
   BoT-SORT's camera motion compensation is an OpenCV C++ VideoStab module with
   no Python implementation upstream, so `track/gmc.py` had to be hand-rolled.
   That is true of the original BoT-SORT repo but **not** of `roboflow/trackers`
   2.6.0, which ships
   `BoTSORTTracker(enable_cmc=True, cmc_method="orb"|"sift"|"sparseOptFlow"|"ecc")`
   with a pure-Python CMC, Apache-2.0. Tracking therefore uses `BoTSORTTracker`
   rather than plain ByteTrack, and **hand-rolled numerical code is off the
   tracking critical path** — which §7 listed as a top risk.
   `pvi/track/gmc.py` survives in a narrower role: estimating how much the
   camera moves, to decide whether a clip is static. The tracker uses CMC
   internally but does not report that quantity, and the `door_delta` gate
   needs it.

2. **Class ids cannot be hard-coded, and the config was wrong.**
   `config/schema.md` listed `detector.classes` as COCO ids `[0,1,2,3,5,7]`,
   assuming the contiguous 80-class scheme. `Roboflow/rf-detr-large` ships
   **91 labels in COCO's original indexing** — `person=1`, `car=3`,
   `motorcycle=4`, `bus=6`, `truck=8`, with `0` being `N/A`. Filtering it by
   `[0,1,2,3,5,7]` would have selected `N/A`, `airplane` and `train` while
   dropping motorcycle, bus and truck — silently, as confident detections of
   the wrong things. The D-FINE-X fallback disagrees again: 80 contiguous
   labels with Pascal-style names (`motorbike`, `aeroplane`).
   Classes are now resolved **by name** against each model's own `id2label` and
   translated to a canonical vocabulary at the detector boundary, so swapping
   the detector cannot change what a downstream class id means. Pinned by tests.

3. **Unconditional CMC is not free.** §5 says "GMC runs unconditionally —
   identity homography on the 7 static clips costs almost nothing". Observed
   behaviour contradicts the premise: when feature matching degenerates, CMC
   does **not** fall back to identity, it produces a wrong warp and corrupts
   data association. On a synthetic featureless clip it fragmented a perfectly
   stationary box into 7 tracks, against 1 with CMC disabled. CMC is therefore
   enabled per clip, from the measured camera motion, rather than always.
   (The synthetic case is adversarial — real footage has features — but it
   establishes that the failure mode is silent corruption rather than graceful
   degradation, which is what the "costs almost nothing" claim assumed away.)

4. **§6.7 is RESOLVED: R4 becomes the open-vocabulary text prompt.** This was
   the one decision the plan deferred to evidence, and the evidence is in
   (`experiments/2026-09-16_door-cue-ground-level/findings.md`).
   The answer is not the binary the plan expected. Grounding DINO grounds
   `"open car door"` reliably on **ground-level** clips — 24/28 frames across
   three clips, boxes verified by eye to land on the actual open door, including
   on the 352×288 grayscale clip — and **fails on the aerial clip**, where it
   missed a clearly open door and returned the token fragment `open`.
   Adopted anyway, because the incumbent heuristic is worse off: needing a static
   camera *and* a parked vehicle, it is dead on 3 of 8 clips after the
   camera-motion correction, one of which contains a door event. The text prompt
   lifts R4 from 5/8 clips to 7/8 and deletes the static-camera branch.
   The heuristic is retired rather than kept as a fallback — the two fail on
   *disjoint* clips, so keeping both would preserve the branch the change exists
   to remove, for one clip's benefit. `gt1125_06` simply has no door cue, which
   costs no recall against the current GT (its 2 positives are both
   `enter_vehicle`) but is a stated limitation.
   Consequences: `door_delta_thresh` (pixel change, [0.04, 0.40]) is replaced by
   `door_conf_thresh` (detector confidence, default 0.30, [0.20, 0.70]); the cue
   runs only on frames already inside a near-span, so it costs a Grounding DINO
   pass over a fraction of frames rather than all of them.

5. **Evaluation matching needed the GT anchors, and the box statistic matters.**
   `problem-definition.md` §3 always required a match to share the
   person-vehicle pair; it was simply unimplemented, and on `NmlzoaDcOuI_6` a
   prediction about a *different vehicle* won a match on temporal overlap alone.
   Implemented via the normalized anchors. The prediction side must be the
   **union** box over the span, not the median: on the panning `mKzCQKTHizw_1`
   the median person box missed the frame-118 anchor by 0.004 in x and discarded
   a tIoU-0.81 correct match.

6. **`ffmpeg 9` removed `-vsync`.** Both `pvi/video.py` and the existing
   `tools/extract_frames.py` used `-vsync 0`, which now fails with
   "Unrecognized option" and yields **zero frames** rather than an error the
   caller notices. Both use `-fps_mode passthrough`.

## 6b. Configuration schema

Full schema in `config/schema.md`; defaults in `config/default.yaml`. The
fixed/tunable split is the contract `hyperparameter-sweep` consumes.

**Fixed** (never varies across runs): data paths, `seed`, device, model ids +
revision hashes, output dirs, decode settings.

**Tunable** (what LOCO and any later sweep search over):

| name | default | range | notes |
|---|---|---|---|
| `det_conf` | 0.35 | [0.10, 0.70] | lower hurts precision, raises tracker noise |
| `tau_near` | 0.15 | [0.05, 0.40] | enter-contact, units of vehicle diagonal |
| `tau_far` | 0.30 | [0.10, 0.80] | leave-contact; **constraint: `tau_far > tau_near`** |
| `min_dwell_s` | 0.5 | [0.3, 0.7] | R1 trigger; must stay below the 0.75 s shortest event |
| `smooth_window_s` | 0.4 | [0.0, 0.8] | floored at 3 frames; see §4 |
| `door_delta_thresh` | 0.12 | [0.04, 0.40] | R4; static vehicles only |
| `context_pad_s` | 1.0 | [0.0, 3.0] | span padding given to the VLM |
| `n_vlm_frames` | 12 | [4, 24] | frames sampled per candidate |
| `crop_margin` | 0.25 | [0.0, 0.6] | box dilation before cropping |
| `vlm_conf_thresh` | 0.5 | [0.0, 0.9] | the main precision/recall dial |

Defaults are first guesses from the GT's geometry, not tuned values — LOCO sets
the reported ones. Two are structurally constrained rather than freely searchable:
`tau_far > tau_near` (hysteresis is meaningless otherwise) and
`min_dwell_s < 0.75` (or R1 cannot fire on the shortest true event).

## 7. Risks

- **VLM determinism.** Greedy decoding is not sufficient on its own; batching and
  attention-kernel nondeterminism can still shift outputs. Mitigation: batch size
  1, deterministic algorithms where available, and a response cache keyed by a
  hash of the exact image bytes + prompt, committed to the repo. A reviewer can
  then reproduce results without a GPU. This is the strongest reproducibility
  lever available and I'd rather commit to it up front.
- **Night and CIF-grayscale detection.** `1THkHYIQ_bY_0` (blown highlights, heavy
  noise) and `HIu4lM4B8hA_1` (352x288, grayscale) will stress person recall.
  Detector failure there is silent — it shows up as missing candidates, not as an
  error. Mitigation: log per-clip detection counts as a first-class diagnostic.
- **ID switches** in the crowded clips (`iMGR`, 6 positives) break R2/R3, which
  depend on true track birth/death rather than a switch artefact.
- **Total positives = 18.** A single event is 5.6% of aggregate recall. Every
  reported number needs its count stated inline.
- **RF-DETR maturity.** ICLR 2026, with HF transformers integration landing
  recently. Mitigated by the `Detector` ABC and the D-FINE-X fallback, but expect
  rougher edges than a two-year-old model.
- **Hand-rolled GMC** (§6.6) is new numerical code on the critical path for one
  clip. Needs its own tests — a silently wrong homography degrades tracking
  without raising anything.

## 8. Environment

Python **3.12** venv at `.venv/` (system Python is 3.14.5; no torch wheels).
`venv` + pinned `requirements.txt`, no conda/poetry. Seeded globally from
`config.seed`; resolved config written next to every output.

## 9. Deliverable shape

```
python -m pvi.cli --clip Videos/NmlzoaDcOuI_6.mp4 --config config/default.yaml
  -> outputs/NmlzoaDcOuI_6.json          # the required artifact
  -> outputs/NmlzoaDcOuI_6.debug.json    # rejected candidates + features
```

Output record, one per (person, vehicle, event), satisfying the task's four
required fields:

```json
{
  "clip_id": "NmlzoaDcOuI_6",
  "clip_meta": {"fps": 6.0, "n_frames": 102, "width": 1280, "height": 720},
  "config_hash": "sha256:...",
  "interactions": [{
    "interaction_id": "NmlzoaDcOuI_6__i001",
    "event_group_id": "NmlzoaDcOuI_6__g001",
    "type": "exit_vehicle",
    "frame_start": 0, "frame_end": 42,
    "time_start_s": 0.0, "time_end_s": 7.0,
    "person": {"track_id": 2, "description": "adult male, green shirt, black trousers, carrying papers"},
    "vehicle": {"track_id": 1, "class": "car", "description": "red sedan, parked, kerbside left"},
    "note": "Opens the passenger door, leans in, closes it and walks away.",
    "confidence": 0.86,
    "evidence": {"rule": "R3_birth_near", "min_d_norm": 0.04, "dwell_s": 6.3}
  }]
}
```

`evidence` is not required by the task. It is there because a reviewer's first
question about any reported interaction will be "why did it think that", and the
answer should not require rerunning the pipeline.
