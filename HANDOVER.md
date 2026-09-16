# Handover — BlackRover person-vehicle interaction detection

`CLAUDE.md` loads automatically. Read these before doing anything else:

1. `docs/problem-definition.md` — scope, data inventory, metrics, Track A decision.
   **Two corrections were made 2026-09-16** (person scale; camera motion) — read them.
2. `docs/design-plan.md` — architecture, contracts, resolved decisions (§6),
   amendments at approval (§6a), **implementation findings that supersede parts
   of §5/§6 (§6c)**, config schema (§6b)
3. `config/schema.md` — fixed vs tunable config split
4. `docs/labeling-protocol.md` — GT conventions + audit log
5. `~/research/literature/object-detection-heterogeneous-surveillance/summary.md`
   — detector/tracker selection, plus a post-hoc note on what the scale
   correction changed

Task statement in `HomeTask.docx` (extract with `unzip` + strip XML).

## Where we are

**The design plan is APPROVED (2026-09-16) and implementation is well underway.**
The `pvi/` package is built, 190 tests pass, and the pipeline runs end to end on
a real clip with the geometric (ablation) judge. The VLM arm is not yet
exercised — weights were still downloading.

### Environment

- `.venv/` — Python 3.12.14, created with `uv`. `requirements.txt` is pinned.
- torch 2.14.0+cu132, transformers 5.17.0, trackers 2.6.0, CUDA works on the A6000.
- **ffmpeg 9 removed `-vsync`.** Both `pvi/video.py` and `tools/extract_frames.py`
  now use `-fps_mode passthrough`. The old spelling yielded *zero frames* rather
  than an error.

### Built and tested

```
pvi/config.py schema.py video.py geometry.py viz.py cli.py
pvi/detect/   base.py hf.py sliced.py openvocab.py
pvi/track/    tracker.py gmc.py
pvi/propose/  spans.py features.py rules.py
pvi/judge/    base.py vlm.py geometric.py prompt.py
pvi/evaluate/ match.py metrics.py run.py
tests/        190 passing
tools/        probe_gt1125.py probe_door_cue.py camera_motion_report.py
              (+ the pre-existing extract_frames / label_gt / validate_gt)
```

Run: `.venv/bin/python -m pvi.cli --clip Videos/<clip>.mp4 [--judge geometric|vlm]`
Evaluate: `.venv/bin/python -m pvi.evaluate.run`
Test: `.venv/bin/python -m pytest tests -q`

### Findings that changed the design

Each is written up where it belongs; this is the index.

1. **`gt1125_06` persons are ~110 px at native 4K, not ~10 px.** The old figure
   was eyeballed from a downscaled contact sheet. The clip is a moving-camera
   problem, not a small-object one. (problem-definition.md, correction note.)
2. **`data/frames/` is downscaled to max width 1600** and is labeling-only.
   `gt1125_06` is 1600×900 there. The pipeline decodes from `Videos/*.mp4` at
   native resolution and never reads it. (design-plan §6a.3.)
3. **3 of 8 clips have a moving camera, not 1.** Both `mKzCQKTHizw_*` clips pan —
   `_0`'s camera follows the runner. Measuring this needed a **one-second**
   baseline; at one frame the drone reads as *more* static than a tripod because
   its drift is below ORB jitter. (problem-definition.md, camera-motion
   correction; `experiments/2026-09-16_camera-motion/`.)
4. **`trackers` 2.6.0 ships a Python CMC**, so §6.6's "GMC must be hand-rolled"
   is obsolete. Tracking uses `BoTSORTTracker(enable_cmc=...)`. `track/gmc.py`
   survives only as the camera-motion diagnostic. (design-plan §6c.1.)
5. **Unconditional CMC is not free** — on degenerate features it corrupts
   association rather than falling back to identity. Enabled per clip.
   (design-plan §6c.3.)
6. **RF-DETR uses COCO's 91-class indexing (`person=1`)**, D-FINE uses 80
   contiguous with Pascal names. The config's hard-coded `[0,1,2,3,5,7]` would
   have selected N/A, airplane and train. Classes are resolved **by name** to a
   canonical vocabulary. (design-plan §6c.2.)
7. **Tile-boundary duplicates**: IoU-only NMS roughly doubled the vehicle count.
   Fixed with intersection-over-smaller suppression, plus an ordering fix so the
   whole box beats the fragment on tied confidence.
   (`experiments/2026-09-16_gt1125-probe/findings.md`.)
8. **The clip-edge guard on R2/R3 was removed.** It suppressed **3 of the 7
   `exit_vehicle` positives** — all start at frame 0. A track born already in
   contact is what an exit looks like; the guard deleted the signal.
   `birth_at_clip_start` / `death_at_clip_end` are now recorded in the evidence.
9. **Matching now checks actor identity** via the GT anchors. Temporal overlap
   alone let a prediction about a *different vehicle* win a match on
   `NmlzoaDcOuI_6`. `problem-definition.md` §3 always required this; it was
   simply unimplemented. The prediction side must be a **union** box over the
   span: on the panning `mKzCQKTHizw_1` a median box missed the anchor by 0.004
   and threw away a tIoU-0.81 correct match.
10. **R4 now uses the open-vocabulary door cue** (design-plan §6c.4). Grounding
   DINO grounds `"open car door"` on 24/28 ground-level frames with boxes on the
   actual door, and fails on the aerial clip. Adopted because the retired
   pixel-change heuristic was dead on 3 clips; R4 goes from 5/8 clips to 7/8 and
   the static-camera branch is gone. `door_delta_thresh` -> `door_conf_thresh`.

### Probe results

`experiments/2026-09-16_gt1125-probe/findings.md`. Arms A and B are decided:

- **The clip is not dropped.** Plain detection covers both GT anchors, finds
  ~5 persons and ~22 vehicles per frame, and detects the box truck.
- **Tiling is kept for `gt1125_06` only**, auto-enabled by resolution
  (`pvi.cli.TILE_MIN_WIDTH = 1920`). It raises anchor confidence 0.66/0.84 →
  0.92/0.93 and finds persons the plain pass misses, at 2.3× runtime.

**Arm C is decided too** — `experiments/2026-09-16_door-cue-ground-level/findings.md`.
The open-vocab door cue is adopted; see finding 10 above.

### End-to-end runs (geometric / ablation judge)

| clip | P | R | tp/fp/fn | pass_by fired |
|---|---|---|---|---|
| `NmlzoaDcOuI_6` | 0.167 | 1.000 | 1/5/0 | 2/2 |
| `mKzCQKTHizw_1` | **1.000** | **1.000** | 1/0/0 | 0/0 |
| pooled | 0.286 | 1.000 | 2/5/0 | 2/2 |

Recall 1.0 with precision 0.286 is the control arm behaving exactly as designed:
it accepts every proposal, so the false-positive count is the number the VLM has
to buy back. `mKzCQKTHizw_1` is a clean single-event clip and the pipeline gets
it exactly right, moving camera and all.

## Next steps

1. **Run the VLM arm.** Qwen2.5-VL-7B was ~1 GB into a ~16 GB download.
   `pvi/judge/vlm.py` is written and cache-backed but has never executed.
2. **Run all 8 clips**, both judges, and compare. The geometric-vs-VLM delta is
   the headline ablation.
3. **LOCO threshold selection** over the agreed four knobs only: `det_conf`,
   `tau_near`, `tau_far`, `vlm_conf_thresh` (design-plan §6a.2). Not written yet.
4. ~~Track fragmentation~~ — **checked, and it is not a problem.** The raw count
   (14 person tracks on a 102-frame clip) looked alarming but the inference was
   wrong. Fragmentation means one identity split into temporally *sequential*
   tracks; these tracks are almost all *concurrent*, which rules that out by
   construction. Verified further: no concurrent pair exceeds mean IoU 0.3, so
   there are no duplicate tracks on one person either, and only 2 of 14 tracks
   have any internal gap. `NmlzoaDcOuI_6` is a busy kerbside — 14 distinct
   people over 17 s is plausible, and the render in the tracking check shows
   several background pedestrians at once. Vehicle `centroid_motion` of 0.2-0.6
   is likewise legitimate: the red sedan drives off, as the clip inventory says.
   No action needed; recorded so the next session does not re-raise it.
5. **Write-up** (≤2 pages) and push the public repo.

## Open question for the user

**Should `Videos/` be committed to the public GitHub repo?** It is 89 MB of the
employer's take-home data (`gt1125_06.mp4` alone is 40 MB, over GitHub's 50 MB
warning threshold though under the 100 MB hard limit). Reproduction needs the
clips, but redistributing their data publicly is their call, not ours. It is
currently **git-ignored pending that decision** — see `.gitignore`.

## Gotchas

- **Frame indexing.** `data/frames/<clip>/` JPEGs are numbered from `000001.jpg`,
  but frame index 0 is `000001.jpg`. GT and all code use 0-based indices.
- **Spans are INCLUSIVE.** `frame_end` is the last frame of contact, so a span
  `[10, 10]` is one frame long. Getting this wrong makes a single-frame event
  score tIoU 0.0 against itself.
- **Config constraints**, validated at load: `tau_far > tau_near`;
  `min_dwell_s < 0.75`; smoothing floored at 3 frames per clip.
- **Normalisation.** All spatial thresholds are normalised by the vehicle-box
  diagonal; all temporal thresholds are in seconds. Never raw pixels or frame
  counts in rule logic.
- **Determinism is graded.** Greedy decoding, VLM batch size 1, float32
  detection, response cache keyed on image bytes + prompt, resolved config
  written next to every output.
- **`attend_vehicle` vs `pass_by`** is the expected precision failure. At n=2 it
  cannot be tuned and has to come from the prompt's type definitions.
- **`data/ground_truth.json` is FROZEN.** If a label must change, log what and
  why in the labeling-protocol audit log.

## Still unverified

- **The VLM path has never run.** `judge/vlm.py`, `judge/prompt.py` and the cache
  are written and unit-tested, but no Qwen forward pass has happened.
- **Model revisions are pinned for RF-DETR only**
  (`f62f7dd5252b61097cbace33886045816dadbde9`, recorded automatically in output).
  `config/default.yaml` still has `revision: null` everywhere; pin them once each
  model has run.
- **No accuracy number on 6 of 8 clips.** Only `NmlzoaDcOuI_6` and
  `mKzCQKTHizw_1` have been run.
- **Tracking quality is only spot-checked**, on `NmlzoaDcOuI_6` and
  `mKzCQKTHizw_1`. Both look sound (see next-steps item 4), but there is no
  quantitative MOT metric and the GT carries no per-frame boxes to compute one
  against.
- **The VIRAT origin of the `NmlzoaDcOuI_*` clips** remains an unchecked
  hypothesis. Worth ~10 minutes; build nothing on it.

## User preferences (also in global CLAUDE.md)

Keep explanations brief. Be explicit about the assumptions a method relies on.
No W&B/MLflow — plain `experiments/<date>_<name>/` dirs with config.yaml and
results.json. Write tests for math, not full coverage. Python only.
