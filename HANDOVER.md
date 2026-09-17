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
The `pvi/` package is built, 214 tests pass, and the pipeline runs end to end on
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
tests/        214 passing
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
9. **Matching RANKS on actor identity; it must not gate.** Using the GT anchors
   as a hard filter was measured to be net-harmful: it cost 5 of 17 true
   positives and lowered *both* precision (0.327 -> 0.231) and recall (0.944 ->
   0.667), since a rejected match becomes a false positive *and* a false
   negative. It failed wherever the person track fragmented. Agreement now sorts
   pairings ahead of non-agreement, with tIoU breaking ties.
   Original problem it solves: temporal overlap alone let a prediction about a
   *different vehicle* win a match on `NmlzoaDcOuI_6`. The prediction side must
   be a **union** box over the span, not a median: on the panning
   `mKzCQKTHizw_1` a median box missed the anchor by 0.004 and threw away a
   tIoU-0.81 correct match.
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

### Current result — all 8 clips, untuned defaults

`experiments/2026-09-16_bytetrack-floor/` (`comparison.csv`, `findings.md`).
Scored with **clip-scoped matching**; see the correction note below.

| judge | P | R | F1 | tp/fp/fn | `pass_by` fired |
|---|---|---|---|---|---|
| geometric (control) | 0.178 | **0.722** | 0.286 | 13/60/5 | 5 of 7 |
| **VLM (shipped)** | **0.400** | 0.667 | **0.500** | 12/18/6 | **1 of 7** |

The VLM more than doubles precision, cuts false positives 60 -> 18 and `pass_by`
false fires 5 -> 1, paying one true positive. That is the hybrid design's whole
premise and it holds.

`1THkHYIQ_bY_0` — the clip whose 8.8 s event was previously unfindable — now
scores **P=1.00 R=1.00** under the VLM. `gt1125_06` carries 11 of the 18
remaining false positives; it is the 4K aerial clip with ~25 vehicles and ~8
people per frame, so its pair count dwarfs the rest.

**The fragmentation fix is a win only via the VLM.** On the control arm it
lowers F1 (0.314 -> 0.286): recall rises but precision falls further. On the
shipped arm it raises F1 (0.476 -> 0.500), because the extra false positives are
the kind the VLM rejects well. The control arm therefore prefers a *different*
proposer configuration than the shipped system — worth remembering before
tuning against it.

### Correction: pooled numbers were inflated ~55%

`full_report` pooled all predictions and GT into one flat matching, but frame
spans are clip-local, so a prediction from one clip could satisfy another's
event. On the real set: 17 pooled true positives where per-clip counts summed to
11. Fixed (`metrics.report_groups`), with a test asserting the old behaviour
would have credited the cross-clip match. **Per-clip numbers were always
correct; only pooled rows were wrong.** All earlier experiment directories carry
a correction banner and have been rescored from their stored outputs.

## Next steps

1. **Run the VLM arm** -- the only thing standing between here and the headline
   ablation. Blocked on bandwidth, not code: Qwen2.5-VL-7B is a 16 GB download
   and three of five shards were still missing at hand-off. Check with
   `python -c "from pvi.judge.vlm import check_weights_available as c; c('Qwen/Qwen2.5-VL-7B-Instruct','cc594898137f460bfe9f0759e9844b3ce807cfb5')"`,
   resume with `snapshot_download(..., revision=...)`, then
   `tools/run_all.py --judges geometric,vlm`.
   Note each interrupted attempt starts a NEW `.incomplete` blob rather than
   resuming the old one, so avoid killing it.
2. **Chase span fragmentation** (see the baseline table above). It is the single
   remaining recall miss and the likely cause of several false positives, since
   one long event becomes four short wrong ones.
3. **LOCO threshold selection — ready to run, not yet run.**
   `python -m pvi.evaluate.loco --judge vlm`. Searches **three** knobs
   (`det_conf`, `tau_near`, `tau_far`); `vlm_conf_thresh` was dropped from the
   agreed four because it never binds. 45 valid settings, but only **24
   tracking passes** — the sweep groups by `det_conf`, the one searched knob
   that affects tracking, and reuses one `ClipTracks` across every `tau`
   combination. Expect a few hours, dominated by `gt1125_06`.
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

## Settled: `Videos/` is not shipped

**Decided by the user 2026-09-16: keep the clips git-ignored.** They are the
task-setter's footage and the repo is public. Do not add them, and do not
"helpfully" un-ignore them when reproduction fails — a reviewer places the 8
`.mp4` files in `Videos/` themselves (README, "Getting the data").

What ships instead, so the results stay inspectable without the clips:
`outputs/`, `experiments/*/` (including annotated probe frames),
`data/ground_truth.json`, and `data/vlm_cache/`.

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

- **`generate()` has never run.** Everything around it is verified: the prompt
  builder, the response parser and the cache are unit-tested, and the multi-image
  message plumbing is checked against the real Qwen processor
  (`tests/test_vlm_plumbing.py`) -- one image placeholder per image, correct
  `pixel_values`/`image_grid_thw`. But no forward pass has happened, so treat the
  VLM arm as unproven.
- ~~Model revisions~~ **all three are now pinned** in `config/default.yaml`.
  A null revision resolves "latest" against the Hub: unreproducible, and it hung
  this session for half an hour on a slow connection.
- **All 8 clips have a geometric number; none has a VLM number.** The
  comparison the ablation exists for is half-done.
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
