# Configuration Schema

Consumed by `pvi.config`. The **fixed / tunable** split below is the contract
that `experiment-runner` and `hyperparameter-sweep` read directly — do not
re-derive ranges at sweep time.

Every run writes its fully-resolved config (defaults merged with overrides, model
revisions resolved to hashes) next to its output, per the determinism requirement
in `docs/problem-definition.md` §4.

## Fixed config

Never varies across experiment runs. Changing any of these invalidates
comparability between runs.

| key | default | notes |
|---|---|---|
| `seed` | 0 | seeds python/numpy/torch; also forces deterministic kernels |
| `paths.videos` | `Videos/` | |
| `paths.frames` | `data/frames/` | **labeling/contact-sheet use only — downscaled to max width 1600.** `gt1125_06` is 1600x900 here, not 3840x2160. The pipeline decodes from `paths.videos` at native resolution and never reads this. |
| `paths.ground_truth` | `data/ground_truth.json` | frozen; see labeling protocol |
| `paths.outputs` | `outputs/` | |
| `paths.vlm_cache` | `data/vlm_cache/` | committed; enables GPU-free reproduction |
| `device` | `cuda:0` | |
| `detector.model_id` | `Roboflow/rf-detr-large` | Apache-2.0; RF-DETR-L. Fallback `ustc-community/dfine-xlarge-coco` |
| `detector.revision` | *(pinned hash)* | resolved at first run, then frozen |
| `detector.classes` | `[person, bicycle, car, motorcycle, bus, truck]` | COCO ids |
| `vlm.model_id` | `Qwen/Qwen2.5-VL-7B-Instruct` | Apache-2.0 |
| `vlm.revision` | *(pinned hash)* | |
| `vlm.max_new_tokens` | 512 | |
| `vlm.temperature` | 0.0 | greedy; non-negotiable for determinism |
| `vlm.batch_size` | 1 | batching perturbs outputs even at t=0 |
| `tracker.impl` | `roboflow-trackers` | Apache-2.0; Ultralytics/BoxMOT are AGPL |
| `tracker.name` | `bytetrack` | |
| `tracker.gmc` | `orb` | hand-rolled ORB+RANSAC; no upstream Python GMC exists |
| `openvocab.model_id` | `IDEA-Research/grounding-dino-base` | Apache-2.0; probe-only until design-plan §6.7 resolves |
| `openvocab.box_threshold` | 0.27 | literature-suggested starting point |
| `openvocab.text_threshold` | 0.25 | prompts must be lowercase, dot-terminated |

## Tunable hyperparameters

Searched by LOCO threshold selection and any later sweep.

| name | type | default | range | meaning |
|---|---|---|---|---|
| `det_conf` | float | 0.35 | [0.10, 0.70] | detection confidence floor |
| `tau_near` | float | 0.15 | [0.05, 0.40] | contact-enter, in vehicle-diagonal units |
| `tau_far` | float | 0.30 | [0.10, 0.80] | contact-leave (hysteresis) |
| `min_dwell_s` | float | 0.5 | [0.3, 0.7] | R1 dwell trigger, seconds |
| `smooth_window_s` | float | 0.4 | [0.0, 0.8] | feature smoothing, seconds |
| `door_delta_thresh` | float | 0.12 | [0.04, 0.40] | R4 vehicle-region change |
| `context_pad_s` | float | 1.0 | [0.0, 3.0] | span padding for the VLM |
| `n_vlm_frames` | int | 12 | [4, 24] | frames sampled per candidate |
| `crop_margin` | float | 0.25 | [0.0, 0.6] | box dilation before crop |
| `vlm_conf_thresh` | float | 0.5 | [0.0, 0.9] | verdict acceptance floor |

### Constraints

Two are structural, not preferences. A sweep that violates them produces
meaningless configurations, so they are validated at config load:

- **`tau_far > tau_near`** — otherwise the hysteresis degenerates and a person
  hovering near the threshold yields fragmented spans instead of one.
- **`min_dwell_s < 0.75`** — the shortest positive event in the GT is 0.75 s, so
  a larger value makes R1 structurally unable to fire on it.

Additionally `smooth_window_s` is floored at 3 frames per clip at runtime
(`max(smooth_window_s * fps, 3)`), because at 6 fps a 0.4 s window is 2.4 frames.

### Defaults are guesses

The defaults above are first estimates from the GT's geometry, **not** tuned
values. Reported numbers come from the LOCO-selected thresholds. Anyone reading
`config/default.yaml` should not mistake it for a result.
