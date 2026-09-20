# PersonCarInteraction — Person-Vehicle Interaction Detection

Given a short silent MP4 clip, emit a machine-readable list of the person-vehicle
interactions it contains — entering, exiting, opening a door, loading cargo,
attending a vehicle — while *not* reporting people who merely walk past.

The last part is the hard part, and it is what the supplied clip set is built to
probe: 7 of the 25 hand-labeled ground-truth events are deliberate `pass_by`
near-misses.

## Approach

**Hybrid: the tracker proposes, a VLM judges.**

```
Videos/*.mp4
  -> decode            native resolution, linear, no seeking
  -> detect            RF-DETR-L, person + COCO vehicle classes
  -> track             ByteTrack-style MOT + ORB/RANSAC motion compensation
  -> propose.features  per (person, vehicle) pair: d_norm(t), speed(t), door_delta(t)
  -> propose.rules     4 recall-oriented rules -> candidate spans
  -> judge             Qwen2.5-VL adjudicates: type, descriptions, accept/reject
  -> outputs/<clip>.json
```

The split exists because the two halves of the problem have different characters.
*"Was this person in sustained contact with this vehicle between t1 and t2"* is a
geometry question with a deterministic answer. *"Was that contact an interaction
or did they just walk past"* is a semantic question that geometry answers badly.
Neither half is load-bearing alone, and `pvi/judge/geometric.py` is the ablation
arm that measures exactly how much the VLM adds.

Nothing is trained. Every learned component is off-the-shelf and pinned by
revision hash.

## Two invariants

The clip set spans **352×288 to 3840×2160** and **6 to 30 fps**. So:

- **No rule compares raw pixels.** Spatial quantities are normalized by the
  vehicle box diagonal (`d_norm`) or by person height (speed, in body-lengths
  per second).
- **No rule counts frames.** Every threshold is declared in seconds and converted
  per clip. The binding constraint is the shortest ground-truth positive, 0.75 s
  — which at 6.67 fps is 5 frames.

A threshold tuned on one clip would otherwise mean something different on the
next one.

## Getting the data

**The source clips are not in this repository, deliberately.** They are the
task-setter's footage; publishing someone else's data is not ours to do. Place
the 8 `.mp4` files in `Videos/` and everything below works unchanged — nothing
is hard-coded to a clip beyond the ground-truth ids.

What *is* committed, so the results stay inspectable without the clips:

| | |
|---|---|
| `outputs/` and `experiments/*/` | every reported number, plus per-clip debug artifacts |
| `data/ground_truth.json` | the 25 hand-labeled events |
| `data/vlm_cache/` | every VLM response, keyed by prompt + quantized frame/box geometry |
| `experiments/*/frames/` | annotated frames from the probes |

Re-running the pipeline needs the clips. Re-reading what it concluded does not.

## Setup

System Python is 3.14, which has no torch wheels. The project needs its own
3.11/3.12 environment.

```bash
uv venv --python /usr/bin/python3.12 .venv
VIRTUAL_ENV=.venv uv pip install -r requirements.txt
```

Requires `ffmpeg` and `ffprobe` on PATH (ffmpeg 9+; the code uses
`-fps_mode passthrough`, as `-vsync` was removed).

## Run

```bash
.venv/bin/python -m pvi.cli --clip Videos/NmlzoaDcOuI_6.mp4 --config config/default.yaml
```

Produces:

| file | contents |
|---|---|
| `outputs/<clip_id>.json` | one record per (person, vehicle, event) |
| `outputs/<clip_id>.debug.json` | rejected candidates and their features |
| `outputs/<clip_id>.config.yaml` | the fully resolved config, including model revisions |

Once every clip has been run, pool the per-clip files into the single artifact
the task asks for:

```bash
python3 tools/pool_outputs.py          # stdlib only; no venv needed
```

| file | contents |
|---|---|
| `outputs/interactions.json` | **the deliverable**: every interaction in the clip set, `clip_id` on each record |

Pooling recomputes nothing. It refuses to run if a clip is missing an output, or
if the per-clip files carry different `config_hash` values — a list merged from
two configs is not one result, and pooling them would erase the only evidence of
that.

Evaluate against the frozen ground truth:

```bash
.venv/bin/python -m pvi.evaluate.run
```

It scores whatever is in `outputs/`, so it takes no `--config`: the config that
produced those files is recorded in `outputs/<clip_id>.config.yaml`.

Tests (the numerical pieces — geometry, hysteresis, matching, GMC, config
constraints):

```bash
.venv/bin/python -m pytest tests -q
```

## Output format

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
    "person":  {"track_id": 2, "description": "adult male, green shirt, black trousers, carrying papers"},
    "vehicle": {"track_id": 1, "class": "car", "description": "red sedan, parked, kerbside left"},
    "note": "Opens the passenger door, leans in, closes it and walks away.",
    "confidence": 0.86,
    "evidence": {"rule": "R3_birth_near", "min_d_norm": 0.04, "dwell_s": 6.3}
  }]
}
```

`evidence` is not required by the task. It is there because the first question
anyone asks about a reported interaction is *"why did it think that"*, and the
answer should not require rerunning the pipeline.

`outputs/interactions.json` is the same records concatenated across clips, with
`clip_id` lifted out of the file header and onto each one, since the task asks
for it per interaction:

```json
{
  "artifact": "person_vehicle_interactions",
  "config_hash": "sha256:...",
  "n_clips": 8,
  "n_interactions": 18,
  "clips": {"NmlzoaDcOuI_6": {"fps": 6.0, "n_frames": 102, "width": 1280, "height": 720}},
  "interactions": [{"clip_id": "NmlzoaDcOuI_6", "interaction_id": "NmlzoaDcOuI_6__i001", "...": "..."}]
}
```

## Determinism

Required by the task and treated as a hard constraint:

- All model checkpoints pinned by **commit hash**, resolved on first run and
  recorded in the output (a branch name pins nothing).
- Greedy decoding, VLM batch size 1 — batching perturbs logits enough to flip a
  borderline token even at temperature 0.
- Fixed seeds; float32 detection rather than fp16, since accumulation order can
  move borderline boxes across the confidence threshold.
- Fixed frame-sampling schedule; no seeking during decode.
- Every VLM response cached under `data/vlm_cache/`, keyed by the prompt text
  plus the frame indices and integer-quantized box coordinates the images are
  drawn from, and **committed to the repo** — so results reproduce with no GPU,
  and a cache miss is visible rather than silently re-rolled. Keying on the
  rendered bytes, as this did originally, meant a sub-pixel box shift from a
  different torch build invalidated the entry and silently re-ran the VLM.
- That matters more than the seed: re-running the set under a rebuilt torch
  reproduced every proposal span exactly but flipped the judge on 4 candidates.
  Tracking and proposal are deterministic; the VLM is reproducible only through
  the cache.
- The fully resolved config is written next to every output.

## Evaluation

Ground truth: `data/ground_truth.json` — 25 hand-labeled events over all 8 clips,
**frozen**. Conventions in `docs/labeling-protocol.md`.

| type | n |
|---|---|
| `exit_vehicle` | 7 |
| `enter_vehicle` | 6 |
| `attend_vehicle` | 2 |
| `open_close_door` | 2 |
| `load_unload` | 1 |
| **positives** | **18** |
| `pass_by` (explicit negatives) | 7 |

**The counts force the reporting, and the reporting says so.** With n=1 for
`load_unload`, a per-type recall is 0% or 100% and means nothing. So:

- **Tier 1, headline:** aggregate detection P/R/F1 over the 18 positives, plus
  the binary interaction-vs-`pass_by` discrimination over all 25.
- **Tier 2, per-type rates with counts inline:** `enter_vehicle` and
  `exit_vehicle` only.
- **Tier 3, confusion-matrix entries only, never a rate:** `attend_vehicle`,
  `open_close_door`, `load_unload`.

Matching is by temporal IoU ≥ 0.3 (loose, because the labeled boundaries are
themselves ±2 frames), with tIoU ≥ 0.5 reported as a stricter secondary number.
Every metric is reported twice — over all events, and over confident events only
— because the gap states how much residual error is definitional rather than
algorithmic.

**Thresholds are selected leave-one-clip-out** and that is the headline number;
the all-data-tuned number is reported alongside, labelled not-held-out. The gap
between them measures how much 8 clips are being overfitted.

With 18 positives a single event is 5.6% of aggregate recall. Every number is
quoted with its denominator, and none of it supports a confidence interval.

## Repository layout

```
pvi/            the package
  config.py     config dataclasses, YAML load, validated constraints
  schema.py     the JSON contract; GT loading
  video.py      deterministic native-resolution decode
  geometry.py   normalized box geometry
  detect/       Detector ABC, HF detector, tiled wrapper, open-vocab probe
  track/        tracker wrapper, ORB+RANSAC global motion compensation
  propose/      pair features, span primitives, the four rules
  judge/        Judge ABC, VLM adjudicator, geometric ablation arm, prompts
  evaluate/     temporal matching, tiered metrics
docs/           problem definition, design plan, labeling protocol
config/         schema.md (fixed vs tunable) and default.yaml
tools/          frame extraction, GT labeler, GT validator, probes
experiments/    one directory per run: config.yaml + results.json + figures
tests/          numerical correctness
```

## Documentation

| doc | what it settles |
|---|---|
| `docs/problem-definition.md` | scope, data inventory, metrics, the ambiguity register |
| `docs/design-plan.md` | architecture, component contracts, resolved decisions |
| `docs/labeling-protocol.md` | GT conventions and the audit log |
| `config/schema.md` | fixed vs tunable configuration split |

## Known limitations

- **8 clips, 18 positives.** Numbers are indicative, not statistically
  meaningful, and are reported as such.
- **Thresholds are tuned against the evaluation set.** LOCO mitigates this; it
  does not eliminate it, because the clips are not independent draws from
  anything.
- **`attend_vehicle` vs `pass_by`** is the expected precision failure. At n=2
  the boundary cannot be tuned and has to come from the prompt's type
  definitions.
- **`gt1125_06`** is the sole moving-camera clip, shot from an oblique overhead
  viewpoint that COCO-trained detectors see little of. Motion compensation
  addresses the ego-motion; nothing here addresses the viewpoint.
- **No cross-clip identity.** Clips are independent by design.
