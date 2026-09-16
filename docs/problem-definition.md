# Problem Definition — Person-Vehicle Interaction Detection

Status: v3, 2026-09-14. Clips inspected; ground truth labeled, audited and frozen
(25 events). Section 3 reflects the realized label counts.

## 1. Problem statement

Given a short, silent MP4 clip, emit a machine-readable list of the person-vehicle
*interactions* it contains. An interaction is a purposeful physical engagement
between a person and a vehicle (entering, exiting, opening a door, loading cargo,
attending the vehicle); mere co-presence or passing by is not. Each reported
interaction must carry a clip id, a frame/time span, a description of the person(s)
involved, and a description of the vehicle involved. This is a take-home exercise:
the deliverable is a clear, reproducible, well-argued pipeline plus a short
write-up, not a production system. Scope is deliberately bounded — clips are
independent and no cross-clip identity is maintained.

## 2. Data inventory

- **Location:** `Videos/` in the repo root. **8 clips, 89 MB total.** (The supplied
  zip contained a duplicated nested `Videos/Videos/`; verified byte-identical with
  `cmp` and removed.)
- **Labels:** none provided.
- **Total volume: 2,313 frames across 8 clips** — small enough that a full
  hand-labeled ground truth is roughly an hour's work. Plan to produce
  `data/ground_truth.json` (schema in section 3).

### Clip inventory

| clip_id | resolution | fps | dur (s) | frames | scene |
|---|---|---|---|---|---|
| `1THkHYIQ_bY_0` | 1280x720 | 30 | 16.5 | 495 | Night, residential driveway, consumer security cam (eufy overlay). Person at a dark sedan, leaning in, then departing with an object. Heavy sensor noise, blown highlights. |
| `gt1125_06` | 3840x2160 | 29.97 | 20.0 | 600 | **Aerial/drone, moving camera.** Street + parking strip beside a football pitch. Persons **~110 px tall at native 4K** (measured 2026-09-16; see the correction note below). A box truck is present. |
| `HIu4lM4B8hA_1` | 352x288 | 25 | 10.0 | 250 | **Grayscale CIF CCTV.** Person opens the driver door of a white hatchback, leans in, withdraws, walks off. Lowest resolution in the set. |
| `iMGR_0AG3a8_2_3` | 960x720 | 6.67 | 27.3 | 182 | Multi-storey car park at night. White SUV with open tailgate + person; a dark sedan **drives in** under headlights; several pedestrians. Multi-actor, multi-vehicle. |
| `mKzCQKTHizw_0` | 640x360 | 29.97 | 10.0 | 300 | Parking lot, daytime. **Two people running (a chase).** They pass parked cars without engaging. Primarily a negative/distractor clip. |
| `mKzCQKTHizw_1` | 640x360 | 29.97 | 9.2 | 276 | Same camera as above. Person in dark clothing opens the door of a silver sedan and leans in. |
| `NmlzoaDcOuI_1` | 1280x720 | **6** | 18.0 | 108 | Daytime kerbside. Person **exits** a silver sedan; later someone engages the door again. Pedestrians pass in the background. |
| `NmlzoaDcOuI_6` | 1280x720 | **6** | 17.0 | 102 | Same camera. Man in a green shirt opens the passenger door of a red sedan, leans in (load/unload), closes it, walks away; the car later drives off. Pedestrians pass close by. |

### Correction: `gt1125_06` person scale (2026-09-16)

Earlier versions of this table said persons in `gt1125_06` are **~10 px tall**.
That is wrong by roughly an order of magnitude. Measured against the two GT
person anchors (frames 222 and 239) on natively-decoded 3840x2160 frames, a
standing adult spans **~110 px** head to foot. The 10 px figure was eyeballed
from a downscaled contact sheet, where a 6x reduction does put a person at
around that size.

What the correction changes, and what it does not:

- **`gt1125_06` is a moving-camera problem, not a small-object problem.** The
  recall-stress framing was built on the wrong number. Ego-motion and the
  top-down viewpoint remain real; object scale at native resolution does not.
- **Tiling is still worth probing, for a different reason.** RF-DETR-L resizes
  its input to 704 px. At 3840 wide that is a 5.45x reduction, so a 110 px
  person reaches the network at ~20 px — genuinely small, though inside the
  regime COCO AP_S describes rather than far below it. A 704 px tile shows the
  same person at ~110 px. So the expected gain is a real one, and the
  design-plan s6.2 "recall near zero, drop the clip" branch is now the unlikely
  outcome rather than the expected one.
- **The viewpoint gap is untouched by tiling.** COCO is overwhelmingly
  ground-level imagery; this is an oblique overhead view. If detection fails
  here it will most likely be for that reason, which tiling does not address.

The probe still runs, and still decides. Only the prior changes.

### Correction: camera motion, 3 clips not 1 (2026-09-16)

Earlier versions said "the camera is static in 7 of 8 clips; `gt1125_06` is the
sole moving-camera case". Measured with ORB+RANSAC over a one-second baseline
(`experiments/2026-09-16_camera-motion/`) and confirmed by eye:

| clip | drift per second (frac. of width) | verdict |
|---|---|---|
| `mKzCQKTHizw_0` | 0.12242 | **moving** — camera pans to follow the runner |
| `mKzCQKTHizw_1` | 0.00883 | **moving** — camera oscillates |
| `gt1125_06` | 0.00428 | **moving** — drone drift |
| `HIu4lM4B8hA_1` | 0.00131 | static |
| `NmlzoaDcOuI_1` | 0.00038 | static |
| `NmlzoaDcOuI_6` | 0.00024 | static |
| `1THkHYIQ_bY_0` | 0.00009 | static |
| `iMGR_0AG3a8_2_3` | 0.00006 | static |

**Measuring this correctly required a one-second baseline, not consecutive
frames.** At a one-frame baseline `gt1125_06` reads 0.00028 — *less* than the
static `mKzCQKTHizw_*` clips' 0.0013 of ORB estimation jitter, so the drone
looked more stationary than a tripod. Drift accumulates over a second; zero-mean
jitter does not. This is a measurement-design point, not a detail: the first
version of the check confidently labelled the drone clip static.

Consequences:

- **R4's `door_delta` is dead on 3 clips, not 1.** It needs a static camera and
  a parked vehicle. `mKzCQKTHizw_1` contains a door-opening event — precisely
  what R4 targets — and R4 cannot fire there.
- **This raises the stakes on the open-vocabulary door cue** (design-plan §6.7).
  A text-prompted detector is camera-motion-independent, so the branch it would
  delete is now failing on three clips rather than one.
- **Motion compensation runs on 3 clips.** It is enabled per clip from this
  measurement rather than always — see design-plan §6c.3 for why "always" is the
  wrong default.
- **The precision stress case is also a moving-camera case.** `mKzCQKTHizw_0`
  was already the hardest clip for false positives; it is harder than recorded.

### What the inventory establishes

- **Extreme heterogeneity is the defining property of this set.** Resolution spans
  352x288 to 3840x2160 (a 90x pixel-count range) and frame rate spans 6 to 30 fps.
  Two direct consequences: every spatial threshold must be **normalized by object
  size** (not pixels), and every temporal threshold must be expressed **in seconds**
  (not frames). A pixel- or frame-based rule tuned on one clip will be meaningless
  on another.
- **The camera moves in 3 of 8 clips — not 1.** *(Corrected 2026-09-16; see the
  camera-motion correction below.)* `gt1125_06` drifts (drone), and **both
  `mKzCQKTHizw_*` clips pan**: the camera in `mKzCQKTHizw_0` actively follows the
  running woman, and `mKzCQKTHizw_1` oscillates (the red bollards traverse the
  frame). A static-camera prior is therefore much weaker than first assumed, and
  the two clips it fails on include the precision stress case and a door event.
- **Distractors are present by design.** `mKzCQKTHizw_0` is a chase past parked
  cars; both `NmlzoaDcOuI_*` clips have pedestrians walking within a metre of the
  subject vehicle. The set is deliberately constructed to punish a naive
  proximity-threshold detector. This confirms the precision-over-recall stance in
  section 3.
- **The recurring positive pattern is door-centric:** approach -> open door ->
  lean in / enter / exit -> close door -> depart. An **open car door** is a strong,
  visually distinctive cue that appears in 5 of the 8 clips.
- **Two clips are hard cases in opposite directions.** `gt1125_06` (ego-motion,
  oblique overhead viewpoint) stresses recall; `mKzCQKTHizw_0` (running
  pedestrians near cars) stresses precision. Report both separately rather than
  only in aggregate.
- **Night / low-light / grayscale / heavy compression** all appear. Detector
  robustness matters more than detector peak accuracy.
- Clip ids look like YouTube video ids plus a segment index, and the
  `NmlzoaDcOuI_*` pair strongly resembles VIRAT ground-camera footage. *Hypothesis
  only, worth ~10 minutes to check:* if any clip traces to a public benchmark with
  released event annotations, that is a legitimate, documentable GT cross-check.
  Do not build the pipeline around it.

## 3. Candidate metrics

No ground truth exists yet, so the honest position is: **evaluation is subjective
until `data/ground_truth.json` is hand-labeled.** Once it is, use the following.

**Primary — event-level detection, per interaction type:**
- Match a predicted interaction to a GT interaction when they share the same
  person-vehicle pair and their frame spans overlap with temporal IoU ≥ 0.3.
  The threshold is deliberately loose because the GT boundaries are themselves
  fuzzy; report tIoU ≥ 0.5 as a secondary, stricter number.
- Report precision, recall, and F1 over matched events.

**Per-clip reporting is mandatory, not just the aggregate.** With n=8, a single
mean hides everything. In particular, report `mKzCQKTHizw_0` (precision stress
test: running pedestrians near parked cars, few or no true interactions) and
`gt1125_06` (recall stress test: moving camera, oblique overhead viewpoint,
~20 px persons after the detector's internal resize) as named cases.

**Ground truth — realized.** `data/ground_truth.json`, 25 events over all 8 clips,
produced with `tools/label_gt.py` under `docs/labeling-protocol.md`, audited by
`tools/validate_gt.py`. **Frozen as of 2026-09-14.**

| type | n |
|---|---|
| `exit_vehicle` | 7 |
| `enter_vehicle` | 6 |
| `attend_vehicle` | 2 |
| `open_close_door` | 2 |
| `load_unload` | 1 |
| **positives** | **18** |
| `pass_by` (explicit negatives) | 7 |

Positive spans: min 0.75 s, median 4.50 s, max 11.40 s.

**Reporting tiers are forced by these counts.** With n=1 for `load_unload`, a
per-type recall for it is 0% or 100% and means nothing. Therefore:

- **Tier 1 — report as headline numbers:** aggregate detection P/R/F1 over all 18
  positives, and the binary interaction-vs-`pass_by` discrimination over all 25.
- **Tier 2 — report per-type, with counts stated inline:** `enter_vehicle` (n=6)
  and `exit_vehicle` (n=7) only.
- **Tier 3 — report as a confusion matrix only, never as a rate:**
  `attend_vehicle`, `open_close_door`, `load_unload` (n=2,2,1).

Quoting a per-type F1 for a class with one or two instances would be the single
easiest way to make this evaluation look more rigorous than it is. Don't.

**Secondary:**
- **Type accuracy** on matched events (confusion matrix over the closed
  vocabulary — does the system say `exit_vehicle` where GT says `enter_vehicle`?).
- **Temporal localization error:** median |Δ| in frames on start and end boundaries.
- **Description quality:** qualitative. Judged by reading, not scored. State this
  plainly rather than inventing a captioning metric on a handful of clips.

**Relative cost of error types.** For this task a **false positive is worse than a
false negative.** The task explicitly contrasts "entering/exiting" against "passing
by", which signals that the interesting failure mode — and the thing being probed —
is over-reporting incidental proximity as interaction. A pipeline that reports every
person who walks near a car has not solved the problem. Tune the operating point
toward precision, and report the recall given up.

Note this is the *opposite* of the usual surveillance/defense bias toward recall,
and it is a property of this exercise, not a general stance.

## 4. Practical constraints

- **Deployment:** offline batch. One clip in, one structured record set out. No
  real-time, no streaming.
- **Runtime budget:** soft. Target a few minutes per clip on the available GPU.
  Nothing in the task rewards speed; clarity wins over throughput.
- **Hardware:** NVIDIA RTX A6000, 48 GB VRAM; 24 CPU cores. Comfortably fits a
  mid-size open-weights video VLM alongside a detector — no quantization needed,
  no cloud inference required.
- **Environment:** system Python is 3.14.5, which PyTorch does not yet ship wheels
  for. The project needs its own **Python 3.11 or 3.12 venv**; this is a hard setup
  constraint, not a preference.
- **Determinism** (explicitly required by the task): pin every model checkpoint by
  revision hash, fix all seeds, use greedy decoding (temperature 0) for any VLM,
  fix the frame-sampling schedule, and record the resolved config alongside every
  output. Two runs on the same clip must produce byte-identical results.
- **External services:** prefer none. All models local and open-weights, so the
  repo is self-contained and reproducible by the reviewer. If a hosted API is used
  for any component, it must be documented and the raw responses cached in-repo so
  results remain reproducible without credentials.

## 5. Track decision

**Track A — exploratory, subjective/light-quantitative evaluation.**

Rationale: no labels ship with the data, the clip count is on the order of a
handful, and nothing is being trained. There are no train/val/test splits to
define because there is no fitting step — every component is zero-shot. The
hand-labeled GT of section 2 upgrades evaluation from "watch it and judge" to a
small quantitative check, but with a sample this size the numbers are indicative,
not statistically meaningful, and must be reported as such. Any threshold tuned
against that GT is tuned on the evaluation set; say so in the write-up rather than
presenting the result as held-out.

## 6. Implication for research

Ruled **out**:
- Supervised training or fine-tuning of any kind — no labels, no data volume.
- Any method needing a class-balanced interaction dataset (action-recognition
  heads trained on enter/exit classes, unless a suitable pre-trained checkpoint
  exists off-the-shelf).
- Metric claims with confidence intervals or significance tests — the sample
  cannot support them.

Ruled **in**:
- Off-the-shelf zero-shot detection + multi-object tracking as the spatial-temporal
  backbone (pinned checkpoints).
- Geometric/temporal reasoning over tracks to *propose* candidate interactions —
  this is where the classical work lives and where determinism comes from.
- A zero-shot video VLM as the *semantic adjudicator* over proposals, supplying
  both the closed-vocabulary type and the free-text/appearance descriptions.
- Hand-labeled GT on a handful of clips as the evaluation substrate.

Chosen architecture (settled with the user before research): **hybrid —
tracker proposes, VLM judges**, emitting a **closed-set interaction type plus a
free-text note**, with **appearance attributes + track ID** for the person and
vehicle description fields.

### Draft interaction vocabulary

Derived from viewing the clips; to be confirmed during the labeling pass.

| type | definition |
|---|---|
| `enter_vehicle` | Person transitions from outside to inside; track terminates at the vehicle. |
| `exit_vehicle` | Person transitions from inside to outside; track originates at the vehicle. |
| `open_close_door` | Door manipulated without a full entry/exit transition. |
| `load_unload` | Object transferred into or out of the vehicle (incl. boot/tailgate). |
| `attend_vehicle` | Sustained physical engagement otherwise uncategorized — leaning on, inspecting, cleaning, reaching through a window. |

Rejected proposals are recorded as `pass_by` in the **debug** artifact only, never
in the deliverable. Keeping them is what makes the precision failure mode
inspectable.

The vocabulary is deliberately **physical and observational, not intentional** —
see ambiguity A6.

## Open items

1. Hand-label `data/ground_truth.json` for all 8 clips (est. ~1 h).
2. Finalize the interaction vocabulary against the labeling pass — the draft in
   section 6 is derived from viewing contact sheets, not frame-by-frame review.

## Ambiguity register

The task requires that ambiguities be resolved explicitly, with alternatives
considered and documented. Decisions taken so far:

| # | Ambiguity | Decision | Alternatives considered |
|---|---|---|---|
| A1 | What counts as an "interaction"? | Closed vocabulary of interaction types + a free-text note per event. Vocabulary to be finalized after viewing the clips. | (a) Binary interacting/not + free text — flexible but unevaluable by type. (b) Pure closed set, no free text — loses the unexpected cases. |
| A2 | Output granularity: one record per (person, vehicle, event) or per (person, vehicle) pair? | One record per event, so a person who enters then later exits yields two records. Matches the required "time span(s)" field. | A single record per pair with a list of spans; rejected as it complicates matching and description. |
| A3 | What does "description of the person(s)" mean? | Appearance attributes (clothing, carried objects) plus the intra-clip track ID. | Track ID + geometry only — defensible but a thin reading of the spec. |
| A4 | Multi-person events (two people enter the same car) | Emit one record per person; link them with a shared `event_group_id`. | A single record with a person list; rejected — makes per-person descriptions awkward and breaks 1:1 evaluation matching. |
| A5 | A person who is inside the vehicle for the whole clip and never visible | Not reported. Only observable interactions are emitted. | Inferring occupancy from vehicle motion; rejected as unobservable and speculative. |

| A6 | Several clips (`1THkHYIQ_bY_0`, `mKzCQKTHizw_1`, `HIu4lM4B8hA_1`) look like break-ins or thefts. Should the type encode intent? | No. The vocabulary stays physical/observational (`open_close_door`, `load_unload`). Any intent reading goes in the free-text `note` only. | An `unauthorized_entry`/`theft` type; rejected — intent is not reliably observable from silent low-res footage, and a wrong criminal label is far more costly than a vague one. |
| A7 | A driver visible inside a moving vehicle (`iMGR_0AG3a8_2_3`) | Not an interaction. Driving is a continuous state, not an event. Only an observable enter/exit *transition* is reported. | Reporting an `occupies_vehicle` span for the clip duration; rejected as unbounded and uninformative. |
| A8 | Very low frame rate (6 fps) in two clips vs. 30 fps in others | All temporal thresholds are defined in **seconds**; all spatial thresholds normalized by the vehicle bbox diagonal. Frame indices appear in the output but never in the rule logic. | Per-clip hand-tuned thresholds; rejected — unreproducible and would not generalize to a ninth clip. |

Open ambiguities, to resolve during the labeling pass: whether a driver visible
through a windscreen counts as an interaction; whether a person touching/leaning
on a parked car without opening it counts; the minimum duration for an interaction
to be reported.
