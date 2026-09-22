# Result Analysis: Person-Vehicle Interaction Detector

## 1. Problem and Approach
Given an mp4 clip (with no audio), produce a list of human-vehicle interactions in the clip. The challenge lies in the definition of interaction; a person passing a vehicle does not constitute an interaction. We created the following pipeline: detect humans and vehicles, track them over time, propose interactions, then judge (classify) them. Each accepted interaction then gets a second VLM call that describes the person and the vehicle.

The split separates geometrically similar events by semantics: contact is a geometry question, interaction-vs-pass-by is not.

Nothing is trained: given the limited data, every component is pretrained and off-the-shelf. Hyperparameters are tuned on the provided clips against hand-made labels.

Determinism is treated as a hard requirement: model revisions are pinned by commit hash, seeds are fixed, decoding is greedy at batch size 1, the fully resolved config is written next to every output, and every VLM response is cached and committed, so a re-run reuses the exact adjudications rather than silently re-rolling them.

## 2. Design Decisions

- **Normalization invariants**: 
    - Proximity distances are measured relative to vehicle-diagonal length, not pixels, due to different image resolutions
    - Time is measured in seconds, not frames, due to different frame-rates
- **Open-vocabulary door cue, not pixel-change heuristic**: Door open/close, a semantic cue for interaction, is detected by Grounding DINO rather than a pixel-change heuristic, because the heuristic failed on three moving-camera clips while the open-vocab detector fails only on the single aerial clip.
- **License-driven model selection**: Because this repo is public, we chose models with Apache-2.0 licenses throughout. This ruled out Ultralytics, BoxMOT, DEIMv2, Co-DETR, and other models.
- **Ambiguity of Interaction**: Decisions can be seen in problem-definition.md
- **Descriptions as fixed-vocabulary slots, in a separate VLM call**: The person (sex, age, upper/lower clothing colour, carrying yes/no) and vehicle (colour, body type, parked/stopped/moving) are returned as slots with `unknown` always allowed, and the description text is built from them. A separate call keeps description changes from re-rolling the cached interaction verdicts. Carrying is yes/no because, as free text, the model named objects that were not there, including a "gun".

## 3. Evaluation Methodology
We hand-labeled 25 events before development: 18 positives and 7 `pass_by` events. Event boundaries are contact-based. The GT was frozen before development began to prevent drift of the label to what we observed.

The primary measure of success is Temporal Intersection-over-Union (tIoU), where a score of 0.3 is considered a match. We deliberately choose a loose threshold because the bounds of an 'interaction' are inherently fuzzy. We also report the score when the threshold is set to 0.5 to determine the quality of localization.

We describe events as one of the following `pass_by` (false), `enter_vehicle`, `exit_vehicle`, `attend_vehicle`, `open_close_door`, `load_unload`. Reporting tiers are as follows:

| tier | classes | # of events | what may be reported|
|---|---|---|---|
| 1 | all positives pooled, binary interaction/`pass_by` | 18 / 25 | The headline number: full Precision/Recall/F1 |
| 2 | enter_vehicle, exit_vehicle | 6, 7 | per-type rates, with inline counts | 
| 3 | attend_vehicle, open_close_door, load_unload | 2, 2, 1 | confusion-matrix entries, no rates |

`pass_by` negatives allow us to convert a FP number to a diagnostic.

**Descriptions** are scored against the GT `person_desc`/`vehicle_desc`, slot by slot, on matched interactions whose person and vehicle agree with the GT anchors (a match about a different person would grade the wrong description). Colours are scored twice: the named colour, and dark/light only, since the GT often records only brightness at night or in grayscale; grey and silver count as one colour. `unknown` counts as an abstention: excluded from accuracy, reported as lost coverage. Vehicle parked/stopped are merged.

## 4. Results

The system was tuned on three knobs: detector confidence (`det_conf`), how close a person must come for contact to begin (`tau_near`), and how far they must leave for it to end (`tau_far`). These differ deliberately so that contact doesn't flicker. We tuned both on all data and leave-one-clip-out (LOCO), with a pure-geometry arm as control.

| | P | R | F1 |
|---|---|---|---|
| VLM, held out | 0.667 | 0.667 | **0.667** |
| VLM, all-data *(not held out)* | 0.684 | 0.722 | **0.703** |
| geometric control, held out | 0.295 | 0.722 | **0.419** |

Our final outputs achieve an F1 of **0.667**, which is below the 0.703 the sweep predicted for this config. This is because we re-ran the VLM under a different torch build and flipped four verdicts (see §5).

1. The overfitting gap is **+0.036**. This gap is small because the selection is stable: 7 of 8 folds pick identical settings.
2. Before tuning, we found the VLM's improvement over a pure geometric method was 0.5 vs 0.286, a gain of 0.214. After tuning, the gap rises to **+0.248**. The ablation supports the architecture.

Secondary localization: With tIoU >= 0.5, we get an F1 of 0.500, against 0.667 primary. This gap is due to inherent boundary ambiguity, which is why 0.3 is primary.

**Descriptions** (10 scored pairs; 2 of 12 matches excluded for mismatched actors):

| person | correct | vehicle | correct |
|---|---|---|---|
| upper colour, dark/light | 10/10 | colour, dark/light | 10/10 |
| age | 9/10 | colour, named | 9/10 |
| lower colour, dark/light | 7/10 | body type | 9/10 |
| sex | 6/10 | state | 9/10 |
| upper colour, named | 3/6 | | |
| lower colour, named | 2/5 | | |
| carrying | 4/10 | | |
| **all person slots** | **41/61** | **all vehicle slots** | **37/40** |

Vehicles are described well. People are described well for brightness and age, but poorly for named colours, sex, and carrying. All four sex errors are in the night car-park clip (`iMGR_0AG3a8_2_3`), and five of the six carrying errors are the model saying "yes" when nothing is carried.

## 5. Limitations:
- **n=18 positives**: One event is 5.6% of recall. No confidence intervals.
- The search grid in the tuning process was not centered correctly. `det_conf` took its maximum value in all 8 folds.
- The VLM judge does not reproduce across torch builds. Rerunning under torch 2.14.0+cu130 vs cu132 resulted in identical proposal spans with flipped judge verdicts on 4 candidates. Tracking and proposal are deterministic, but the semantic layer is not.
- The VLM prompt was iterated against the evaluation set. We tried three prompts, and kept the best. This is tuning that was not included in the LOCO.
- **Label Noise**: We didn't properly differentiate between doors opening and closing and people entering or exiting the vehicles.
- **`gt1125_06` has no door cue** because the open vocabulary detector fails on the aerial imagery. Since both events are labeled enter_vehicle, we did not lose any recall, but that is more an effect of label noise and luck than anything else.
- With only n=2 `attend_vehicle` events, `attend_vehicle` vs `pass_by` cannot be tuned, and the decision rests solely on the prompt wording.
- **Descriptions rest on 10 pairs** and are sensitive to prompt wording: changing only the carrying question moved vehicle body type from 6/9 to 9/10. Carrying (4/10) is not reliable.
- In one iMGR pair the GT says "young girl, pink pants" and the prediction "adult male, black trousers"; this is more likely the wrong person than a wrong description, which would mean the anchor-based actor check is too lenient. The GT also calls the iMGR vehicle a hatchback where it is arguably an SUV.

## 6. Next Steps:

1. Recenter the LOCO grid
2. More labelled data. Every number is limited by the number of cases, not the methodology.
3. A stronger VLM to adjudicate between `attend_vehicle` and `pass_by`, and to describe people in night footage.
4. A stricter actor check (per-frame anchors rather than a union box) before scoring descriptions.