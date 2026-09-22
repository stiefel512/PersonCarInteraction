### Person-Vehicle Interaction Detection — Write-up

Full numbers, tables and discussion: [`RESULT_ANALYSIS.md`](RESULT_ANALYSIS.md). Run instructions: [`README.md`](README.md). Deliverable: [`outputs/interactions.json`](outputs/interactions.json).

#### 1. Approach

The pipeline has four stages: detect people and vehicles, track, propose contacts, then have a VLM judge each proposal. Each accepted interaction then gets a VLM-produced description of the participants.

- **Detect + track:** RF-DETR-L and ByteTrack, with ORB camera-motion compensation.

- **Propose:** proximity with hysteresis (`tau_near`/`tau_far`), dwell time, the track starting or ending at the vehicle, and an open-door cue from Grounding DINO.

- **Judge:** Qwen2.5-VL-7B sees frames with the person boxed in red and the vehicle in blue. It picks `pass_by` or one of five interaction types. It is also told what the tracker saw, e.g. "first appeared at the vehicle".

- **Describe:** a separate call returns fixed-vocabulary slots: person sex, age, clothing colours and carrying yes/no; vehicle colour, body type and state.

Nothing is trained. Three proposal thresholds are tuned with leave-one-clip-out (LOCO) cross-validation against 25 hand-labelled events (18 interactions, 7 `pass_by` negatives). The labels were frozen before development.

**Determinism:** model revisions are pinned, the seed is fixed, decoding is greedy at batch size 1, the resolved config is written next to every output, and every VLM response is cached and committed. A reviewer can reproduce the outputs without a GPU.

#### 2. Assumptions

The full list of ambiguities is in [`docs/problem-definition.md`](docs/problem-definition.md) (A1–A8).

- An interaction is purposeful physical engagement; proximity is not enough. Intent is never labelled (A6).

- One record per person per event. A driver seen only inside a moving car is not an interaction (A2, A7).

- All thresholds are in seconds and in units of the vehicle's diagonal, never frames or pixels, because frame rates and resolutions vary across clips (A8).

#### 3. Headline results

|  | result | source (`outputs/`) |
| - | - | :-: |
| Detection F1, held-out LOCO (tIoU ≥ 0.3) | **0.667** (P 0.667, R 0.667) | `loco_vlm.json` → `headline_held_out` |
| Detection F1, shipped artifact | **0.667** | `metrics.json` → `pooled.primary` |
| Overfitting gap (all-data minus held-out) | +0.036 | `loco_vlm.json` → `overfitting_gap_f1` |
| `pass_by` negatives reported as interactions | 0 / 7 | `metrics.json` → `pooled.primary.tier1_pass_by` |
| Description accuracy, person / vehicle slots | 41/61 / 37/40 | `metrics.json` → `descriptions` |


The VLM judge adds +0.248 F1 over geometry alone, mostly through precision: 0.667 against 0.295 (`outputs/loco_geometric.json` → `headline_held_out`). Descriptions of vehicles are reliable. Descriptions of people are weak on named colours, sex (6/10) and carrying (4/10), especially in night footage. Details: `RESULT_ANALYSIS.md` §4 and `experiments/2026-09-22_description-slots-v2/`.

#### 4. Limitations

- **Small sample:** 18 positives, so one event is 5.6% of recall, and descriptions rest on 10 scored pairs. No confidence intervals.

- **The VLM judge is not reproducible across torch builds:** 4 verdicts flipped between CUDA builds. The committed cache is what makes the outputs reproducible.

- **The judge prompt was chosen on the evaluation set,** outside the LOCO loop.

- **Hard cases:** the aerial clip has no door cue, and `attend_vehicle` vs `pass_by` rests on prompt wording alone (n = 2).

The full list is in `RESULT_ANALYSIS.md` §5.

#### 5. Next steps

1. More labelled data; every number above is limited by n, not by the method.

2. Re-centre the LOCO grid: `det_conf` hit its upper bound in all 8 folds.

3. A stronger VLM for `attend_vehicle` vs `pass_by` and for describing people at night.

4. A stricter check that a prediction is about the same person as the GT event before its description is scored.

