# Ground-Truth Labeling Protocol

Companion to `docs/problem-definition.md` §3. Produces `data/ground_truth.json`.
Tool: `tools/label_gt.py`.

The point of writing this down is that the boundary-error metric measures the
*system's* disagreement with the GT only if the GT is internally consistent.
Without a fixed convention it mostly measures the annotator's drift.

## Unit of annotation

One record per **(person, vehicle, event)**. Two people entering the same car are
two records sharing an `event_group_id`. One person who enters and later exits is
two records (`enter_vehicle`, then `exit_vehicle`) — not one span.

## Boundary convention: contact-based

**Start** = the first frame at which the person makes physical contact with the
vehicle, *or* first breaks its silhouette — whichever comes first. Reaching for a
door handle counts from the frame the hand meets the vehicle, not from the frame
the person starts walking toward it.

**End** = the last frame of contact/occlusion.

Two exceptions, both forced by visibility:

| type | rule |
|---|---|
| `enter_vehicle` | End = the last frame any part of the person is visible. |
| `exit_vehicle`  | Start = the first frame any part of the person is visible. |

**Why contact and not approach.** Contact is the least ambiguous observable in the
footage, and it is what the geometric proposal stage actually measures (minimum
normalized distance → 0). Aligning the GT convention with the observable the
system computes avoids manufacturing boundary error that reflects a definitional
mismatch rather than a detection failure. The alternative — "committed approach
begins" — is defensible but requires reading intent from gait, which two
annotators will not agree on.

## Precision to aim for

Nearest confidently-identifiable frame; ±2 frames is fine and expected. At 6 fps
one frame is 167 ms, so do not agonise. This tolerance is exactly why the primary
metric matches at tIoU ≥ 0.3.

## Label the negatives

Record `pass_by` for every genuine near-miss — the chase in `mKzCQKTHizw_0`, the
pedestrians walking within a metre of the subject car in both `NmlzoaDcOuI_*`
clips. These never appear in the deliverable output, but they convert a bare false
positive count into something diagnostic: *"of N labeled near-misses, the system
falsely fired on k"*. That is the number this dataset was built to probe.

## Description fields: fixed slot order

Free text, but keep the slots in a fixed order so GT and VLM output are comparable
by eye without inventing a captioning metric.

- **person** — build/approx age, upper garment + colour, lower garment + colour,
  carried object. *"adult male, green polo, dark shorts, none"*
- **vehicle** — colour, body type, state, position in frame.
  *"red sedan, parked, kerbside centre-left"*

## Ambiguity flag

Tick `ambiguous` whenever you would not defend the call to a second annotator —
unclear whether contact occurred, unclear which of two people is involved,
occluded throughout. Report every metric **twice**: over all events, and over
confident events only. The gap between the two is an honest statement of how much
of the residual error is definitional rather than algorithmic.

## Order of work

1. **Label before running the pipeline.** Once you have seen predictions your
   boundaries drift toward them, and the evaluation quietly stops being
   independent. This is the single most important rule here.
2. **Pass 1 — enumerate.** Play each clip at speed (`space`), list the events,
   do not set boundaries yet. Catches events that frame-stepping tunnel-vision
   misses.
3. **Pass 2 — localize.** Frame-step each event, set `[` and `]`, click the person
   and vehicle anchors, fill descriptions, `Return`.
4. **Save** (`s`) after each clip.

## Self-agreement check

Re-label two clips a day later, cold, and compare. The resulting boundary
disagreement is *your own noise floor* — the system cannot meaningfully be scored
below it. Quoting that floor alongside the results costs twenty minutes and
pre-empts the obvious objection to an n=8 evaluation.

## Freeze it

Once evaluation starts, `data/ground_truth.json` is frozen. If a label must change
afterwards, note what and why in this file rather than silently editing — an
evaluation set edited after seeing results is no longer an evaluation set.

## Anchors

One left-click marks the person, one right-click the vehicle, stored as
**normalized** `(x, y)` plus the frame they were set on. Normalized because the
set spans 352×288 to 4K and pixel coordinates would not be comparable across
clips. Their only job is to disambiguate *which* actor an event refers to when
matching predictions to GT in the multi-actor clips (`iMGR_0AG3a8_2_3`,
`NmlzoaDcOuI_*`). Full per-frame boxes are **not** needed: the metric is temporal,
not spatial.

## Audit log

`tools/validate_gt.py` checks the invariants this protocol implies. Run it after
any edit to `data/ground_truth.json`.

Findings from the 2026-09-14 audit pass, and their resolution:

- **14 duplicate `event_id`s** (hard error). Caused by a bug in `label_gt.py`,
  which derived the next index from `count + 1` — so deleting an event and adding
  another reused an id. Tool fixed to use `max(existing index) + 1`; ids
  reassigned in frame order; pre-fix file kept as `data/ground_truth.pre-idfix.json`.
  Because `event_group_id` defaults to the event's own id, the collisions also
  produced spurious multi-event "groups". No intentional grouping existed
  (verified: every group id equalled its own event id), so all were reset.
- **`HIu4lM4B8hA_1` e002/e004** (38-148, 149-187, same person, abutting): merged
  to 38-187. Frame-stepping 140-160 showed contact never breaks, so under the
  contact-based convention this is one event.
- **`iMGR_0AG3a8_2_3` e004/e006** (99-104, 105-181, same person, abutting):
  kept separate — two distinct door actions on one vehicle (tailgate closed, then
  a side door opened). Recorded in each event's `note`.
- **`gt1125_06` e001/e002** and **`iMGR` e005/e007**: vehicle anchors are 0.14 and
  0.48 apart respectively, confirming different vehicles in both cases. Correctly
  ungrouped. This is what the anchors are for.

Remaining advisory warnings, accepted as-is: one 0.27 s `pass_by` below the
duration floor (negatives are not reported output); three two-slot descriptions;
three anchors placed 1-5 frames outside their span (anchors are spatial hints,
their frame index is not used in matching).
