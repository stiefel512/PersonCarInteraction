# Headline ablation — geometric vs VLM judge

> **CORRECTED 2026-09-16.** Every pooled number below was computed with a flat
> cross-clip matching and is inflated by roughly 55%. Frame spans are clip-local,
> so a prediction from one clip could satisfy another clip's event. The per-clip
> numbers are unaffected. Corrected pooled figures and the full explanation:
> `../2026-09-16_bytetrack-floor/findings.md`. Superseded by that run.

All 8 clips, `config/default.yaml` (untuned), tIoU >= 0.3.
`comparison.csv` here; the two superseded VLM prompts are in
`../2026-09-16_ablation/` (v1) and `../2026-09-16_ablation-v2/` (v2).

## Result

| judge | P | R | F1 | tp/fp/fn | `pass_by` fired |
|---|---|---|---|---|---|
| geometric (control) | 0.327 | **0.944** | 0.486 | 17/35/1 | 6 of 7 |
| **VLM (v3, shipped)** | **0.583** | 0.778 | **0.667** | 14/10/4 | **0 of 7** |

**The VLM earns its place.** Against the rule-only control it nearly doubles
precision (0.327 -> 0.583), cuts false positives from 35 to 10, rejects **every
one of the 7 labeled near-misses**, and raises F1 by 0.18. It pays 3 true
positives for that (17 -> 14).

This is the comparison the whole hybrid design rests on, and it comes out the
way the design assumed: geometry proposes at high recall, semantics filters.
Neither half would be acceptable alone -- the control reports 35 false positives
against 18 real events, and the proposer is what finds 17 of those 18 in the
first place.

Per clip:

| clip | geometric | VLM v3 |
|---|---|---|
| `1THkHYIQ_bY_0` | 0.00/0.00/0.00 | 0.00/0.00/0.00 |
| `HIu4lM4B8hA_1` | 0.14/0.33/0.20 | 0.33/0.33/0.33 |
| `NmlzoaDcOuI_1` | 0.29/0.67/0.40 | 0.50/0.33/0.40 |
| `NmlzoaDcOuI_6` | 0.17/1.00/0.29 | **1.00/1.00/1.00** |
| `gt1125_06` | 0.14/1.00/0.25 | 0.33/1.00/0.50 |
| `iMGR_0AG3a8_2_3` | 0.38/0.50/0.43 | 0.60/0.50/0.55 |
| `mKzCQKTHizw_0` | 0.25/1.00/0.40 | 0.50/1.00/0.67 |
| `mKzCQKTHizw_1` | 1.00/1.00/1.00 | **1.00/1.00/1.00** |

## How the prompt got here, including the version that failed

Three versions were run in full. The middle one is recorded because it was a
**negative result that looked positive on the first few clips**, and stopping
there would have shipped a worse system.

| | P | R | F1 | `pass_by` fired |
|---|---|---|---|---|
| v1 — strong `pass_by` rule, no tracker evidence | 0.556 | 0.556 | 0.556 | 0/7 |
| v2 — weakened `pass_by` rule + unconditional evidence | 0.438 | 0.778 | **4/7** | 4/7 |
| v3 — strong rule + **edge-conditioned** evidence | **0.583** | 0.778 | **0.667** | 0/7 |

**v1's failure was systematic, not noise.** Every one of its 34 rejections was
typed `pass_by` at 0.86-0.96 confidence -- not one was a low-confidence
positive. Every identifiable lost positive was an `exit_vehicle`, including one
at tIoU 0.97 where all four proposal rules fired. The ground truth labels an
exit from the first frame the person is *visible*, so the model saw someone
already beside a car who then walked away, and v1's rule said "stands near it ->
pass_by" with no exception.

**v2 fixed that and broke something better.** Weakening the `pass_by` rule and
telling the model "their first appearance is at the vehicle, never seen
approaching" recovered 4 true positives -- and let 4 of 7 labeled near-misses
through. F1 moved +0.004: an even trade, and under the task's stated
precision-over-recall stance, a worse system than v1.

**The bug was in the evidence, not the balance.** Track birth means the tracker
first saw the person; it does **not** mean they emerged from a vehicle. A
pedestrian walking into shot beside a parked car satisfies "first appearance is
at the vehicle" exactly as well as a real exit does. v2 was feeding the model a
true-but-misleading claim on precisely the negatives.

**v3 makes that distinction geometrically** (`propose.rules.at_frame_edge`):
did the person's first box touch the frame border? The evidence now says
opposite things in the two cases -- "in open view, not at the edge; they did not
walk in from off-camera" versus "first became visible at the EDGE, so they
walked in from off-camera; they did not come out of the vehicle" -- and the
strong `pass_by` default is restored with the exception narrowed to the
open-view case. The margin is a fraction of frame size, so it means the same
thing at 352x288 and at 4K.

Result: v2's recall, v1's negatives, better precision than either.

## `vlm_conf_thresh` is inert, and that matters for LOCO

Across all three versions **no candidate was ever rejected by the confidence
threshold.** Qwen is decisive: it answers `pass_by` at 0.9, not at 0.4. So
`config/schema.md`'s "main precision/recall dial" does nothing at this operating
point, and lowering it would recover zero lost positives.

That is a problem for the agreed LOCO scope (design-plan §6a.2), which searches
`det_conf`, `tau_near`, `tau_far` and `vlm_conf_thresh`. One of the four is a
no-op. Options, for the user to decide:
- drop it and search three, or
- replace it with a knob that *is* live -- e.g. thresholding an interaction
  score of `conf if type != pass_by else 1 - conf`, which would make a
  low-confidence `pass_by` recoverable. Note the observed `pass_by`
  confidences are 0.86-0.96, so even this may have little room.

## Caveats

- **Untuned defaults.** LOCO has not run; these numbers will move.
- **18 positives, 7 negatives.** One event is 5.6% of recall. No confidence
  intervals are supportable, and the per-clip cells are 1-6 events each.
- **The prompt was iterated against the evaluation set.** Three versions were
  scored on the same 8 clips and the best kept. That is threshold-tuning by
  another name and it inflates these numbers; it is exactly what LOCO is meant
  to quantify, and the write-up must say so.
- `1THkHYIQ_bY_0` scores 0 under both judges. Its single positive is lost in the
  proposer to span fragmentation, so the judge never sees a matching candidate.
