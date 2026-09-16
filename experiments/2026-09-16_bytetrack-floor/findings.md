# Current result — after the fragmentation fix, corrected scoring

All 8 clips, `config/default.yaml` (untuned), tIoU >= 0.3, **matched per clip**.
`comparison.csv` here. This supersedes every pooled number in the earlier
experiment directories; see "The scoring bug" below.

## Headline ablation

| judge | P | R | F1 | tp/fp/fn | `pass_by` fired |
|---|---|---|---|---|---|
| geometric (control) | 0.178 | **0.722** | 0.286 | 13/60/5 | 5 of 7 |
| **VLM (shipped)** | **0.400** | 0.667 | **0.500** | 12/18/6 | **1 of 7** |

The VLM more than doubles precision (0.178 -> 0.400), cuts false positives from
60 to 18, takes `pass_by` false fires from 5 to 1, and raises F1 by 0.21, paying
one true positive. The hybrid split holds: geometry proposes at high recall,
semantics filters.

Per clip, VLM arm:

| clip | P | R | F1 | tp/fp/fn |
|---|---|---|---|---|
| `1THkHYIQ_bY_0` | **1.00** | **1.00** | **1.00** | 1/0/0 |
| `HIu4lM4B8hA_1` | 0.50 | 0.33 | 0.40 | 1/1/2 |
| `NmlzoaDcOuI_1` | 0.33 | 0.33 | 0.33 | 1/2/2 |
| `NmlzoaDcOuI_6` | 0.50 | 1.00 | 0.67 | 1/1/0 |
| `iMGR_0AG3a8_2_3` | **0.80** | 0.67 | **0.73** | 4/1/2 |
| `mKzCQKTHizw_0` | 0.50 | 1.00 | 0.67 | 1/1/0 |
| `mKzCQKTHizw_1` | 0.50 | 1.00 | 0.67 | 1/1/0 |
| `gt1125_06` | 0.15 | 1.00 | 0.27 | 2/11/0 |

`gt1125_06` carries most of the remaining false positives (11 of 18). It is the
4K aerial clip with ~25 vehicles and ~8 people per frame, so the pair count is
an order of magnitude above any other clip.

## Was the fragmentation fix worth it?

This was genuinely open, and the two arms disagree.

| | geometric | VLM |
|---|---|---|
| before | P 0.211 / R 0.611 / F1 0.314 | P 0.417 / R 0.556 / F1 0.476 |
| after | P 0.178 / R **0.722** / F1 **0.286** | P 0.400 / R **0.667** / F1 **0.500** |

**On the control arm the fix looks bad**: recall rises 0.611 -> 0.722 but
precision falls further, so F1 drops. Judged on the proposer alone, it should be
reverted.

**On the shipped arm it is a clear win**: the same recall gain survives, the
precision cost mostly does not, and F1 rises 0.476 -> 0.500.

That difference is the answer to the question the fix was made under: the extra
false positives are exactly the kind the VLM is good at rejecting, so it absorbs
them. Keeping the fix is therefore contingent on keeping the VLM judge -- worth
stating, because it means the control arm is not merely a weaker version of the
system, it prefers a different proposer configuration.

The fix's target is now perfect: `1THkHYIQ_bY_0`, whose 8.8 s `attend_vehicle`
was unfindable because the event fragmented into four sub-threshold candidates,
scores **P=1.00 R=1.00** under the VLM.

## The scoring bug

**Every pooled number reported before this run was inflated by roughly 55%.**

`full_report` pooled all predictions and all ground truth into ONE flat
matching. Frame spans are clip-local, so a prediction from one clip could
satisfy an event in another purely because the numbers overlapped. Minimal
demonstration (`tests/test_match.py`): a prediction from clip A that misses its
own event entirely scores P=1.00 by landing on clip B's event.

On the real set this reported 17 pooled true positives where the per-clip counts
summed to 11. Corrected figures, all recomputed from the same stored outputs via
`run_all.py --rescore`:

| config | was | corrected |
|---|---|---|
| geometric, before fix | P 0.327 / R 0.944 / F1 0.486 | P 0.211 / R 0.611 / F1 0.314 |
| VLM v1 | P 0.556 / R 0.556 / F1 0.556 | P 0.444 / R 0.444 / F1 0.444 |
| VLM v2 | P 0.438 / R 0.778 / F1 0.560 | P 0.344 / R 0.611 / F1 0.440 |
| VLM v3 | P 0.583 / R 0.778 / F1 0.667 | P 0.417 / R 0.556 / F1 0.476 |

The per-clip numbers were always computed correctly and are unchanged. Only the
pooled rows were wrong.

**The conclusions drawn from those numbers survive**, which is luck rather than
design -- the bug inflated every configuration similarly, so the comparisons
between them held even though the absolute values did not. Matching is now
clip-scoped (`metrics.report_groups`), with a test asserting the old flat
behaviour would have credited the cross-clip match.

## Caveats

- **Untuned.** LOCO has not been run. These are first-guess defaults.
- **18 positives, 7 negatives.** One event is 5.6% of recall. No confidence
  interval is supportable.
- **The prompt was iterated against this evaluation set** (three versions,
  best kept). That is threshold-tuning by another name and it inflates these
  numbers; LOCO exists to quantify exactly that, and the write-up must say so.
- `vlm_conf_thresh` remains inert -- no candidate has ever been rejected by it.
