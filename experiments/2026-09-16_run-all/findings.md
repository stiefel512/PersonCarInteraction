# Full-set baseline — geometric (ablation) judge

All 8 clips, `config/default.yaml` defaults (untuned), geometric judge.
Numbers: `comparison.csv`, `results.json`. Outputs: `geometric/`.

The geometric judge **accepts every proposal**. So this run measures the
proposer alone, and its precision is exactly the deficit the VLM has to make up.

## Result

| clip | P | R | F1 | tp/fp/fn | pass_by fired |
|---|---|---|---|---|---|
| `1THkHYIQ_bY_0` | 0.000 | 0.000 | 0.000 | 0/5/1 | 0/1 |
| `HIu4lM4B8hA_1` | 0.143 | 0.333 | 0.200 | 1/6/2 | 0/0 |
| `NmlzoaDcOuI_1` | 0.286 | 0.667 | 0.400 | 2/5/1 | 2/2 |
| `NmlzoaDcOuI_6` | 0.167 | 1.000 | 0.286 | 1/5/0 | 2/2 |
| `gt1125_06` | 0.143 | 1.000 | 0.250 | 2/12/0 | 0/0 |
| `iMGR_0AG3a8_2_3` | 0.375 | 0.500 | 0.429 | 3/5/3 | 0/1 |
| `mKzCQKTHizw_0` | 0.250 | 1.000 | 0.400 | 1/3/0 | 1/1 |
| `mKzCQKTHizw_1` | **1.000** | **1.000** | **1.000** | 1/0/0 | 0/0 |
| **POOLED** | **0.327** | **0.944** | **0.486** | 17/35/1 | **6/7** |

## Reading it

**The proposer is doing its job.** 17 of 18 positives are proposed. The design
splits recall (proposer) from precision (judge), and the recall half works at
default thresholds with no tuning.

**Precision 0.327 and 6 of 7 `pass_by` fired is the expected control result**,
not a failure. A judge that accepts everything cannot reject a near-miss. This
is the number the VLM arm has to move, and it is why the ablation exists.

**`gt1125_06` finds both positives** — the recall stress case, on a moving
camera at 4K. Tiling earned its place. 12 false positives there is the most of
any clip, unsurprising in a frame containing ~25 vehicles and ~6 people.

**`mKzCQKTHizw_0`, the precision stress case, fired on its 1 labeled `pass_by`**
— exactly the confusion the clip was chosen to provoke.

**One positive is missed outright**: `1THkHYIQ_bY_0`'s `attend_vehicle` 0-265,
an 8.8 s event on the night consumer-cam clip. Detection is not the cause —
that clip yields 2096 detections and 7 person tracks. The cause is **span
fragmentation**: the proposer emits four short candidates (53-69, 69-93,
125-166, 225-247) against one long GT span, each overlapping only 6-16%, so none
reaches tIoU 0.3. `HIu4lM4B8hA_1` shows the same pattern against its 150-frame
`attend_vehicle`.

That is the clearest lead for the next improvement, and it is a proposer
problem, not a detector one: the hysteresis span breaks when a person leaning
into a car drifts past `tau_far`, or when their track breaks on hard imagery.
`tau_far` is in the LOCO search, so some of this may resolve under tuning.

## A correction made while producing this table

The first version of this run scored **P=0.231 R=0.667 F1=0.343**. Those numbers
were wrong, and the cause was the evaluation, not the pipeline.

Actor identity was being used as a **hard gate**: a prediction whose union box
did not contain the GT anchor was refused a match. Measured, that cost **5 of 17
true positives** and lowered *both* precision (0.327 → 0.231) and recall
(0.944 → 0.667) — a rejected match becomes a false positive *and* a false
negative, so a strict actor test is not the conservative choice it appears to
be. It failed on correct detections wherever the person track fragmented, which
is exactly what happens on the night and CIF-grayscale clips. Two positives had
a candidate at tIoU 0.62 and 0.59 thrown away this way.

Actor agreement now **ranks** pairings instead of gating them: a pairing whose
actors agree sorts ahead of one whose actors do not, with tIoU breaking ties
within each group. That keeps what the check was introduced for — on
`NmlzoaDcOuI_6` a prediction about a different vehicle no longer wins over the
correct one — without discarding correct matches when tracking is imperfect.

## Caveats

- **Untuned.** These are `config/default.yaml` first guesses, not LOCO-selected
  values. Reported numbers will move.
- **18 positives.** One event is 5.6% of aggregate recall. Nothing here supports
  a confidence interval.
- The VLM column of this table is still missing; the comparison it exists for is
  half-done.
