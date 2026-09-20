# Write-up outline (≤2 pages)

> ## ⚠ Numbers corrected 2026-09-19 — WRITEUP.md still has the old ones
>
> `pvi/evaluate/loco.py` matched predictions against ground truth in one flat
> pool across all 8 clips — the same cross-clip bug `metrics.report_groups` was
> fixed for on 09-16, which LOCO never received. It credited predictions against
> other clips' events. Fixed, and both LOCO reports recomputed from the cache.
>
> **What to change in WRITEUP.md §4:**
>
> | | currently says | should say |
> |---|---|---|
> | VLM, held out | 0.520 / 0.722 / **0.605** | 0.667 / 0.667 / **0.667** |
> | VLM, all-data *(not held out)* | 0.737 / 0.778 / **0.757** | 0.684 / 0.722 / **0.703** |
> | geometric control, held out | **0.581** | 0.295 / 0.722 / **0.419** |
> | overfitting gap | +0.152 | **+0.036** |
> | VLM gain over control | +0.024 | **+0.248** |
> | `tau_near` selected | 0.06 | **0.10** |
>
> **Two claims in §4 now say the opposite of the truth and must be rewritten,
> not just renumbered:**
>
> 1. *"the VLM's improvement... the gap drops to 0.024. This ablation actually
>    shows that our architecture provides less value than anticipated."* — The
>    gap is **+0.248**. The ablation supports the architecture; it does not
>    undercut it.
> 2. *"fold-agreement would argue for overfitting... the VLM `tau_near` took 4
>    distinct values across its 8 folds."* — `tau_near` is now **unanimous**
>    across all 8 folds, and `det_conf` too; only `tau_far` varies, taking 2
>    values. Fold agreement now argues *against* the overfitting reading.
>
> Also in §5: *"`det_conf` took its maximum value in 7 of 8 folds"* is now
> **8 of 8**. The mis-centred-grid limitation stands and is stronger.

Skeleton only — headers plus what belongs in each. Rough space budget in
brackets; 2 pages is the binding constraint and §4–5 are what a reviewer will
actually judge, so protect their space.

---

## 1. Problem and approach  [~1/4 page]

- One sentence on the task, and the sentence that makes it hard: *not* reporting
  people who merely walk past.
- The pipeline in one diagram or four lines: detect → track → propose → judge.
- The one-line rationale for the split: contact over an interval is a geometry
  question; interaction-vs-pass-by is a semantic one. Neither half suffices.
- State up front that nothing is trained — every component off-the-shelf, pinned.

## 2. Design decisions  [~1/3 page]

Pick the 3–4 that carry the most weight. Candidates, strongest first:

- **Normalisation invariants** — vehicle-diagonal for space, seconds for time.
  Forced by 352×288→4K and 6→30 fps. Cheap to state, explains a lot of the code.
- **Open-vocab door cue over the pixel-change heuristic** — decided by probe, and
  the answer was split (works ground-level, fails aerial). Good example of
  deciding on evidence rather than argument.
- **Licence-driven model selection** — Apache-2.0 throughout because the repo is
  public; ruled out Ultralytics/BoxMOT (AGPL), DEIMv2 (backbone), Co-DETR
  (unlicensed weights).
- Keep the ambiguity register itself in `problem-definition.md`; reference it
  rather than reproducing it.

## 3. Evaluation methodology  [~1/3 page]

- 25 hand-labeled events (18 positive, 7 `pass_by`), contact-based boundaries,
  frozen before any pipeline run. Say *why* frozen.
- tIoU ≥ 0.3 primary, 0.5 secondary; matching scoped **per clip**.
- **The reporting tiers, and why**: n=1 for `load_unload` means a per-type rate
  is noise. This is the paragraph that shows evaluation judgement — worth the
  space.
- One line that the `pass_by` negatives convert a bare FP count into a
  diagnostic.

## 4. Results  [~1/2 page — the core]

- Lead with the **held-out** number, not the tuned one.
- The table worth printing (LOCO, three knobs, matching scoped per clip):

  | | P | R | F1 |
  |---|---|---|---|
  | VLM, held out | 0.667 | 0.667 | **0.667** |
  | VLM, all-data *(not held out)* | 0.684 | 0.722 | 0.703 |
  | geometric control, held out | 0.295 | 0.722 | 0.419 |

- **Three numbers, and they are not interchangeable — do not conflate them:**

  | number | what it is |
  |---|---|
  | **0.667** | LOCO held out. The headline. |
  | **0.703** | all-data tuned, scored on the same clips that chose the config. |
  | **0.667** | `outputs/` rescored — the shipped artifact. |

  The shipped artifact matching the held-out figure is a **coincidence**, not
  the same quantity. It scores below the 0.703 its own config earned in the
  sweep because the VLM re-ran under a rebuilt torch and flipped 4 verdicts
  (see §5). Quote 0.667 as the headline and, if you print the artifact's score,
  say which of the three it is.

- **The two findings to actually argue:**
  1. The **+0.036 overfitting gap**. Small, and the honest framing is that it is
     small *because the selection is stable*, not because the method
     generalises well — 7 of 8 folds pick an identical setting, so held-out and
     all-data are nearly the same config. Report the gap, then say that.
  2. **The VLM earns its place: +0.248 F1 over the geometric control held out**
     (0.667 vs 0.419), and it fires on **0 of 7** labeled `pass_by` near-misses
     against the control's **5 of 7**. The trade is precision for a little
     recall — 0.295 → 0.667 precision, 0.722 → 0.667 recall — which is exactly
     what the hybrid was designed to buy, since a `pass_by` reported as an
     interaction is the failure the task singles out.
- Secondary localisation number: tIoU ≥ 0.5 gives 0.500 against 0.667 primary.
  The gap is boundary slop, which the labeling protocol already calls ±2 frames.
- Per-clip spread, briefly: `iMGR` 1.000 and `gt1125_06` 0.800 against
  `1THkHYIQ_bY_0` and `NmlzoaDcOuI_6` at 0.000. With ~2.25 positives per clip,
  one clip swings the pooled number.
- Fold agreement, correctly stated: `det_conf` and `tau_near` **unanimous**
  across 8 folds, `tau_far` taking 2 values. Use this to support (1), not to
  argue overfitting.

## 5. Limitations  [~1/3 page]

Do not hedge these into meaninglessness; each is specific and checkable.

- **n=18 positives.** One event is 5.6% of recall. No confidence intervals.
- **The grid was mis-centred** — `det_conf` pinned at the top edge of
  `(0.35, 0.50, 0.65)` in **8 of 8** folds, so 0.667 is a lower bound, not an
  optimum. (`tau_near` at 0.10 is properly interior, so this is a `det_conf`
  problem specifically.)
- **The VLM judge does not reproduce across torch builds.** Re-running the set
  under torch 2.14.0+cu130, where the sweep ran under +cu132, reproduced every
  proposal span *exactly* but flipped the judge's verdict on 4 candidates —
  enough to move pooled F1 from 0.703 to 0.667. Tracking and proposal are
  deterministic; the semantic layer is only as stable as the stack under it, and
  the committed response cache, not the seed, is what actually pins it. This is
  worth stating plainly: it is the single biggest threat to the determinism the
  task grades.
- **A scoring bug reached the reported numbers once already** (LOCO pooled
  matching flat across clips, found 09-19). Worth one line — a reviewer trusts
  numbers more from someone who says where they broke.
- **The prompt was iterated against the evaluation set** (three versions, best
  kept). That is tuning by another name and LOCO does not absorb it.
- **`gt1125_06` has no door cue** (open-vocab fails on aerial); its two positives
  are both `enter_vehicle`, so no recall is lost — but say so rather than let it
  pass silently.
- `attend_vehicle` vs `pass_by` at n=2 cannot be tuned; it rests on prompt wording.

## 6. What I would do next  [~1/6 page]

Three, concrete, in priority order. Suggested:

- Re-centre the LOCO grid — `det_conf` selects the top edge in 8 of 8 folds, so
  the search never bracketed the optimum.
- More labelled data before more modelling — every number here is
  denominator-limited, not method-limited.
- Qwen2.5-VL-32B for the `attend_vehicle`/`pass_by` boundary (the cache is
  already salted by model, so it is a config change).

---

### Notes on framing

- The strongest thing to sell is **evaluation discipline**, not F1. The numbers
  are modest; the reason they are trustworthy is the interesting part.
- Two things reviewers will probe, so pre-empt both: why the pooled numbers are
  clip-scoped, and why `vlm_conf_thresh` is declared tunable but not searched.
- The clip-scoped-matching point is now load-bearing twice over, since the
  project got it wrong twice and caught it twice. That is a strength to state,
  not a weakness to hide.
- Cut §1–2 before cutting §4–5 if you run long.
