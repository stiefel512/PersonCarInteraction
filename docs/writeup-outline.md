# Write-up outline (≤2 pages)

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
- The table worth printing (LOCO, three knobs):

  | | P | R | F1 |
  |---|---|---|---|
  | VLM, held out | 0.520 | 0.722 | **0.605** |
  | VLM, all-data *(not held out)* | 0.737 | 0.778 | 0.757 |
  | geometric control, held out | — | — | 0.581 |

- **The two findings to actually argue:**
  1. The **+0.152 overfitting gap**. Reporting 0.757 would have been defensible
     and wrong. This is the most honest number in the project.
  2. Once both arms are tuned, **the VLM's margin nearly vanishes** (0.605 vs
     0.581) — against 0.500 vs 0.286 untuned. Say plainly that the ablation
     undercuts the architecture's apparent value.
- Per-clip spread, briefly: `iMGR` 0.909 vs two folds at 0.000. With ~2.25
  positives per fold, one clip swings the pooled number.
- Fold agreement as evidence for (1): geometric near-unanimous, VLM `tau_near`
  taking 4 distinct values across 8 folds.

## 5. Limitations  [~1/3 page]

Do not hedge these into meaninglessness; each is specific and checkable.

- **n=18 positives.** One event is 5.6% of recall. No confidence intervals.
- **The grid was mis-centred** — `det_conf` pinned at the top edge in 7 of 8
  folds, so 0.605 is a lower bound, not an optimum.
- **The prompt was iterated against the evaluation set** (three versions, best
  kept). That is tuning by another name and LOCO does not absorb it.
- **`gt1125_06` has no door cue** (open-vocab fails on aerial); its two positives
  are both `enter_vehicle`, so no recall is lost — but say so rather than let it
  pass silently.
- `attend_vehicle` vs `pass_by` at n=2 cannot be tuned; it rests on prompt wording.

## 6. What I would do next  [~1/6 page]

Three, concrete, in priority order. Suggested:

- Re-centre the LOCO grid (boundary selections mean the search never bracketed
  the optimum).
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
- Cut §1–2 before cutting §4–5 if you run long.
