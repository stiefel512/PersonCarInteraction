# BlackRover — Person-Vehicle Interaction Detection

Take-home task: given a short silent MP4 clip, output a machine-readable list of
person-vehicle interactions. Task statement in `HomeTask.docx`.

See `docs/problem-definition.md` for problem scope, metrics, and track decision.
See `docs/design-plan.md` for architecture, component contracts, and config schema.

## Project-specific notes

- **Framework:** PyTorch. System Python is 3.14.5, which has no torch wheels —
  use a dedicated **Python 3.11/3.12 venv** for this project.
- **Hardware:** RTX A6000, 48 GB VRAM.
- **Determinism is a graded requirement** of the task: pin model revisions, fix
  seeds, greedy decoding only, fixed frame-sampling schedule, resolved config
  written next to every output.
- Deliverable is a public GitHub repo: source + README run instructions, the
  output artifact, and a ≤2-page write-up.

## Ground truth

`data/ground_truth.json`, produced with `tools/label_gt.py`.
Conventions in `docs/labeling-protocol.md` — contact-based boundaries, explicit
`pass_by` negatives, `ambiguous` flag, frozen once evaluation starts.
Regenerate frames with `python3 tools/extract_frames.py` (stdlib + Pillow, runs on
system Python — no venv needed).
Config schema (fixed vs tunable split) in `config/schema.md`.

## Method selection

Detector/tracker choice follows a literature review (2026-09-16) —
see `~/research/literature/object-detection-heterogeneous-surveillance/summary.md`.
