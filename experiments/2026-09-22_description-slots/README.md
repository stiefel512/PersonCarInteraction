# Description slot accuracy (2026-09-22)

Structured description pass (`pvi/judge/describe.py`) added as a separate VLM call
on accepted candidates; scored by `pvi/evaluate/descriptions.py`.

- Config: `config.yaml` (identical resolved config across all 8 clips; seed 0,
  greedy decoding, Qwen2.5-VL-7B @ cc594898).
- Judge verdicts unchanged vs. HEAD 9f38b1c on all 8 clips (all judge prompts
  were cache hits; 18 new cache entries = 18 description calls).
- Scored pairs: 10 of 12 primary TPs (2 excluded for actor mismatch).
- `results.json`: full description report incl. per-pair outcomes.
- `per_slot.csv`: per-slot correct/wrong/abstain, accuracy, coverage.

**Superseded by `2026-09-22_description-slots-v2`.** The v1 description cache
entries were removed from `data/vlm_cache/` (prompt changed, never hit again),
so these numbers need a GPU re-run to reproduce.
