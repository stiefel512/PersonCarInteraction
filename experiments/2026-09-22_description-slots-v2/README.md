# Description slot accuracy, v2: carrying as yes/no (2026-09-22)

Supersedes `2026-09-22_description-slots` (v1). Change: the person's free-text
`carried_object` slot became closed `carrying` (yes|no|unknown). In v1 the model
named objects absent from the GT, including "gun".

- Judge verdicts still identical to HEAD 9f38b1c on all 8 clips.
- The describe prompt text changed, so ALL description slots were regenerated,
  not just `carrying`. Vehicle slots moved as a side effect (body type 6/9 -> 9/10),
  which is prompt sensitivity, not an improvement from the carrying change.
- Only the v2 description responses are in `data/vlm_cache/`.
