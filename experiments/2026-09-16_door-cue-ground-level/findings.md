# Open-vocabulary door cue — findings (design-plan §6.7 / probe arm C)

This is the decision the design plan deferred to evidence: does R4 stay a
pixel-change heuristic, or become a text prompt?

**It becomes a text prompt on ground-level clips, and does not work on the
aerial one.** The plan assumed a binary answer; the honest answer is split, and
the split still favours adoption.

## Evidence

Model `IDEA-Research/grounding-dino-base` @ `12bdfa3120f3e7ec7b434d90674b3396eccf88eb`,
prompt `"open car door. person. car."`, box threshold 0.27.

| clip | resolution | frames with a door hit | grounded label | confidence |
|---|---|---|---|---|
| `HIu4lM4B8hA_1` | 352×288 grayscale | **12/12** | `open car door` | 0.29–0.60 |
| `mKzCQKTHizw_1` | 640×360 | **4/4** | `open car door` | 0.28–0.39 |
| `iMGR_0AG3a8_2_3` | 960×720 night | **8/12** | `open car door` | 0.28–0.40 |
| `gt1125_06` | 3840×2160 aerial | **2/7** | `open` (fragment) | at threshold |

Frames were drawn from inside GT event spans where a door is known to be open.

**Localization was checked by eye, not just counted.** On `mKzCQKTHizw_1` f145
the two boxes land squarely on the open door aperture with the person leaning
in; on `HIu4lM4B8hA_1` f030 — the lowest-resolution clip in the set, grayscale —
they sit on the open driver door. Annotated frames are in `frames/`.

**The aerial failure is a real miss, not a threshold artefact.** At `gt1125_06`
frame 222 a driver door is unambiguously open (visible in the native-resolution
crop) and the cue did not find it. Its two firings there returned the token
fragment `open` rather than the grounded phrase, which is the model signalling
weak phrase grounding rather than a confident detection.

This matches the literature review exactly: Grounding DINO's documented
weaknesses are small objects and aerial imagery.

## Why the split still favours adoption

The incumbent heuristic measures appearance change inside the vehicle box
against a temporal median. It requires **a static camera and a parked vehicle**.
After the camera-motion correction (3 of 8 clips move, not 1), it is dead on:

- `gt1125_06` — moving camera
- `mKzCQKTHizw_0` — moving camera
- `mKzCQKTHizw_1` — moving camera, **and it contains a door event**

So R4 is currently available on 5 of 8 clips, and unavailable on one where it is
needed. The text prompt is camera-motion-independent and works on every
ground-level clip tested, including the two the heuristic cannot serve.

| | heuristic | text prompt |
|---|---|---|
| clips where R4 can fire | 5/8 | **7/8** |
| fails on | 3 moving-camera clips | 1 aerial clip |
| needs a static-camera branch | yes | **no** |

Adoption also deletes a conditional branch, which was the architectural argument
in the first place.

## Decision

- **R4 uses the open-vocabulary cue.** The pixel-change heuristic is retired
  rather than kept as a fallback: the two fail on *disjoint* clips, so keeping
  both would mean maintaining the static-camera branch the change exists to
  delete, for a single clip.
- **`gt1125_06` has no door cue.** `open_close_door` cannot be proposed there.
  Its 2 GT positives are both `enter_vehicle`, which R2 proposes, so this costs
  no recall against the current ground truth — but it is a stated limitation,
  not a silent one.
- The cue runs only on frames already inside a near-span, so the cost is a
  Grounding DINO pass over a small fraction of frames rather than all of them.

## Caveats

- **28 frames across 3 clips**, chosen from inside known door events. This
  measures whether the cue *fires where a door is open*; it does not measure how
  often it fires where none is. False-positive behaviour on door-free footage is
  untested, and `door_conf_thresh` exists to control it.
- Confidences cluster at 0.28–0.40, not far above the 0.27 box threshold. The
  margin is thinner than the hit rate suggests.
- `iMGR_0AG3a8_2_3` at 8/12 is the weakest ground-level clip — it is the night
  car-park scene, so low light costs something here as expected.
