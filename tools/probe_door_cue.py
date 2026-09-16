"""Open-vocabulary door cue on GROUND-LEVEL clips (design-plan §6.7, arm C part 2).

Arm C on `gt1125_06` found that Grounding DINO grounds "person" and "car" well
on aerial footage but does not reliably ground "open car door" -- it missed the
clearly open driver door at frame 222 and its two firings sat at the box
threshold, one returning the token fragment "open".

That is not enough to decide the design, because `gt1125_06` is the *hardest*
case for this model: the literature review records small objects and aerial
imagery as Grounding DINO's documented weaknesses. The cue would be adopted for
every clip, and most clips are ground-level with large, clearly visible doors.

So this runs the same prompt on frames drawn from the ground-level clips whose
ground truth contains a door-centric event, where a door is known to be open.
If it fails there too, the cue is dead. If it works there, the honest conclusion
is that R4 becomes camera-independent on 5 clips and stays unavailable on the
aerial one -- which is still a gain, since R4's pixel-change heuristic is
currently dead on 3 clips.

Usage:
    .venv/bin/python tools/probe_door_cue.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from PIL import Image, ImageDraw

from pvi.detect.openvocab import OpenVocabDetector, format_prompt
from pvi.schema import load_ground_truth
from pvi.video import probe, read_frames

PHRASES = ["open car door", "person", "car"]

# Ground-level clips whose GT contains a door-centric event. Frames are sampled
# inside those event spans, where a door is known to be open or being handled.
DOOR_EVENT_TYPES = {"open_close_door", "load_unload"}
EXTRA_CLIPS = {
    # Described in problem-definition.md §2 as door-centric, though their GT
    # events are typed exit/enter rather than open_close_door.
    "HIu4lM4B8hA_1": "person opens the driver door of a white hatchback",
    "mKzCQKTHizw_1": "person opens the door of a silver sedan and leans in",
}
N_FRAMES_PER_EVENT = 4


def pick_frames() -> dict[str, list[int]]:
    gts = load_ground_truth("data/ground_truth.json")
    out: dict[str, list[int]] = {}
    for g in gts:
        if g.clip_id == "gt1125_06" or not g.is_positive:
            continue
        if g.type in DOOR_EVENT_TYPES or g.clip_id in EXTRA_CLIPS:
            span = g.frame_end - g.frame_start
            if span <= 0:
                continue
            step = max(1, span // (N_FRAMES_PER_EVENT - 1))
            frames = [min(g.frame_end, g.frame_start + i * step)
                      for i in range(N_FRAMES_PER_EVENT)]
            out.setdefault(g.clip_id, [])
            for f in frames:
                if f not in out[g.clip_id]:
                    out[g.clip_id].append(f)
    return {k: sorted(v) for k, v in out.items()}


def main() -> None:
    outdir = Path(f"experiments/{date.today():%Y-%m-%d}_door-cue-ground-level")
    (outdir / "frames").mkdir(parents=True, exist_ok=True)

    ov = OpenVocabDetector()
    selection = pick_frames()
    results: dict = {}

    for clip_id, indices in selection.items():
        video = Path("Videos") / f"{clip_id}.mp4"
        meta = probe(video)
        frames = read_frames(video, indices, meta)
        per_frame = {}
        for i in indices:
            dets = ov.detect([frames[i]], [i], PHRASES)[0]
            by_label: dict[str, int] = {}
            for d in dets:
                by_label[d.label] = by_label.get(d.label, 0) + 1
            door = [d for d in dets if "door" in d.label or d.label == "open"]
            per_frame[str(i)] = {
                "by_label": by_label,
                "n_door_hits": len(door),
                "door_hits": [{"label": d.label, "conf": round(d.conf, 4),
                               "box": [round(v, 1) for v in d.box]} for d in door],
            }
            img = Image.fromarray(frames[i])
            dr = ImageDraw.Draw(img)
            w = max(2, int(round(min(img.size) / 180)))
            for d in dets:
                is_door = "door" in d.label or d.label == "open"
                dr.rectangle(d.box, outline=(255, 40, 40) if is_door else (255, 200, 0),
                             width=w * 2 if is_door else w)
                dr.text((d.box[0] + 4, max(0, d.box[1] - 20)),
                        f"{d.label} {d.conf:.2f}",
                        fill=(255, 40, 40) if is_door else (255, 200, 0))
            img.save(outdir / "frames" / f"{clip_id}_f{i:06d}.png")

        hits = sum(v["n_door_hits"] for v in per_frame.values())
        results[clip_id] = {
            "resolution": f"{meta.width}x{meta.height}",
            "n_frames": len(indices),
            "frames_with_a_door_hit": sum(1 for v in per_frame.values()
                                          if v["n_door_hits"]),
            "total_door_hits": hits,
            "per_frame": per_frame,
        }
        print(f"{clip_id:20s} {meta.width}x{meta.height:<5} "
              f"{results[clip_id]['frames_with_a_door_hit']}/{len(indices)} frames "
              f"with a door hit ({hits} boxes)")

    cfg = {"experiment": "open-vocab door cue on ground-level clips",
           "date": str(date.today()), "seed": 0,
           "model_id": "IDEA-Research/grounding-dino-base",
           "revision": ov.revision,
           "prompt": format_prompt(PHRASES),
           "box_threshold": 0.27, "text_threshold": 0.25,
           "frames": selection,
           "rationale": "arm C on gt1125_06 tested the model's weakest regime "
                        "(aerial, small objects); this tests the regime the cue "
                        "would actually be adopted for"}
    (outdir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (outdir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {outdir}/results.json")


if __name__ == "__main__":
    main()
