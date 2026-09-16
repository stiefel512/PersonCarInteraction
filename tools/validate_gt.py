"""Consistency audit for data/ground_truth.json.

Checks the invariants the labeling protocol implies but the labeler does not
enforce. Advisory: prints findings, exits non-zero only on hard errors.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MIN_DUR_S = 0.5          # protocol's proposed floor
SLOT_MIN_COMMAS = 2      # "build, upper, lower[, carrying]"

errors: list[str] = []
warnings: list[str] = []


def main() -> int:
    gt = json.loads((ROOT / "data" / "ground_truth.json").read_text())
    clips = json.loads((ROOT / "data" / "clips.json").read_text())
    ev = gt["events"]

    ids = Counter(e["event_id"] for e in ev)
    for eid, n in ids.items():
        if n > 1:
            errors.append(f"duplicate event_id {eid} x{n}")

    for e in ev:
        eid, cid = e["event_id"], e["clip_id"]
        if cid not in clips:
            errors.append(f"{eid}: unknown clip_id")
            continue
        m = clips[cid]
        fps, nf = m["fps"], m["n_frames"]

        if e["frame_start"] > e["frame_end"]:
            errors.append(f"{eid}: start > end")
        if not (0 <= e["frame_start"] < nf and 0 <= e["frame_end"] < nf):
            errors.append(f"{eid}: frames {e['frame_start']}-{e['frame_end']} outside 0..{nf-1}")

        for fk, tk in (("frame_start", "time_start_s"), ("frame_end", "time_end_s")):
            expect = round(e[fk] / fps, 3)
            if abs(e[tk] - expect) > 0.011:
                errors.append(f"{eid}: {tk}={e[tk]} but {fk}/fps={expect}")

        dur = e["time_end_s"] - e["time_start_s"]
        if dur < MIN_DUR_S:
            warnings.append(f"{eid}: duration {dur:.2f}s < {MIN_DUR_S}s floor ({e['type']})")

        if not e.get("vehicle_desc"):
            errors.append(f"{eid}: empty vehicle_desc")
        if not e.get("person_desc"):
            errors.append(f"{eid}: empty person_desc")
        for k in ("person_desc", "vehicle_desc"):
            if e.get(k) and e[k].count(",") < SLOT_MIN_COMMAS:
                warnings.append(f"{eid}: {k} has <{SLOT_MIN_COMMAS+1} slots: {e[k]!r}")

        for k in ("person_anchor", "vehicle_anchor"):
            a = e.get(k)
            if a is None:
                warnings.append(f"{eid}: no {k} (matching in multi-actor clips will be ambiguous)")
            elif not (e["frame_start"] <= a["frame"] <= e["frame_end"]):
                warnings.append(f"{eid}: {k} frame {a['frame']} outside span")

    # same-clip, same-person-description events that touch or overlap
    by_clip: dict[str, list[dict]] = defaultdict(list)
    for e in ev:
        by_clip[e["clip_id"]].append(e)
    for cid, es in by_clip.items():
        es = sorted(es, key=lambda x: x["frame_start"])
        for i, a in enumerate(es):
            for b in es[i + 1:]:
                if b["frame_start"] > a["frame_end"] + 1:
                    continue
                if a["person_desc"] == b["person_desc"]:
                    kind = "overlap" if b["frame_start"] <= a["frame_end"] else "abut"
                    same_grp = a["event_group_id"] == b["event_group_id"]
                    warnings.append(
                        f"{a['event_id']} / {b['event_id']}: {kind} with identical person_desc"
                        f" ({a['type']} / {b['type']}), grouped={same_grp}")

    groups = Counter(e["event_group_id"] for e in ev)
    multi = {g: n for g, n in groups.items() if n > 1}

    print("=== errors ===")
    print("\n".join(f"  {x}" for x in errors) or "  none")
    print("\n=== warnings ===")
    print("\n".join(f"  {x}" for x in warnings) or "  none")
    print("\n=== summary ===")
    print(f"  {len(ev)} events over {len(by_clip)}/{len(clips)} clips")
    pos = [e for e in ev if e["type"] != "pass_by"]
    print(f"  {len(pos)} positive, {len(ev)-len(pos)} pass_by negatives")
    print(f"  multi-event groups: {multi or 'none'}")
    print(f"  ambiguous: {sum(e['ambiguous'] for e in ev)}")
    unlabeled = set(clips) - set(by_clip)
    if unlabeled:
        print(f"  UNLABELED CLIPS: {sorted(unlabeled)}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
