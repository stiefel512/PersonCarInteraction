"""Pool the per-clip deliverables into the single artifact the task asks for.

The task requires "a machine-readable artifact that lists all person-vehicle
interactions in the given set of clips", with `clip_id` among the fields of
*each interaction*. The pipeline is per-clip by design -- clips are independent
(task statement: no inter-clip identity) and `pvi.cli` takes one clip in and
writes one JSON out -- so there `clip_id` sits at file level. This lifts it into
every record and concatenates.

Nothing is recomputed and nothing is re-judged. Where this file and the per-clip
files disagree, the per-clip files are right.

Usage:
    python3 tools/pool_outputs.py              # stdlib only; no venv needed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Field order within each pooled record. clip_id leads because it is the field
# the pooled artifact exists to add; the rest follows the per-clip file.
_LEAD = ("clip_id", "interaction_id", "event_group_id", "type",
         "frame_start", "frame_end", "time_start_s", "time_end_s",
         "person", "vehicle")


def pool(outputs: Path, clips: Path, allow_partial: bool = False) -> dict[str, Any]:
    """Concatenate outputs/<clip_id>.json into one artifact.

    Refuses two things rather than papering over them:

    - a missing clip, because a pooled artifact that silently covers 6 of 8
      clips reads as a complete answer for the set and is not one;
    - disagreeing `config_hash` values, because interactions produced under
      different thresholds are not one result and pooling them would erase the
      only evidence of that.
    """
    inventory = sorted(json.loads(clips.read_text()))

    missing = [c for c in inventory if not (outputs / f"{c}.json").exists()]
    if missing and not allow_partial:
        raise SystemExit(
            f"refusing to pool: no output for {', '.join(missing)}.\n"
            f"A pooled file covering {len(inventory) - len(missing)} of "
            f"{len(inventory)} clips reads as a complete answer for the set.\n"
            "Run the missing clips, or pass --allow-partial to record the gap "
            "in the artifact itself.")

    present = [c for c in inventory if c not in missing]
    hashes: dict[str, str] = {}
    clip_meta: dict[str, Any] = {}
    records: list[dict[str, Any]] = []

    for clip_id in present:
        raw = json.loads((outputs / f"{clip_id}.json").read_text())
        if raw["clip_id"] != clip_id:
            raise SystemExit(f"{clip_id}.json declares clip_id "
                             f"{raw['clip_id']!r}")
        hashes[clip_id] = raw["config_hash"]
        clip_meta[clip_id] = raw["clip_meta"]
        for rec in raw["interactions"]:
            merged = {"clip_id": clip_id, **rec}
            ordered = {k: merged[k] for k in _LEAD if k in merged}
            ordered.update({k: v for k, v in merged.items() if k not in ordered})
            records.append(ordered)

    distinct = sorted(set(hashes.values()))
    if len(distinct) > 1:
        detail = "\n".join(f"  {c}: {h}" for c, h in sorted(hashes.items()))
        raise SystemExit("refusing to pool: clips were produced under "
                         f"{len(distinct)} different configs.\n{detail}")

    # Deterministic order: by clip, then by when the interaction starts.
    records.sort(key=lambda r: (r["clip_id"], r["frame_start"],
                                r["interaction_id"]))

    artifact: dict[str, Any] = {
        "artifact": "person_vehicle_interactions",
        "config_hash": distinct[0] if distinct else None,
        "n_clips": len(present),
        "n_interactions": len(records),
        "clips": clip_meta,
        "interactions": records,
    }
    if missing:
        artifact["INCOMPLETE_missing_clips"] = missing
    return artifact


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", type=Path, default=Path("outputs"))
    ap.add_argument("--clips", type=Path, default=Path("data/clips.json"))
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <outputs>/interactions.json")
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()

    art = pool(args.outputs, args.clips, args.allow_partial)
    out = args.out or args.outputs / "interactions.json"
    out.write_text(json.dumps(art, indent=2) + "\n")

    by_type: dict[str, int] = {}
    for r in art["interactions"]:
        by_type[r["type"]] = by_type.get(r["type"], 0) + 1
    print(f"wrote {out}: {art['n_interactions']} interactions over "
          f"{art['n_clips']} clips")
    for t, n in sorted(by_type.items(), key=lambda kv: -kv[1]):
        print(f"  {t:18s} {n}")
    if art.get("INCOMPLETE_missing_clips"):
        print("  INCOMPLETE:", ", ".join(art["INCOMPLETE_missing_clips"]),
              file=sys.stderr)


if __name__ == "__main__":
    main()
