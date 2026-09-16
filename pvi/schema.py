"""The JSON contract: interaction records, ground-truth events, and clip meta.

Two vocabularies live here and they are deliberately not the same one:

- `INTERACTION_TYPES` -- what the deliverable may contain.
- `GT_TYPES`          -- those plus `pass_by`, which exists only in the ground
  truth and the debug artifact. A `pass_by` in `outputs/<clip>.json` would be a
  bug: the deliverable reports interactions, and a rejected proposal is not one.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

INTERACTION_TYPES: tuple[str, ...] = (
    "enter_vehicle",
    "exit_vehicle",
    "open_close_door",
    "load_unload",
    "attend_vehicle",
)

NEGATIVE_TYPE = "pass_by"
GT_TYPES: tuple[str, ...] = INTERACTION_TYPES + (NEGATIVE_TYPE,)

# Reporting tiers forced by the realized GT counts (problem-definition.md s3).
# Tier 3 classes have n = 2, 2, 1; quoting a per-type rate for them would be
# noise dressed as a result, so metrics.py refuses to.
TIER2_TYPES: tuple[str, ...] = ("enter_vehicle", "exit_vehicle")
TIER3_TYPES: tuple[str, ...] = ("attend_vehicle", "open_close_door", "load_unload")


@dataclass(frozen=True)
class ClipMeta:
    clip_id: str
    width: int
    height: int
    fps: float
    n_frames: int

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps

    def to_s(self, frame: int) -> float:
        """Frame index -> seconds. Frame indices are 0-based throughout.

        Note data/frames/<clip>/ JPEGs are named from 000001.jpg, so frame 0 is
        000001.jpg. The pipeline does not read those files, but the ground truth
        was labeled against them and uses 0-based indices.
        """
        return frame / self.fps

    def to_frames(self, seconds: float) -> int:
        """Seconds -> whole frames, rounded. The single place a seconds-valued
        threshold becomes a frame count in rule logic."""
        return int(round(seconds * self.fps))

    @staticmethod
    def load_all(path: str | Path) -> dict[str, "ClipMeta"]:
        raw = json.loads(Path(path).read_text())
        return {
            k: ClipMeta(clip_id=v["clip_id"], width=v["width"], height=v["height"],
                        fps=v["fps"], n_frames=v["n_frames"])
            for k, v in raw.items()
        }


@dataclass(frozen=True)
class GTEvent:
    event_id: str
    event_group_id: str
    clip_id: str
    type: str
    frame_start: int
    frame_end: int
    time_start_s: float
    time_end_s: float
    person_desc: str = ""
    vehicle_desc: str = ""
    person_anchor: dict[str, Any] | None = None
    vehicle_anchor: dict[str, Any] | None = None
    ambiguous: bool = False
    note: str = ""

    @property
    def is_positive(self) -> bool:
        return self.type != NEGATIVE_TYPE


def load_ground_truth(path: str | Path) -> list[GTEvent]:
    """Load the frozen ground truth. Unknown types raise rather than pass through
    silently -- a typo'd type would otherwise vanish from every per-type count."""
    raw = json.loads(Path(path).read_text())
    events = []
    for e in raw["events"]:
        if e["type"] not in GT_TYPES:
            raise ValueError(f"{e['event_id']}: unknown type {e['type']!r}")
        events.append(GTEvent(**{k: e.get(k) for k in GTEvent.__dataclass_fields__
                                 if k in e}))
    return events


@dataclass
class PersonRef:
    track_id: int
    description: str = ""


@dataclass
class VehicleRef:
    track_id: int
    cls: str = ""
    description: str = ""


@dataclass
class Interaction:
    """One (person, vehicle, event) record -- the unit the task asks for.

    `evidence` is not required by the task. It is here because a reviewer's
    first question about any reported interaction is "why did it think that",
    and answering that should not require rerunning the pipeline.
    """
    interaction_id: str
    event_group_id: str
    type: str
    frame_start: int
    frame_end: int
    time_start_s: float
    time_end_s: float
    person: PersonRef
    vehicle: VehicleRef
    note: str = ""
    confidence: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.type not in INTERACTION_TYPES:
            raise ValueError(
                f"{self.type!r} is not a reportable interaction type. "
                f"{NEGATIVE_TYPE!r} belongs in the debug artifact, never the "
                "deliverable."
            )
        if self.frame_end < self.frame_start:
            raise ValueError(f"{self.interaction_id}: frame_end < frame_start")


def to_json(clip: ClipMeta, interactions: Iterable[Interaction],
            config_hash: str) -> dict[str, Any]:
    return {
        "clip_id": clip.clip_id,
        "clip_meta": {"fps": clip.fps, "n_frames": clip.n_frames,
                      "width": clip.width, "height": clip.height},
        "config_hash": config_hash,
        "interactions": [_norm(asdict(i)) for i in interactions],
    }


def _norm(d: dict[str, Any]) -> dict[str, Any]:
    # `cls` is a Python keyword-adjacent name; the JSON contract says "class".
    v = d.get("vehicle")
    if isinstance(v, dict) and "cls" in v:
        v["class"] = v.pop("cls")
    return d


def write_output(path: str | Path, clip: ClipMeta,
                 interactions: Iterable[Interaction], config_hash: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(to_json(clip, interactions, config_hash), indent=2) + "\n")
